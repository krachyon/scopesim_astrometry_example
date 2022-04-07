import multiprocessing
import tempfile
from typing import Callable, Tuple, Optional
import anisocado
from image_registration.fft_tools import upsample_image
import scopesim
from scopesim_templates import stars
from photutils.centroids import centroid_quadratic
from astropy.table import Table
import appdirs
from pathlib import Path
from util import *


filter_name = 'MICADO/filters/TC_filter_K-cont.dat'

# generators should be able to run in parallel but scopesim tends to lock up on the initialization
scopesim_lock = multiprocessing.Lock()

COLUMN_NAMES = ('x', 'y','m', 'f')

WORKING_DIR = Path(appdirs.user_cache_dir('scopesim_workspace'))


def make_psf(psf_wavelength: float = 2.15,
             shift: Tuple[int] = (0, 14), N: int = 511,
             transform: Callable[[np.ndarray], np.ndarray] = lambda x: x) -> scopesim.effects.Effect:
    """
    create a psf effect for scopesim to be as close as possible to how an anisocado PSF is used in simcado
    :param psf_wavelength:
    :param shift:
    :param N: ? Size of kernel?
    :param transform: function to apply to the psf array
    :return: effect object you can plug into OpticalTrain
    """
    # this whole thing... There's bound to be a way cleaner alternative right?

    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [psf_wavelength], pixelSize=pixel_scale.value, N=N)
    image = hdus[2]
    image.data = np.squeeze(image.data)  # remove leading dimension, we're only looking at a single picture, not a stack

    # re-sample to shift center
    actual_center = np.array(centroid_quadratic(image.data, fit_boxsize=5))
    expected_center = np.array(center_of_image(image.data))
    xshift, yshift = expected_center - actual_center
    resampled = upsample_image(image.data, xshift=xshift, yshift=yshift).real
    image.data = resampled
    image.data = transform(image.data)

    filename = tempfile.NamedTemporaryFile('w', suffix='.fits').name
    image.writeto(filename)

    # noinspection PyTypeChecker
    tmp_psf = anisocado.AnalyticalScaoPsf(N=N, wavelength=psf_wavelength)
    strehl = tmp_psf.strehl_ratio

    # Todo: passing a filename that does not end in .fits causes a weird parsing error
    return scopesim.effects.FieldConstantPSF(
        name='my_anisocado_psf',
        filename=filename,
        wavelength=psf_wavelength,
        psf_side_length=N,
        strehl_ratio=strehl)


def setup_optical_train(psf_effect: Optional[scopesim.effects.Effect] = None,
                        custom_subpixel_psf: Optional[Callable] = None) -> scopesim.OpticalTrain:
    """
    Create a Micado optical train with custom PSF
    :return: OpticalTrain object
    """
    # So anyway, this function isn't really ideal and created by trial and error because I couldn't figure out a
    #  saner way to do it.

    assert not (psf_effect and custom_subpixel_psf), 'PSF effect can only be applied if custom_subpixel_psf is None'

    # TODO Multiprocessing sometimes seems to cause some issues in scopesim, probably due to shared connection object
    # #  File "ScopeSim/scopesim/effects/ter_curves.py", line 247, in query_server
    # #     tbl.columns[i].name = colname
    # #  UnboundLocalError: local variable 'tbl' referenced before assignment
    # mutexing this line seems to solve it...
    with scopesim_lock:
        micado = scopesim.OpticalTrain('MICADO')

    # ignore this exists
    if custom_subpixel_psf:
        micado.cmds["!SIM.sub_pixel.flag"] = "psf_eval"
        scopesim.rc.__currsys__['!SIM.sub_pixel.psf'] = custom_subpixel_psf

    else:
        micado.cmds["!SIM.sub_pixel.flag"] = True
        # the previous psf had that optical element so put it in the same spot.
        # Todo This way of looking up the index is pretty stupid. Is there a better way?
        element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')
        if not psf_effect:
            psf_effect = make_psf()

        micado.optics_manager.add_effect(psf_effect, ext=element_idx)


    # disable old psf
    # TODO - why is there no remove_effect with a similar interface?
    #  Why do I need to go through a dictionary attached to a different class?
    # TODO - would be nice if Effect Objects where frozen, e.g. with the dataclass decorator. Used ".included" first and
    # TODO   was annoyed that it wasn't working...
    micado['relay_psf'].include = False
    micado['micado_ncpas_psf'].include = False

    # TODO Apparently atmospheric dispersion is messed up. Ignore both dispersion and correction for now
    if 'armazones_atmo_dispersion' in micado.effects:
        micado['armazones_atmo_dispersion'].include = False
    micado['micado_adc_3D_shift'].include = False

    return micado


def download(to_directory=WORKING_DIR) -> None:
    """
    get scopesim files if not present in current directory
    :return: No
    """
    if not os.path.exists(to_directory):
        os.makedirs(to_directory)
    with work_in(to_directory):
        if not os.path.exists('./MICADO'):
            scopesim.download_package(["locations/Armazones",
                                       "telescopes/ELT",
                                       "instruments/MICADO"])


def scopesim_grid(N1d: int = 16,
                  seed: int = 1000,
                  border=64,
                  perturbation: float = 15.,
                  magnitude=lambda N: N * [18],
                  psf_transform=lambda x: x,
                  custom_subpixel_psf=None) \
        -> Tuple[np.ndarray, Table]:
    """
    Use scopesim to create a somewhat regular grid of stars because crowding is a pain to deal with
    :param N1d:  Grid of N1d x N1d Stars will be generated
    :param seed: initalize RNG for predictable results
    :param border: how many pixels on the edge to leave empty
    :param perturbation: perturb each star position with a uniform random pixel offset
    :return: image and input catalogue
    """
    # yes, I know i should use the random.rng interface...
    np.random.seed(seed)

    N = N1d ** 2
    spectral_types = ['A0V'] * N

    y = pixel_to_mas(np.tile(np.linspace(border, max_pixel_coord.value - border, N1d), reps=(N1d, 1)))
    x = y.T.copy()
    x += np.random.uniform(0, perturbation * pixel_scale.value, x.shape)
    y += np.random.uniform(0, perturbation * pixel_scale.value, y.shape)

    m = np.array(magnitude(N))

    source = stars(filter_name=filter_name,
                   amplitudes=m,
                   spec_types=spectral_types,
                   x=x.ravel(), y=y.ravel())
    # ignore this as well...
    if custom_subpixel_psf:
        detector = setup_optical_train(custom_subpixel_psf=custom_subpixel_psf)
    else:
        detector = setup_optical_train(psf_effect=make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    # TODO magnitude to flux gives really weird answers. Is there a sane way to predict the total flux in the detector pixels
    #  based on the magnitude you pass here?
    table = Table((to_pixel_scale(x).ravel(),
                   to_pixel_scale(y).ravel(),
                   magnitude_to_flux(m), m), names=COLUMN_NAMES)
    return observed_image, table


if __name__ == '__main__':
    download()
    # you can now do photometry/astrometry on the image and compare it against table

    # TODO it would be nice if the default of scopesim was to just use a common system-wide directory. If I modify
    #  a config file I'd rather point to it explicitly. Would also be cool to have a way to interactively edit the files
    #  that does not involve combing through the layers of rc dicts...
    with work_in(WORKING_DIR):
        image, table = scopesim_grid()

    # show result
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # discard overscan
    plt.imshow(image[10:-10, 10:-10], norm=LogNorm())
    plt.plot(table['x']-10, table['y']-10, 'rx', ms=3)
    plt.show()