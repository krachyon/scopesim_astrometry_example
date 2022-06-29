from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from matplotlib.colors import LogNorm
import util
from scopesim_generate_example import OUTPUT_NAME


def getdata_safer(filename, *args, **kwargs):
    """
    Wrapper for astropy.io.fits.getdata to coerce data into float64 with native byteorder.
    FITS are default big-endian which can cause weird stuff to happen as numpy stores it as-is in memory.
    Especially if you use `bottleneck`
    :param filename: what to read
    :param args: further args to fits.getdata
    :param kwargs: further keyword-args to fits.getdata
    :return: the read data
    """
    import astropy.io.fits
    data = astropy.io.fits.getdata(filename, *args, **kwargs) \
        .astype(np.float64, order='C', copy=False)

    assert data.dtype.byteorder == '='
    assert data.flags.c_contiguous

    return data


def estimate_fwhm(psf: photutils.psf.EPSFModel) -> float:
    """
    Use a 2D symmetric gaussian fit to estimate the FWHM of an empirical psf
    :param psf: psfmodel to estimate
    :return: FWHM in pixel coordinates, takes into account oversampling parameter of EPSF
    """
    from astropy.modeling import fitting
    from astropy.modeling.functional_models import Gaussian2D

    # Not sure if this would work for non-quadratic images
    assert (psf.data.shape[0] == psf.data.shape[1])
    assert (psf.oversampling[0] == psf.oversampling[1])
    dim = psf.data.shape[0]
    center = int(dim / 2)
    gauss_in = Gaussian2D(x_mean=center, y_mean=center, x_stddev=5, y_stddev=5)

    # force a symmetric gaussian
    gauss_in.y_stddev.tied = lambda model: model.x_stddev

    x, y = np.mgrid[:dim, :dim]
    gauss_out = fitting.LevMarLSQFitter()(gauss_in, x, y, psf.data)

    # have to divide by oversampling to get back to original scale
    return gauss_out.x_fwhm / psf.oversampling[0]


def astrometry(image: np.ndarray,
               reference_table: Optional[Table] = None,
               known_psf: Optional[photutils.EPSFModel] = None):
    """
    All the steps necessary to do basic PSF astrometry with photutils. You may want to pull this function apart
    and check the output from the individual steps
    :param image:
    :param reference_table:
    :param known_psf:
    :return:
    """
    if known_psf:
        fwhm = estimate_fwhm(known_psf)
    else:
        fwhm = fwhm_guess

    # get image stats and build finder
    mean, median, std = sigma_clipped_stats(image)
    initial_finder = photutils.IRAFStarFinder(threshold=median * threshold_factor,
                                                 fwhm=fwhm * fwhm_factor,
                                                 sigma_radius=sigma_radius,
                                                 minsep_fwhm=separation_factor,
                                                 brightest=n_brightest,
                                                 peakmax=peakmax)

    if reference_table:
        stars_tbl = reference_table.copy()
    else:
        stars_tbl = initial_finder(image)
        stars_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    # extract star cutouts and fit EPSF from them
    image_no_background = image - median
    stars = photutils.extract_stars(NDData(image_no_background), stars_tbl, size=cutout_size)
    initial_epsf, fitted_stars = photutils.EPSFBuilder(oversampling=oversampling,
                                               maxiters=epsf_iters,
                                               progress_bar=True,
                                               smoothing_kernel=smoothing_kernel).build_epsf(stars)

    # Use PSF to hopefully get better detctions from the image
    # We need to build a convolution-kernel by evaluating the EPSF
    y, x = util.centered_grid(np.array(initial_epsf.data.shape)/initial_epsf.oversampling)
    kernel = initial_epsf(x, y)
    epsf_finder = photutils.StarFinder(threshold=median * threshold_factor,
                                       kernel=kernel,
                                       min_separation=separation_factor*fwhm,  # because why not use different units...
                                       brightest=n_brightest,
                                       peakmax=peakmax)

    stars_tbl_refined = epsf_finder(image)
    stars_tbl_refined.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    grouper = photutils.DAOGroup(group_radius * fwhm)
    stars_refined = photutils.extract_stars(NDData(image_no_background), stars_tbl_refined, size=cutout_size)

    epsf, fitted_stars = photutils.EPSFBuilder(oversampling=oversampling,
                                               maxiters=epsf_iters,
                                               progress_bar=True,
                                               smoothing_kernel=smoothing_kernel).build_epsf(stars_refined)

    if reference_table:
        # You already have a decent initial guess for the detections. You may want to skip all detection steps above
        # and derive an EPSF from the initial guesses.
        photometry = photutils.BasicPSFPhotometry(
            group_maker=grouper,
            finder=None,
            bkg_estimator=photutils.MMMBackground(),  # Don't really know what kind of background estimator is preferred
            aperture_radius=fwhm,
            fitshape=fitshape,
            psf_model=epsf)
        stars_tbl.rename_columns(['x', 'y'], ['x_0', 'y_0'])
        size = len(stars_tbl)
        # randomly perturb guesses to make fit actually do something. It tends to not modify the initial guesses...
        stars_tbl['x_0'] += np.random.uniform(0.1, 0.2, size) * np.random.choice([-1, 1], size)
        stars_tbl['y_0'] += np.random.uniform(0.1, 0.2, size) * np.random.choice([-1, 1], size)

        result = photometry(image, init_guesses=stars_tbl)
    else:
        ## Ideally you would do something like this for the final astrometric step. However I have not found a way
        ## to make the iterative subtraction do anything useful, as it uses the same finder with fixed
        ## threshold in each iteration.
        # photometry = photutils.IterativelySubtractedPSFPhotometry(
        #     group_maker=grouper,
        #     finder=epsf_finder,
        #     bkg_estimator=photutils.MMMBackground(),
        #     aperture_radius=fwhm,
        #     fitshape=fitshape,
        #     psf_model=epsf,
        #     niters=photometry_iters
        # )

        photometry = photutils.BasicPSFPhotometry(
            group_maker=grouper,
            finder=epsf_finder,  # You could build the finder for the third time with the second EPSF...
            bkg_estimator=photutils.MMMBackground(),  # Don't really know what kind of background estimator is preferred
            aperture_radius=fwhm,
            fitshape=fitshape,
            psf_model=epsf)

        result = photometry(image)

    return result


# Parameters go here
known_psf_oversampling = 4  # for precomputed psf
known_psf_size = 201  # dito

# photometry
photometry_iters = 3
fitshape = 11

# epsf_fit
cutout_size = 21
smoothing_kernel = 'quadratic'  # can be quadratic, quartic or custom array e.g. with util.make_gauss_kernel
oversampling = 2
epsf_iters = 5

# grouper
group_radius = 1.5  # in fhwm

# starfinder
fwhm_guess = 3.
threshold_factor = 1  # img_median*this
fwhm_factor = 1.  # fwhm*this
separation_factor = 3  #again in units of fwhm

# honestly I don't really know what this does, seems there's some truncation of a enhancement kernel...
# has a decent effect on the outcome though...
sigma_radius = 1.5
# use only n stars for epsf determination/each photometry iteration. usefull in iteratively subtracted photometry
n_brightest = 2000
minsep_fwhm = 0.3  # only find stars at least this*fwhm apart
peakmax = 100_000  # only find stars below this pixel value to avoid saturated stars in photometry


image_name = OUTPUT_NAME.with_suffix('.fits')  # path to image you want to analyze
# End parameters

if __name__ == '__main__':
    image_data = getdata_safer(image_name)
    reference_table = None  # read reference here if desired
    known_psf = None  # provide a-priori epsf if known

    result_table = astrometry(image_data, reference_table, known_psf)

    # discard edges -> overscan uglifies colormap
    edge = 10
    plt.imshow(image_data[edge:-edge, edge:-edge], cmap='inferno', norm=LogNorm())
    plt.plot(result_table['x_fit']-edge, result_table['y_fit']-edge, 'gx', markersize=5)
    plt.show()
