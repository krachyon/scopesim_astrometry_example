"""
Each of
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from astropy.nddata import NDData
from astropy.table import Table
from matplotlib.colors import LogNorm
from photutils import IRAFStarFinder, BasicPSFPhotometry, DAOGroup, MMMBackground, FittableImageModel, extract_stars
from photutils.psf.incremental_fit_photometry import IncrementalFitPhotometry, make_groups, FitStage, \
    all_simultaneous, all_individual, brightest_simultaneous, not_brightest_individual
from photutils.psf.median_epsf_builder import make_epsf_combine
from scipy.spatial import ConvexHull

from scopesim_generate_example import OUTPUT_NAME
from util import centered_grid

image_path = OUTPUT_NAME.with_suffix('.fits')
table_path = OUTPUT_NAME.with_suffix('.dat')

image = fits.getdata(image_path)
table = Table.read(table_path, format='ascii.ecsv')


def make_fake_epsf():
    # Just evaluate a Gaussian for a rough PSFy shape.
    # again, results won't be very good, but illustrates usage.
    y_grid, x_grid = centered_grid((11, 11))
    fake_epsf = FittableImageModel(Gaussian2D(x_stddev=4, y_stddev=4)(x_grid, y_grid), oversampling=1)
    return fake_epsf


def make_guess_table():
    guess_table = table.copy()
    guess_table.rename_columns(['x', 'y'], ['x_0', 'y_0'])
    # perturb initial guess a bit for the fitter to do something
    guess_table['x_0'] += np.random.uniform(-0.1, 0.1, len(guess_table))
    guess_table['y_0'] += np.random.uniform(-0.1, 0.1, len(guess_table))
    guess_table['flux_0'] = 300  # The output from scopesim is not easily converted to ADU
    return guess_table


def non_uniform_threshold():
    """Let's pretend we're in the source detection step and we have a noise map of our image
    This metric here isn't really that great (threshold is literary the pixel value above which a peak is found, not
    in units of standard deviation) but illustrates the usage.
    """
    # take everything about one sigma above the median of the image, assuming pure Poisson noise
    # this is an array with shape==image.shape
    threshold_array = np.median(image) + np.sqrt(image)

    # This is useless but you could in theory bring your totally handmade array
    stupid_threshold = np.random.random(image.shape)

    finder = IRAFStarFinder(threshold=threshold_array, fwhm=3, exclude_border=True)

    detection_table = finder(image)
    return detection_table


def fit_with_bounds():
    """Showcasing the changes made to the photomtery classes to enable using a bounded fit"""
    guess_table = make_guess_table()

    # This means the following: take the current parameter value and subtract lower to get
    # the lower bound, add upper to get the upper bound.
    # it would probably be more intuitive to have the user add the minus-sign themselves to the lower...
    # also there should probably be a way to put in an absolute value like 0 for flux...
    # in this case here, if we know the disturbance, we can be sure to not be off by more than +-0.1 px
    bounds = {'x_0': (0.1, 0.1), 'y_0': (0.1, 0.1), 'flux_0': (3000, 1e10)}

    fake_epsf = make_fake_epsf()

    phot = BasicPSFPhotometry(psf_model=fake_epsf,
                              group_maker=DAOGroup(10),
                              bkg_estimator=MMMBackground(),
                              bounds=bounds,
                              fitshape=(7, 7),
                              fitter=TRFLSQFitter())
    phot_table = phot(image, init_guesses=guess_table)
    return phot_table


def custom_epsf_build():
    """Shows using the FFT-resample and median-based EPSF building function"""
    # size is a lot larger than you would typically use for real data
    stars = extract_stars(NDData(image), catalogs=table, size=(31, 31))
    epsf = make_epsf_combine(stars)
    # maxiter>1 adds a recentering step. I don't really know if that is a good idea. Here it's not, because we have
    # the exact locations already in the table. But feel free to disturb it like in the function above and see what happens
    epsf_recentered = make_epsf_combine(stars, maxiter=2)
    return epsf, epsf_recentered


def visualize_custom_grouping(max_size=10, halo_radius=60):
    """This is an example of how to check what the custom grouping algorithm does.
    Core members of the group share a color, the polygon is indicative of which sources are included in
    a group's halo. It is not very pretty on this image/catalog, so try generating your own!"""

    target_table = table.copy()
    target_table.rename_columns(['x', 'y'], ['x_fit', 'y_fit'])
    target_table['update'] = True
    groups = make_groups(target_table, max_size=max_size, halo_radius=halo_radius)

    group_ids = np.sort(np.unique([i for group in groups for i in group['group_id']]))
    max_id = np.max(group_ids)

    plt.figure()
    plt.imshow(image, norm=LogNorm())
    for group in groups:
        core_group = group[group['update'] == True]
        group_id = core_group['group_id'][0]

        cmap = plt.get_cmap('prism')

        xy_curr = np.array((group['x_fit'], group['y_fit'])).T

        if len(group) >= 3:
            hull = ConvexHull(xy_curr)
            vertices = xy_curr[hull.vertices]
            poly = plt.Polygon(vertices, fill=False, color=cmap(group_id / max_id))
            plt.gca().add_patch(poly)

        plt.scatter(core_group['x_fit'], core_group['y_fit'], color=cmap(group_id / max_id), s=8)

        plt.annotate(group_id, (np.mean(core_group['x_fit']), np.mean(core_group['y_fit'])),
                     color='white', backgroundcolor=(0, 0, 0, 0.25), alpha=0.7)


def incremental_fit_photometry():
    pass
    # the EPSF /needs/ to be an FittableImageModel (not EPSFmodel) due to how the interpolator is extracted.
    epsf = make_fake_epsf()

    # The sequence of fit stages is the
    # bounds are symmetric += around the current value of the parameter

    # this is closest to what photutils does normally
    fit_stages_like_photutils = [
        FitStage(fitshape=1, xbound=None, ybound=None, fluxbound=None, row_generator=all_simultaneous)
    ]

    # This one is useful as a first pass to coarsely estimate the flux without reverse-engineering ScopeSim
    fit_stages_scopesim = [
        FitStage(fitshape=1, xbound=1e-8, ybound=1e-8, fluxbound=None, row_generator=all_individual),
        # add more here for a real analysis
    ]

    # This should be pretty close to an iterative subtraction approach. Made up bounds,
    # needs to be replaced with sensible avalues for given problem
    fit_stages_iterative_subtraction = [
        FitStage(fitshape=1, xbound=1, ybound=1, fluxbound=2000, row_generator=all_individual),
        FitStage(fitshape=1, xbound=0.5, ybound=0.5, fluxbound=1000, row_generator=all_individual),
        FitStage(fitshape=1, xbound=0.25, ybound=0.25, fluxbound=500, row_generator=all_individual),
        FitStage(fitshape=1, xbound=0.1, ybound=0.1, fluxbound=200, row_generator=all_individual),

        # add more here for a real analysis
    ]

    # Something like this seemed to work well in practice on ScopeSim data with crowding. But it's overkill
    # here and really slow.
    fit_stages_overkill = [
        FitStage(1, 1e-11, 1e-11, np.inf, all_individual),
        # first stage: get flux approximately right
        FitStage(1, 0.6, 0.6, 200_000, brightest_simultaneous(2)),
        # only two brightest sources of group
        FitStage(1, 0.6, 0.6, 100_000, not_brightest_individual(2)),  # rest gets optimized
        FitStage(1, 0.3, 0.3, 100_000, brightest_simultaneous(4)),
        FitStage(1, 0.3, 0.3, 50_000, not_brightest_individual(4)),
        FitStage(20, 0.2, 0.2, 20_000, all_individual),
        FitStage(20, 0.1, 0.1, 5000, all_simultaneous)
    ]

    fit_stages_grid = [
        FitStage(5, 1e-10, 1e-11, np.inf, all_individual),  # first stage: get flux approximately right
        FitStage(5, 0.6, 0.6, 10, all_individual),
        FitStage(5, 0.3, 0.3, 500_000, all_individual)
    ]
    # This class only works if you have a valid guess_table, no attempt will be made at finding additional sources
    photometry = IncrementalFitPhotometry(bkg_estimator=MMMBackground(),
                                         psf_model=epsf,
                                         fit_stages=fit_stages_grid,
                                         max_group_size=6,
                                         group_extension_radius=40)
    phot_table = photometry(image, init_guesses=make_guess_table())
    return phot_table


if __name__ == '__main__':
    print(non_uniform_threshold())
    print(fit_with_bounds())

    epsf, epsf_recentered = custom_epsf_build()
    plt.imshow(epsf.data, norm=LogNorm())
    plt.title('epsf no recentering')
    plt.figure()
    plt.imshow(epsf_recentered.data, norm=LogNorm())
    plt.title('epsf one re-centering step')
    visualize_custom_grouping()
    print(incremental_fit_photometry())
    plt.show()
