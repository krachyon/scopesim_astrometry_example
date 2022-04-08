import contextlib
import os
import pathlib
from typing import Union

import astropy.units as u
import numpy as np


@np.vectorize
def center_of_index(length: int) -> float:
    """given an array with extent length, what index hits the center of the array?"""
    return (length - 1) / 2


def center_of_image(img: np.ndarray) -> tuple[float, float]:
    """in pixel coordinates, pixel center convention

    (snippet to verify the numpy convention)
    img=np.random.randint(1,10,(10,10))
    y,x = np.indices(img.shape)
    imshow(img)
    plot(x.flatten(),y.flatten(),'ro')
    """
    assert len(img.shape) == 2
    # xcenter, ycenter
    return tuple(center_of_index(img.shape)[::-1])


@contextlib.contextmanager
def work_in(path: Union[str, pathlib.Path]):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    LICENSE: MIT
    from: https://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


pixel_count = 1024 * u.pixel
pixel_scale = 0.004 * u.arcsec / u.pixel

max_pixel_coord = pixel_count - 1 * u.pixel  # size 1024 to max index 1023


# Writing these is probably responsible for a significant number of grey hairs on my part.
#  And it still doesn't make sense to me. It would really help if scopesim was unit-aware
#  or offered the conversion between angular and pixel scale somehow
def to_pixel_scale(as_coord):
    """
    convert position of objects from arcseconds to pixel coordinates
    Numpy/photutils center convention
    """
    if hasattr(as_coord, 'unit'):
        if as_coord.unit is None:
            as_coord *= u.arcsec
        else:
            as_coord = as_coord.to(u.arcsec)
    else:
        as_coord *= u.arcsec

    shifted_pixel_coord = as_coord / pixel_scale
    # FIXME the -0.5 pixel are a fudge factor for scopesim. IMHO center of image should be at 0,0 as
    #  but it's offset
    pixel = shifted_pixel_coord + max_pixel_coord / 2 - 0.5 * u.pixel
    return pixel.to(u.pixel).value


def pixel_to_mas(px_coord):
    """
    convert position of objects from pixel coordinates to arcseconds
    Numpy/photutils center convention
    """
    if not isinstance(px_coord, u.Quantity):
        px_coord *= u.pixel

    # shift bounds (0,1023) to (-511.5,511.5)
    # FIXME the -0.5 pixel are a fudge factor for scopesim. IMHO center of image should be at 0,0 as
    #  but it's offset by 0.5 pixel..
    # I have no idea if this works like this anymore or the offset has changed...
    coord_shifted = px_coord - max_pixel_coord / 2 + 0.5 * u.pixel
    mas = coord_shifted * pixel_scale
    return mas.value


@np.vectorize
def flux_to_magnitude(flux):
    return -2.5 * np.log10(flux)


@np.vectorize
def magnitude_to_flux(magnitude):
    return 10 ** (-magnitude / 2.5)
