import numpy as np

from sklearn import mixture
from numpy.random import default_rng


def reduce_palette(image: np.ndarray, n_colours: int = 4, spread: float = 0):
    """
    Discretises the RGB palette of an numpy array image to n_colours GMM clusters.
    Each pixel is repositioned by the three dimensional parametric line equation:
        p_new = spread * p_0 + (1 - spread) * c_0
    where p_0 is the original RGB coordinates of the pixel and c_0 is the coordinates of the
    centre of the pixel's cluster.

    :param image: MxNx3 numpy array representing an image
    :param n_colours: Number of clusters in the RGB space.
    :param spread: Amount of give for each pixel from its cluster centre.
    :return: The altered image.
    """

    centres, labels = get_palette(image, n_colours)

    new_image = np.clip(spread * image + (1 - spread) * centres[labels], 0, 255).astype(int)

    return new_image


def change_palette(image: np.ndarray, new_palette: np.ndarray, spread: float = 0):
    """
    Replaces the palette (modal RGB colours) of image with a new palette.

    :param image: MxNx3 numpy array representing an image
    :param new_palette: Kx3 numpy array replacement palette.
    :param spread: Amount of give for each pixel from its cluster centre.
    :return: The altered image.
    """
    n_colours = new_palette.shape[0]

    centres, labels = get_palette(image, n_colours)

    new_image = spread * (image - centres[labels]) + new_palette[labels]

    return new_image


def get_palette(image: np.ndarray, n_colours: int = 4):
    """
    Applies a GMM to the RGB pixels of an image to find the n_colours modes of the image's colour space.
    Returns the palette (modes) and the mode corresponding to each pixel.

    :param image: MxNx3 numpy array representing an image
    :param n_colours: Number of clusters in the RGB space.
    :return: The palette (modes) and the mode corresponding to each pixel.
    """

    image_flat = image.reshape(-1, 3)

    # GMM is fitted with a sample of 1000 pixels improve speed.
    image_sample = image_flat[default_rng().choice(image_flat.shape[0], 1000, replace=False)]

    gmm = mixture.GaussianMixture(n_components=n_colours)
    gmm.fit(image_sample)

    centres = gmm.means_
    labels = gmm.predict(image_flat).reshape(image.shape[:2])

    return centres, labels

