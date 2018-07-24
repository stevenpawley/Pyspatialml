import os
import random
import numpy as np
import rasterio
import scipy
from shapely.geometry import Point


def random_sample(size, dataset, random_state=None):
    """Generates a random sample of according to size, and samples the pixel
    values from a GDAL-supported raster

    Parameters
    ----------
    size : int
        Number of random samples to obtain

    dataset : rasterio.io.DatasetReader
        Opened Rasterio DatasetReader

    random_state : int
        integer to use within random.seed

    Returns
    -------
    valid_samples: array-like
        Numpy array of extracted raster values, typically 2d

    valid_coordinates: 2d array-like
        2D numpy array of xy coordinates of extracted values"""

    src = dataset

    # set the seed
    np.random.seed(seed=random_state)

    # create np array to store randomly sampled data
    # we are starting with zero initial rows because data will be appended,
    # and number of columns are equal to n_features
    valid_samples = np.zeros((0, src.count))
    valid_coordinates = np.zeros((0, 2))

    # loop until target number of samples is satified
    satisfied = False

    n = size
    while satisfied is False:

        # generate random row and column indices
        Xsample = np.random.choice(range(0, src.shape[1]), n)
        Ysample = np.random.choice(range(0, src.shape[0]), n)

        # create 2d numpy array with sample locations set to 1
        sample_raster = np.empty((src.shape[0], src.shape[1]))
        sample_raster[:] = np.nan
        sample_raster[Ysample, Xsample] = 1

        # get indices of sample locations
        is_train = np.nonzero(np.isnan(sample_raster) == False)

        # sample at random point locations
        samples = src.read(masked=True)[:, is_train[0], is_train[1]]

        # append only non-masked data to each row of X_random
        samples = samples.filled(np.nan)
        invalid_ind = np.isnan(samples).any(axis=0)
        samples = samples[:, ~invalid_ind]
        valid_samples = np.append(valid_samples, np.transpose(samples), axis=0)

        is_train = np.array(is_train)
        is_train = is_train[:, ~invalid_ind]
        valid_coordinates = np.append(
            valid_coordinates, is_train.T, axis=0)

        # check to see if target_nsamples has been reached
        if len(valid_samples) >= size:
            satisfied = True
        else:
            n = size - len(valid_samples)

    return valid_samples, valid_coordinates


def stratified_sample(dataset, n, window=None, crds=False):
    """Creates random points of [size] within each category of a
    GDAL-supported raster

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Opened Rasterio DatasetReader raster that contains the strata (categories)
        to generate n random points from

    n : int
        number of random points to generate within each strata

    window : tuple, optional
        Optionally restrict sampling to this area of the stratified raster
        as a tuple ((xmin, xmax), (ymin, ymax)) of row,col positions

    crds : boolean, optional
        Optionally return the randomly sampled locations as coordinates based
        on the transform of the stratified raster, otherwise return row,col
        positions

    Returns
    -------
    selected : 2d array-like
        row and column positions representing stratified random sample
        locations"""

    src = dataset

    # get number of unique categories
    strata = src.read(1, window=window)
    categories = np.unique(strata)
    categories = categories[np.nonzero(categories != src.nodata)]
    categories = categories[~np.isnan(categories)]

    # store selected coordinates
    selected = np.zeros((0, 2))

    for cat in categories:

        # get row,col positions for cat strata
        ind = np.transpose(np.nonzero(strata == cat))

        if n > ind.shape[0]:
            msg = ('Sample size is greater than number of pixels in ',
                   'strata {0}'.format(str(ind)))
            msg = os.linesep.join([msg, 'Sampling using replacement'])
            Warning(msg)

        # random sample
        sample = np.random.uniform(
            low=0, high=ind.shape[0], size=n).astype('int')
        # sample = np.random.choice(range(0, ind.shape[0]), n)
        xy = ind[sample, :]

        # convert to coordinates
        # xs, ys = rasterio.transform.xy(src.transform, xs, ys)
        selected = np.append(selected, xy, axis=0)

    if crds is True:
        selected = rasterio.transform.xy(
            src.transform, selected[:, 0], selected[:, 1])

    return selected


def filter_points(xy, min_dist=0, remove='first'):
    """Filter points in geodataframe using a minimum distance buffer

    Parameters
    ----------
    xy : 2d array-like
        Numpy array containing point locations (n_samples, xy)

    min_dist : int or float, optional (default=0)
        Minimum distance by which to filter out closely spaced points

    remove : str, optional (default='first')
        Optionally choose to remove 'first' occurrences or 'last' occurrences

    Returns
    -------
    xy : 2d array-like
        Numpy array filtered coordinates"""

    dm = scipy.spatial.distance_matrix(xy, xy)
    np.fill_diagonal(dm, np.nan)

    if remove == 'first':
        dm[np.tril_indices(dm.shape[0], -1)] = np.nan
    elif remove == 'last':
        dm[np.triu_indices(dm.shape[0], -1)] = np.nan

    d = np.nanmin(dm, axis=1)

    return xy[np.greater_equal(d, min_dist)]


def get_random_point_in_polygon(poly):
    """Generates random shapely Point geometry objects within a single
    shapely Polygon object

    Parameters
    ----------
    poly : Shapely Polygon object

    Returns
    -------
    p : Shapely Point object"""

    (minx, miny, maxx, maxy) = poly.bounds

    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))

        if poly.contains(p):
            return p
