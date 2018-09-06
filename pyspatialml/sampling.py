import os
import random
import numpy as np
import rasterio
import rasterio.features
import scipy
import geopandas
from shapely.geometry import Point
from rasterio.sample import sample_gen
from .__main import _maximum_dtype
from itertools import chain


def extract(dataset, response, field=None):
    """Sample a GDAL-supported raster dataset point of polygon
    features in a Geopandas Geodataframe or a labelled singleband
    raster dataset

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Opened Rasterio DatasetReader containing data to be sampled
        Raster can be a multi-band raster or a virtual tile format raster

    response: rasterio.io.DatasetReader or Geopandas DataFrame
        Single band raster containing labelled pixels, or
        a Geopandas GeoDataframe containing either point or polygon features

    field : str, optional
        Field name of attribute to be used the label the extracted data
        Used only if the response feature represents a GeoDataframe

    Returns
    -------
    X : array-like
        Numpy masked array of extracted raster values, typically 2d

    y: 1d array like
        Numpy masked array of labelled sampled

    xy: 2d array-like
        Numpy masked array of row and column indexes of training pixels"""

    src = dataset

    # extraction for geodataframes
    if isinstance(response, geopandas.geodataframe.GeoDataFrame):
        if len(response.geom_type.unique()) > 1:
            raise ValueError(
                'response_gdf cannot contain a mixture of geometry types')

        # polygon features
        if all(response.geom_type == 'Polygon'):

            # generator for shape geometry for rasterizing
            if field is None:
                shapes = (geom for geom in response.geometry)

            if field is not None:
                shapes = ((geom, value) for geom, value in zip(
                          response.geometry, response[field]))

            arr = np.zeros((src.height, src.width))
            arr[:] = -99999
            arr = rasterio.features.rasterize(
                shapes=shapes, fill=-99999, out=arr,
                transform=src.transform, default_value=1)

            # get indexes of labelled data
            rows, cols = np.nonzero(arr != -99999)

            # convert to coordinates and extract raster pixel values
            y = arr[rows, cols]
            xy = np.transpose(rasterio.transform.xy(src.transform, rows, cols))

        # point features
        elif all(response.geom_type == 'Point'):

            xy = response.bounds.iloc[:, 2:].values

            if field:
                y = response[field].values
            else:
                y = None

        # line features
        elif all(response.geom_type == 'LineString'):

            # interpolate points along lines
            pixel_points = []

            for i, p in response.iterrows():
                points_along_line = [p.geometry.interpolate(distance=i) for i in
                                     np.arange(0, p.geometry.length, min(src.res))]
                pixel_points.append(points_along_line)

            response = geopandas.GeoDataFrame(geometry=list(chain.from_iterable(pixel_points)),
                                              crs=src.crs)

            xy = response.bounds.iloc[:, 2:].values

            if field:
                y = response[field].values
            else:
                y = None

    # extraction for labelled pixels
    elif isinstance(response, rasterio.io.DatasetReader):

        # some checking
        if field is not None:
            Warning('field attribute is not used for response_raster')

        # open response raster and get labelled pixel indices and values
        arr = response.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)

        # extract values at labelled indices
        xy = np.transpose(rasterio.transform.xy(response.transform, rows, cols))
        y = arr.data[rows, cols]

    elif isinstance(response, np.ndarray):
        y = None
        xy = response

    # clip points to extent of raster
    extent = src.bounds

    valid_idx = np.where((xy[:, 0] > extent.left) &
                         (xy[:, 0] < extent.right) &
                         (xy[:, 1] > extent.bottom) &
                         (xy[:, 1] < extent.top))[0]

    xy = xy[valid_idx, :]

    if y is not None:
        y = y[valid_idx]

    # extract values at labelled indices
    X = extract_xy(xy, src)

    # summarize data and mask nodatavals in X, y, and xy
    if y is not None:
        y = np.ma.masked_where(X.mask.any(axis=1), y)

    Xmask_xy = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
    xy = np.ma.masked_where(Xmask_xy, xy)

    return X, y, xy


def extract_xy(xy, dataset):
    """Samples pixel values from a GDAL-supported raster dataset
    using an array of xy locations

    Parameters
    ----------
    xy : 2d array-like
        x and y coordinates from which to sample the raster (n_samples, xy)

    raster : str
        Path to GDAL supported raster

    Returns
    -------
    data : 2d array-like
        Masked array containing sampled raster values (sample, bands)
        at x,y locations"""

    src = dataset

    # read all bands if single dtype
    if src.dtypes.count(src.dtypes[0]) == len(src.dtypes):
        training_data = np.vstack([i for i in sample_gen(src, xy)])

    # read single bands if multiple dtypes
    else:
        dtype = _maximum_dtype(src)
        training_data = np.zeros((xy.shape[0], src.count), dtype=dtype)

        for band in range(src.count):
            training_data[:, band] = np.vstack(
                [i for i in sample_gen(src, xy, indexes=band+1)]).squeeze()

    training_data = np.ma.masked_equal(training_data, src.nodatavals)

    if isinstance(training_data.mask, np.bool_):
        mask_arr = np.empty(training_data.shape, dtype='bool')
        mask_arr[:] = False
        training_data.mask = mask_arr

    return training_data


def sample(size, dataset, strata=None, random_state=None):
    """Generates a random sample of according to size, and samples the pixel
    values from a GDAL-supported raster

    Parameters
    ----------
    size : int
        Number of random samples or number of samples per strata
        if strategy='stratified'

    dataset : rasterio.io.DatasetReader
        Opened Rasterio DatasetReader

    strata : rasterio.io.DatasetReader, optional (default=None
        To use stratified instead of random sampling, strata can be supplied
        using an open rasterio DatasetReader object

    random_state : int
        integer to use within random.seed

    Returns
    -------
    samples: array-like
        Numpy array of extracted raster values, typically 2d

    valid_coordinates: 2d array-like
        2D numpy array of xy coordinates of extracted values"""

    src = dataset

    # set the seed
    np.random.seed(seed=random_state)

    if strata is None:

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
            rows, cols = np.nonzero(np.isnan(sample_raster) == False)

            # convert row, col indices to coordinates
            xy = np.transpose(rasterio.transform.xy(src.transform, rows, cols))

            # sample at random point locations
            samples = extract_xy(xy, src)

            # append only non-masked data to each row of X_random
            samples = samples.astype('float32').filled(np.nan)
            invalid_ind = np.isnan(samples).any(axis=1)
            samples = samples[~invalid_ind, :]
            valid_samples = np.append(valid_samples, samples, axis=0)

            xy = xy[~invalid_ind, :]
            valid_coordinates = np.append(
                valid_coordinates, xy, axis=0)

            # check to see if target_nsamples has been reached
            if len(valid_samples) >= size:
                satisfied = True
            else:
                n = size - len(valid_samples)

    else:

        # get number of unique categories
        strata = strata.read(1)
        categories = np.unique(strata)
        categories = categories[np.nonzero(categories != src.nodata)]
        categories = categories[~np.isnan(categories)]

        # store selected coordinates
        selected = np.zeros((0, 2))

        for cat in categories:

            # get row,col positions for cat strata
            ind = np.transpose(np.nonzero(strata == cat))

            if size > ind.shape[0]:
                msg = 'Sample size is greater than number of pixels in strata {0}'.format(str(ind))
                msg = os.linesep.join([msg, 'Sampling using replacement'])
                Warning(msg)

            # random sample
            sample = np.random.uniform(
                low=0, high=ind.shape[0], size=size).astype('int')
            xy = ind[sample, :]

            selected = np.append(selected, xy, axis=0)

        # convert row, col indices to coordinates
        x, y = rasterio.transform.xy(
            src.transform, selected[:, 0], selected[:, 1])
        valid_coordinates = np.stack((x, y)).transpose()

        # extract data
        valid_samples = extract_xy(valid_coordinates, src)

    return valid_samples, valid_coordinates


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
