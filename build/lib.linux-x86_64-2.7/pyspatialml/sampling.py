#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import rasterio
from shapely.geometry import Point
import random
import os


#def extract(raster, response_raster=None, response_gdf=None, field=None):
#    """Samples a list of GDAL rasters using a labelled numpy array
#
#    Parameters
#    ----------
#    raster : str
#        Path to GDAL supported raster containing data to be sampled.
#        Raster can be a multi-band image or a virtual tile format raster
#    response_raster : str, optional
#        Path to GDAL supported raster containing labelled pixels.
#        The raster should contain only a single band
#    response_gdf : Geopandas DataFrame, optional
#        GeoDataFrame where the GeoSeries is assumed to consist entirely of
#        Polygon feature class types
#    field : str, optional
#        Field name of attribute to be used the label the extracted data
#
#    Returns
#    -------
#    X : array-like
#        Numpy masked array of extracted raster values, typically 2d
#    y: 1d array like
#        Numpy masked array of labelled sampled
#    xy: 2d array-like
#        Numpy masked array of row and column indexes of training pixels"""
#
#    # some checking
#    if response_raster is None and response_gdf is None:
#        raise ValueError(
#            'Either the response_raster or response_gdf ' +
#            'parameters need arguments')
#    if response_raster and response_gdf:
#        raise ValueError(
#            'response_raster and response_gdf are mutually exclusive. ' +
#            'Specify only one as an arguments')
#
#    # extraction for geodataframes
#    if response_gdf is not None:
#        if len(np.unique(response_gdf.geom_type)) > 1:
#            raise ValueError(
#                'response_gdf cannot contain a mixture of geometry types')
#
#        # polygon features
#        if all(response_gdf.geom_type == 'Polygon'):
#            src = rasterio.open(raster)
#
#            # generator for shape geometry for rasterizing
#            if field is None:
#                shapes = (geom for geom in response_gdf.geometry)
#
#            if field is not None:
#                shapes = ((geom, value) for geom, value in zip(
#                          response_gdf.geometry, response_gdf[field]))
#
#            arr = np.zeros((src.height, src.width))
#            arr[:] = -99999
#            arr = rasterio.features.rasterize(
#                shapes=shapes, fill=-99999, out=arr,
#                transform=src.transform, default_value=1)
#
#            # get indexes of labelled data
#            rows, cols = np.nonzero(arr != -99999)
#
#            # convert to coordinates and extract raster pixel values
#            y = arr[rows, cols]
#            xy = np.transpose(rasterio.transform.xy(src.transform, rows, cols))
#            X = __extract_points(xy, raster)
#
#            src.close()
#
#        # point features
#        elif all(response_gdf.geom_type == 'Point'):
#            xy = response_gdf.bounds.iloc[:, 2:].as_matrix()
#            if field:
#                y = response_gdf[field].as_matrix()
#            else:
#                y = None
#            X = __extract_points(xy, y, raster)
#
#    # extraction for labelled pixels
#    elif response_raster is not None:
#        # some checking
#        if field is not None:
#            Warning('field attribute is not used for response_raster')
#
#        # open response raster and get labelled pixel indices and values
#        src = rasterio.open(response_raster)
#        arr = src.read(1, masked=True)
#        rows, cols = np.nonzero(~arr.mask)
#
#        # extract values at labelled indices
#        xy = np.transpose(rasterio.transform.xy(src.transform, rows, cols))
#        X = __extract_points(xy, raster)
#        y = arr.data[rows, cols]
#
#        src.close()
#
#    # summarize data and mask nodatavals in X, y, and xy
#    y = np.ma.masked_where(X.mask, y)
#    xy = np.ma.masked_where(X.mask, xy)
#
#    return(X, y, xy)


def extract(raster, response_raster=None, response_gdf=None, field=None):
    """Samples a list of GDAL rasters using a labelled numpy array

    Parameters
    ----------
    raster : str
        Path to GDAL supported raster containing data to be sampled.
        Raster can be a multi-band image or a virtual tile format raster
    response_raster : str, optional
        Path to GDAL supported raster containing labelled pixels.
        The raster should contain only a single band
    response_gdf : Geopandas DataFrame, optional
        GeoDataFrame where the GeoSeries is assumed to consist entirely of
        Polygon feature class types
    field : str, optional
        Field name of attribute to be used the label the extracted data

    Returns
    -------
    X : array-like
        Numpy masked array of extracted raster values, typically 2d
    y: 1d array like
        Numpy masked array of labelled sampled
    xy: 2d array-like
        Numpy masked array of row and column indexes of training pixels"""

    # some checking
    if response_raster is None and response_gdf is None:
        raise ValueError(
            'Either the response_raster or response_gdf ' +
            'parameters need arguments')
    if response_raster and response_gdf:
        raise ValueError(
            'response_raster and response_gdf are mutually exclusive. ' +
            'Specify only one as an arguments')

    # extraction for geodataframes
    if response_gdf is not None:
        if len(np.unique(response_gdf.geom_type)) > 1:
            raise ValueError(
                'response_gdf cannot contain a mixture of geometry types')

        # polygon features
        if all(response_gdf.geom_type == 'Polygon'):
            src = rasterio.open(raster)

            # generator for shape geometry for rasterizing
            if field is None:
                shapes = (geom for geom in response_gdf.geometry)

            if field is not None:
                shapes = ((geom, value) for geom, value in zip(
                          response_gdf.geometry, response_gdf[field]))

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

            src.close()

        # point features
        elif all(response_gdf.geom_type == 'Point'):
            xy = response_gdf.bounds.iloc[:, 2:].as_matrix()
            if field:
                y = response_gdf[field].as_matrix()
            else:
                y = None

    # extraction for labelled pixels
    elif response_raster is not None:
        # some checking
        if field is not None:
            Warning('field attribute is not used for response_raster')

        # open response raster and get labelled pixel indices and values
        src = rasterio.open(response_raster)
        arr = src.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)

        # extract values at labelled indices
        xy = np.transpose(rasterio.transform.xy(src.transform, rows, cols))
        y = arr.data[rows, cols]

        src.close()

    return(__extract_points_raster(xy, y, raster))


def __extract_points(xy, raster):
    """Samples a list of GDAL rasters using a xy locations

    Parameters
    ----------
    xy : 2d array-like
        x and y coordinates from which to sample the raster, (n_samples, xy)
    raster : str
        Path to GDAL supported raster

    Returns
    -------
    data : 2d array-like
        Masked array containing sampled raster values (sample, bands)
        at x,y locations"""

    from rasterio.sample import sample_gen

    with rasterio.open(raster) as src:
        training_data = np.vstack([i for i in sample_gen(src, xy)])
        training_data = np.ma.masked_where(
            training_data == np.any(src.nodatavals), training_data)

    raster.close()
    return (training_data)


def __extract_points_raster(xy, y, raster):
    """Samples a list of GDAL rasters using a xy locations

    Parameters
    ----------
    xy : 2d array-like, optional
        x and y coordinates from which to sample the raster, (n_samples, xy)
    y : 1d array-like
        1d numpy array containing training data labels
    raster : str
        Path to GDAL supported raster

    Returns
    -------
    X : 2d array-like
        Masked array containing sampled raster values (sample, bands)
        at x,y locations
    y : 1d array-like
        Masked 1d array containined label values. These contain masked entries
        for missing X values
    xy : 2d array-like
        x,y coordinates of sampled values"""

    src = rasterio.open(raster)

    # convert spatial coordinates to row,col indices of raster
    rowcol = np.transpose(rasterio.transform.rowcol(
            src.transform, xy[:, 0], xy[:, 1]))

    # remove duplicates, i.e. points that fall into same pixel twice
    unique_ind = np.unique(rowcol, return_index=True, axis=0)[1]
    rowcol = rowcol[unique_ind, :]
    y = y[unique_ind]

    # remove rows,cols outside of src dimensions
    invalid_ind = np.nonzero(rowcol[:, 0] >= src.height)
    invalid_ind = np.append(invalid_ind, np.nonzero(rowcol[:, 1] >= src.width))
    not_in_indice = [i for i in range(rowcol.shape[0]) if i not in invalid_ind]
    rowcol = rowcol[not_in_indice]
    y = y[not_in_indice]
    row, col = rowcol[:, 0], rowcol[:, 1]

    # create a 2d array matching the predictors width, height
    # fill labelled pixels with labelled values
    rsp_arr = np.zeros((src.height, src.width))
    rsp_arr[:] = np.nan
    rsp_arr[row, col] = y
    X = np.zeros((0, src.count))
    X[:] = np.nan

    for i, window in src.block_windows():
        src_arr = src.read(window=window)

        # subset the response 2d array into the window size
        row_min, row_max = window.row_off, window.row_off+window.height
        col_min, col_max = window.col_off, window.col_off+window.width
        rsp_arr_window = rsp_arr[row_min:row_max, col_min:col_max]

        # extract src values at labelled pixel locations
        X = np.vstack((X, np.transpose(
                src_arr[:, ~np.isnan(rsp_arr_window)])))

    # mask nodata values
    mx, my = np.nonzero(X != src.nodatavals)
    valid = np.ma.masked_all((X.shape))
    valid[mx, my] = 1
    X = np.ma.masked_where(valid.mask, X)

    # get y values from rasterized row,col locations
    y = rsp_arr[~np.isnan(rsp_arr)]

    # mask y values for missing values in X
    y = np.ma.masked_where(X.mask.any(axis=1), y)

    # convert rowcols into xy and mask for missing values in X
    xy = np.transpose(rasterio.transform.xy(src.transform, row, col))
    xy = np.ma.masked_where(
        np.tile(X.mask.any(axis=1), (2, 1)).transpose(), xy)

    src.close()

    return (X, y, xy)


def random_sample(size, raster, random_state=None):
    """Generates a random sample of according to size, and samples the pixel
    values within the list of rasters

    Parameters
    ----------
    size : int
        Number of random samples to obtain
    rasters : list, str
        List of paths to GDAL supported rasters
    random_state : int
        integer to use within random.seed

    Returns
    -------
    valid_samples: array-like
        Numpy array of extracted raster values, typically 2d
    valid_coordinates: 2d array-like
        2D numpy array of xy coordinates of extracted values"""

    # set the seed
    np.random.seed(seed=random_state)

    # create np array to store randomly sampled data
    # we are starting with zero initial rows because data will be appended,
    # and number of columns are equal to n_features
    src = rasterio.open(raster)
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

    return (valid_samples, valid_coordinates)


def stratified_sample(stratified, n, window=None, crds=False):
    """Creates random points of [size] within each category of a raster

    Parameters
    ----------
    stratified : str
        Path to a GDAL-supported raster that contains the strata (categories)
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
        locations
"""

    src = rasterio.open(stratified)

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

    src.close()

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

    return(xy[np.greater_equal(d, min_dist)])


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
