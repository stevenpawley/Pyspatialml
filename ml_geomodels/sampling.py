#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
import numpy.ma as ma
import rasterio
from rasterio import features
import tempfile
from shapely.geometry import Point
import random


def extractPoints(gdf, rasters, field=None):
    """
    Samples a list of GDAL rasters using a point data set

    Parameters
    ----------
    gdf: Geopandas DataFrame
    field: Field name of attribute to be used the label the extracted data
    rasters: List of paths to GDAL supported rasters

    Returns
    -------

    X: Numpy array of extracted raster values
    y: Numpy array of labels
    coordinates: Numpy array of xy coordinates of samples
    """

    from rasterio.sample import sample_gen

    # get coordinates and label for each point in points_gdf
    coordinates = np.array(gdf.bounds.iloc[:, :2])

    # get names of rasters without path or extension
    grid_names = [os.path.basename(i).split(
            os.path.extsep)[0] for i in rasters]

    # loop through each raster band in explanatory rasters
    for r, name in zip(rasters, grid_names):
        grid = rasterio.open(r, 'r')
        gdf[name] = [float(i) for i in sample_gen(grid, coordinates)]

    return (gdf)


def extractPolygons(gdf, rasters, field=None, na_rm=True, lowmem=False):

    """
    Samples a list of GDAL rasters using a point data set

    Args
    ----
    gdf: Geopandas DataFrame
    rasters: List of paths to GDAL supported rasters
    field: Field name of attribute to be used the label the extracted data

    Returns
    -------

    X: Numpy array of extracted raster values
    y: Numpy array of labels
    coordinates: Numpy array of xy coordinates of samples
    """

    template = rasterio.open(rasters[0], mode='r')
    response_np = np.zeros((template.height, template.width))
    response_np[:] = -99999

    # this is where we create a generator of geom to use in rasterizing
    if field is None:
        shapes = (geom for geom in gdf.geometry)

    if field is not None:
        shapes = ((geom, value) for geom, value in zip(
                  gdf.geometry, gdf[field]))

    features.rasterize(
        shapes=shapes, fill=-99999, out=response_np,
        transform=template.transform, default_value=1)

    response_np = ma.masked_where(response_np == -99999, response_np)

    X, y, y_indexes = extractPixels(
            response_np, rasters, field=None, na_rm=True, lowmem=False)

    return(X, y, y_indexes)


def extractPixels(response_np, rasters, field=None, na_rm=True, lowmem=False):

    """
    Samples a list of GDAL rasters using a labelled numpy array

    Args
    ----
    response_np: Masked 2D numpy array representing labelled pixels
    rasters: List of paths to GDAL supported rasters
    lowmem: Use low memory version using numpy memmap

    Returns
    -------

    X: Numpy array of extracted raster values
    y: Numpy array of labels
    y_indexes: Numpy array of row and column indexes of training pixels
    """

    # determine number of predictors
    n_features = len(rasters)

    # returns indices of labelled values
    is_train = np.nonzero(~response_np.mask)

    # get the labelled values
    training_labels = response_np.data[is_train]
    n_labels = np.array(is_train).shape[1]

    # Create a masked array with the dimensions of the number of columns
    training_data = ma.zeros((n_labels, n_features))
    training_data[:] = np.nan

    # Loop through each raster and extract values at training locations
    if lowmem is True:
        template = rasterio.open(rasters[0])
        feature = np.memmap(tempfile.NamedTemporaryFile(),
                            dtype='float32', mode='w+',
                            shape=(template.height, template.width))

    for band, rasterfile in enumerate(rasters):

        # open rasterio
        rio_rast = rasterio.open(rasterfile)

        if lowmem is False:
            feature = rio_rast.read(1, masked=True)
            training_data[0:n_labels, band] = feature[is_train]
        else:
            feature[:] = rio_rast.read(1, masked=True)[:]
            training_data[0:n_labels, band] = \
                ma.masked_where(
                    feature[is_train] == rio_rast.nodata,
                    feature[is_train])

    # convert indexes of training pixels from tuple to n*2 np array
    is_train = np.array(is_train).T

    # Remove nan rows from training data
    training_data = training_data.filled(np.nan)

    if na_rm is True:
        X = training_data[~np.isnan(training_data).any(axis=1)]
        y = training_labels[~np.isnan(training_data).any(axis=1)]
        y_indexes = is_train[~np.isnan(training_data).any(axis=1)]
    else:
        X = training_data
        y = training_labels
        y_indexes = is_train

    coordinates = np.array(rasterio.transform.xy(
        transform=rio_rast.transform, rows=y_indexes[:, 0],
        cols=y_indexes[:, 1], offset='center')).T

    return (X, y, coordinates)


def sample_random(target_nsamples, rasters, random_state=None):

    """
    Randomly samples a list of rasters

    Args
    ----
    target_nsamples: Number of random samples to obtain
    rasters: List of paths to GDAL supported rasters

    Returns
    -------
    valid_samples: Numpy array of extracted raster values
    valid_coordinates: 2D numpy array of xy coordinates of extracted values
    """

    # set the seed
    np.random.seed(seed=random_state)

    # determine number of GDAL rasters to sample
    n_features = len(rasters)

    # open rasters
    dataset = [0] * n_features
    for n in range(n_features):
        dataset[n] = rasterio.open(rasters[n], mode='r')

    # create np array to store randomly sampled data
    # we are starting with zero initial rows because data will be appended,
    # and number of columns are equal to n_features
    valid_samples = np.zeros((0, n_features))
    valid_coordinates = np.zeros((0, 2))

    # loop until target number of samples is satified
    satisfied = False

    n = target_nsamples
    while satisfied is False:

        # generate random row and column indices
        Xsample = np.random.choice(range(0, dataset[0].shape[1]), n)
        Ysample = np.random.choice(range(0, dataset[0].shape[0]), n)

        # create 2d numpy array with sample locations set to 1
        sample_raster = np.empty((dataset[0].shape[0], dataset[0].shape[1]))
        sample_raster[:] = np.nan
        sample_raster[Ysample, Xsample] = 1

        # get indices of sample locations
        is_train = np.nonzero(np.isnan(sample_raster) == False)

        # create ma array with rows=1 and cols=n_features
        samples = ma.masked_all((len(is_train[0]), n_features))

        # loop through each raster and sample
        for r in range(n_features):
            feature = dataset[r].read(1, masked=True)
            samples[0:n, r] = feature[is_train]

        # append only non-masked data to each row of X_random
        samples = samples.filled(np.nan)

        valid_samples = np.append(
                valid_samples, samples[~np.isnan(samples).any(axis=1)], axis=0)

        is_train = np.array(is_train).T
        valid_coordinates = np.append(valid_coordinates,
                                      is_train[~np.isnan(samples).any(axis=1)],
                                      axis=0)

        # check to see if target_nsamples has been reached
        if len(valid_samples) >= target_nsamples:
            satisfied = True
        else:
            n = target_nsamples - len(valid_samples)

    return (valid_samples, valid_coordinates)


def filter_points(gdf, min_dist=0, remove='first'):
    """
    Filter points in geodataframe using a minimum distance

    Args
    ----
    gdf: Geodataframe
    min_dist: Points with coordinates closer than minimum distance
              are removed
    remove: Remove 'first' occurrences or 'last' occurrences

    Returns
    -------
    gdf: Geodataframe with filtered points
    """

    crds = gdf['geometry'].bounds.iloc[:, 2:4].as_matrix()
    dm = scipy.spatial.distance_matrix(crds, crds)
    np.fill_diagonal(dm, np.nan)

    if remove == 'first':
        dm[np.tril_indices(dm.shape[0], -1)] = np.nan
    elif remove == 'last':
        dm[np.triu_indices(dm.shape[0], -1)] = np.nan

    d = np.nanmin(dm, axis=1)

    return(gdf.iloc[np.greater_equal(d, min_dist)])


def get_random_point_in_polygon(poly):
    """
    Generates random shapely Point geometry objects within a single
    shapely Polygon object
    from https://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    Args
    ----
    poly: Shapely Polygon object
    
    Returns
    -------
    p: Shapely Point object
    """
    
    (minx, miny, maxx, maxy) = poly.bounds
    while True:
         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly.contains(p):
             return p
