#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy
from shapely.geometry import Point
import random
from sklearn.metrics import confusion_matrix
from numpy.random import RandomState
import math
import rasterio
from rasterio.warp import reproject
from tempfile import NamedTemporaryFile as tmpfile
import geopandas as gpd


def filter_points(gdf, min_dist=0, remove='first'):
    """Filter points in geodataframe using a minimum distance buffer

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing Point feature types within the GeoSeries
    min_dist : int or float, optional (default=0)
        Minimum distance by which to filter out closely spaced points
    remove : str, optional (default='first')
        Optionally choose to remove 'first' occurrences or 'last' occurrences

    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame containing with filtered point features
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
    """Generates random shapely Point geometry objects within a single
    shapely Polygon object

    Parameters
    ----------
    poly : Shapely Polygon object

    Returns
    -------
    p : Shapely Point object

    Notes
    -----
    from https://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    """

    (minx, miny, maxx, maxy) = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p


def RandomPoints(gdf, n_points):
    """Fit n_points random points per polygon"""
    random_points = []
    
    for i, poly in gdf.iterrows():
        random_points.append(
            get_random_point_in_polygon(poly.geometry) for k in range(n_points))
        
    points = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(random_points), crs=gdf.crs)
    
    return points


def specificity_score(y_true, y_pred):
    """Calculate specificity score metric for a binary classification

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels
    y_pred: 1d array-like
        Predicted labels, as returned by the classifier

    Returns
    -------
    specificity : float
        Returns the specificity score, or true negative rate, i.e. the
        proportion of the negative label (label=0) samples that are correctly
        classified as the negative label
    """

    cm = confusion_matrix(y_true, y_pred)
    tn = float(cm[0][0])
    fp = float(cm[0][1])

    return tn/(tn+fp)


def spatial_loocv(estimator, X, y, coordinates, size, radius,
                  random_state=None):
    """Spatially buffered leave-One-out cross-validation
    Uses a circular spatial buffer to separate samples that are used to train
    the estimator, from samples that are used to test the prediction accuracy

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    coordinates : 2d-array like
        Spatial coordinates, usually as xy, that correspond to each sample in
        X.
    size : int
        Sample size to process (number of leave-one-out runs)
    radius : int or float
        Radius for the spatial buffer around test point
    random_state : int
        random_state is the seed used by the random number generator

    Returns
    -------
    y_test : 1d array-like
        Response variable values in the test partitions
    y_pred : 1d array-like
        Predicted response values by the estimator
    y_prob : array-like
        Predicted probabilities by the estimator

    Notes
    -----
    This python function was adapted from R code
    https://davidrroberts.wordpress.com/2016/03/11/spatial-leave-one-out-sloo-cross-validation/
    """

    # determine number of classes and features
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    rstate = RandomState(random_state)

    # randomly select the testing points
    ind = rstate.choice(range(X.shape[0]), size)
    X_test = X[ind, :]
    y_test = y[ind]
    coordinates_test = coordinates[ind, :]

    # variables to store predictions and probabilities
    y_pred = np.empty((0,))
    y_prob = np.empty((0, n_classes))

    # loop through the testing points
    for i in range(size):
        # Training data (test point & buffer removed)
        # empty numpy arrays to append training that is > radius from test loc
        X_train = np.empty((0, n_features))
        y_train = np.empty((0))

        # loop through each point in the original training data
        # and append to X_train, y_train if coordinates > minimum radius
        for j in range(X.shape[0]):
            if math.sqrt((coordinates[j, 0] - coordinates_test[i, 0])**2 +
                         (coordinates[j, 1] - coordinates_test[i, 1])**2) \
                          > radius:
                X_train = np.vstack((X_train, X[j]))
                y_train = np.append(y_train, y[j])

        # Build the model
        estimator.fit(X_train, y_train)

        # Predict on test point
        y_pred = np.append(y_pred, estimator.predict(X_test[i].reshape(1, -1)))
        y_prob = np.vstack((
            y_prob, estimator.predict_proba(X_test[i].reshape(1, -1))))

    return (y_test, y_pred, y_prob)


def read_memmap(src, bands=None):
    """Read a rasterio._io.RasterReader object into a numpy.memmap file

    Parameters
    ----------
    src : rasterio._io.RasterReader
        Rasterio raster that is open.
    bands : int or list, optional (default = None)
        Optionally specify to load a single band, a list of bands, or None for
        all bands within the raster.

    Returns
    -------
    arr : array-like
        2D or 3D numpy.memmap containing raster data. Nodata values are
        represented by np.nan. By default a 3D numpy.memmap object is returned
        unless a single band is specified, in which case a 2D numpy.memmap
        object is returned.
    """

    # default loads all bands as 3D numpy array
    if bands is None:
        bands = range(1, src.count+1, 1)
        arr_shape = (src.count, src.shape[0], src.shape[1])

    # if a single band is specified use 2D numpy array
    elif len(bands) == 1:
        arr_shape = (src.shape[0], src.shape[1])

    # use 3D numpy array for a subset of bans
    else:
        arr_shape = (len(bands), src.shape[0], src.shape[1])

    # numpy.memmap using float32
    arr = np.memmap(filename=tmpfile(), shape=arr_shape, dtype='float32')
    arr[:] = src.read(bands)
    arr[arr == src.nodata] = np.nan

    return arr


def reclass_nodata(input, output, src_nodata=None, dst_nodata=-99999,
                   intern=False):
    """Reclassify raster no data values and save to a new raster

    Parameters
    ----------
    input : str
        File path to raster that is to be reclassified
    output : str
        File path to output raster
    src_nodata : int or float, optional
        The source nodata value. Pixels with this value will be reclassified
        to the new dst_nodata value.
        If not set, it will default to the nodata value stored in the source
        image.
    dst_nodata : int or float, optional (default -99999)
        The nodata value that the outout raster will receive after
        reclassifying pixels with the src_nodata value.
    itern : bool, optional (default=False)
        Optionally return the reclassified raster as a numpy array

    Returns
    -------
    out: None, or 2D numpy array of raster with reclassified nodata pixels
    """

    r = rasterio.open(input)
    width, height, transform, crs = r.width, r.height, r.transform, r.crs
    img_ar = r.read(1)
    if src_nodata is None:
        src_nodata = r.nodata

    img_ar[img_ar == src_nodata] = dst_nodata
    r.close()

    with rasterio.open(path=output, mode='w', driver='GTiff', width=width,
                       height=height, count=1, transform=transform, crs=crs,
                       dtype=str(img_ar.dtype), nodata=dst_nodata) as dst:
        dst.write(img_ar, 1)

    if intern is True:
        return (img_ar)
    else:
        return()


def align_rasters(rasters, template, outputdir, method="Resampling.nearest",
                  src_nodata=None, dst_nodata=None):

    """Aligns a list of rasters (paths to files) to a template raster.
    The rasters to that are to be realigned are assumed to represent
    single band raster files.

    Nodata values are also reclassified to the template raster's nodata values

    Parameters
    ----------
    rasters : List of str
        List containing file paths to multiple rasters that are to be realigned
    template : str
        File path to raster that is to be used as a template to transform the
        other rasters to.
    outputdir : str
        Directory to output the realigned rasters. This should not be the
        existing directory unless it is desired that the existing rasters to be
        realigned should be overwritten.
    method : str
        Resampling method to use. One of the following:
            Resampling.average,
            Resampling.bilinear,
            Resampling.cubic,
            Resampling.cubic_spline,
            Resampling.gauss,
            Resampling.lanczos,
            Resampling.max,
            Resampling.med,
            Resampling.min,
            Resampling.mode,
            Resampling.nearest,
            Resampling.q1,
            Resampling.q3
    src_nodata : int or float, optional
        The source raster nodata value. Pixels with this value will be
        transformed to the new dst_nodata value.
        If not set, it will default to the nodata value stored in the source
        image.
    dst_nodata : int or float
        The nodata value that the outout raster will receive after realignment
        If not set, the source rasters nodata value will be used, or the
        GDAL default of 0
    """
    # check resampling methods
    methods = dir(rasterio.enums.Resampling)
    methods = [i for i in methods if i.startswith('__') is False]
    methods = ['Resampling.' + i for i in methods]

    if method not in methods:
        raise ValueError('Invalid resampling method: ' + method + os.linesep +
                         'Valid methods are: ' + str(methods))

    # open raster to be used as the template and create numpy array
    template = rasterio.open(template)
    kwargs = template.meta.copy()

    for raster in rasters:

        output = os.path.join(outputdir, os.path.basename(raster))

        with rasterio.open(raster) as src:
            with rasterio.open(output, 'w', **kwargs) as dst:
                reproject(source=rasterio.band(src, 1),
                          destination=rasterio.band(dst, 1),
                          dst_transform=template.transform,
                          dst_nodata=template.nodata,
                          dst_crs=template.nodata)
    template.close()
    return()
