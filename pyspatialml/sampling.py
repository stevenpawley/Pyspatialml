import os
import random
import numpy as np
import rasterio
import scipy
from rasterio.sample import sample_gen
from shapely.geometry import Point


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

        # point features
        elif all(response_gdf.geom_type == 'Point'):

            xy = response_gdf.bounds.iloc[:, 2:].values

            if field:
                y = response_gdf[field].values
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


    # clip points to extent of raster
    src = rasterio.open(raster)
    extent = src.bounds

    valid_idx = np.where((xy[:, 0] > extent.left) &
                         (xy[:, 0] < extent.right) &
                         (xy[:, 1] > extent.bottom) &
                         (xy[:, 1] < extent.top))[0]

    xy = xy[valid_idx, :]
    y = y[valid_idx]

    # extract values at labelled indices
    X = __extract_points(xy, raster)

    # summarize data and mask nodatavals in X, y, and xy
    y = np.ma.masked_where(X.mask.any(axis=1), y)
    Xmask_xy = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
    xy = np.ma.masked_where(Xmask_xy, xy)

    src.close()

    return X, y, xy


def __extract_points(xy, raster):
    """Samples pixel values from a raster using an array of xy locations

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

    with rasterio.open(raster) as src:

        # read all bands if single dtype
        if src.dtypes.count(src.dtypes[0]) == len(src.dtypes):
            training_data = np.vstack([i for i in sample_gen(src, xy)])

        # read single bands if multiple dtypes
        else:
            dtype = maximum_dtype(src)
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


def random_sample(size, raster, random_state=None):
    """Generates a random sample of according to size, and samples the pixel
    values within the list of rasters

    Parameters
    ----------
    size : int
        Number of random samples to obtain

    raster : str
        Paths to GDAL supported raster

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

    return valid_samples, valid_coordinates


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
        locations"""

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
