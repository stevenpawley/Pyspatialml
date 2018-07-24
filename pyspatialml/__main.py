import numpy as np
import rasterio
from tqdm import tqdm
import geopandas
from rasterio.sample import sample_gen


def _predfun(img, estimator):
    """Prediction function for classification or regression response

    Parameters
    ----------
    img : 3d numpy array of raster data

    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    Returns
    -------
    result_cla : 2d numpy array
        Single band raster as a 2d numpy array containing the
        classification or regression result"""

    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape each image block matrix into a 2D matrix
    # first reorder into rows,cols,bands(transpose)
    # then resample into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape(
        (n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()

    # prediction
    result_cla = estimator.predict(flat_pixels)

    # replace mask and fill masked values with nodata value
    result_cla = np.ma.masked_array(
        result_cla, mask=flat_pixels_mask.any(axis=1))
    result_cla = np.ma.filled(result_cla, fill_value=-99999)

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, rows, cols))

    return result_cla


def _probfun(img, estimator):
    """Class probabilities function

    Parameters
    ----------
    img : 3d numpy array of raster data

    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    Returns
    -------
    result_proba : 3d numpy array
        Multi band raster as a 3d numpy array containing the
        probabilities associated with each class.
        Array is in (class, row, col) order"""

    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape each image block matrix into a 2D matrix
    # first reorder into rows,cols,bands(transpose)
    # then resample into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape(
        (n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()

    # predict probabilities
    result_proba = estimator.predict_proba(flat_pixels)

    # reshape class probabilities back to 3D image [iclass, rows, cols]
    result_proba = result_proba.reshape(
        (rows, cols, result_proba.shape[1]))
    flat_pixels_mask = flat_pixels_mask.reshape((rows, cols, n_features))

    # flatten mask into 2d
    mask2d = flat_pixels_mask.any(axis=2)
    mask2d = np.where(mask2d != mask2d.min(), True, False)
    mask2d = np.repeat(mask2d[:, :, np.newaxis],
                       result_proba.shape[2], axis=2)

    # convert proba to masked array using mask2d
    result_proba = np.ma.masked_array(
        result_proba,
        mask=mask2d,
        fill_value=np.nan)
    result_proba = np.ma.filled(
        result_proba, fill_value=-99999)

    # reshape band into rasterio format [band, row, col]
    result_proba = result_proba.transpose(2, 0, 1)

    return result_proba


def maximum_dtype(src):
    """Returns a single dtype that is large enough to store data
    within all raster bands

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Rasterio datasetreader in the opened mode

    Returns
    -------
    dtype : str
        Dtype that is sufficiently large to store all raster
        bands in a single numpy array"""

    if 'complex128' in src.dtypes:
        dtype = 'complex128'
    elif 'complex64' in src.dtypes:
        dtype = 'complex64'
    elif 'complex' in src.dtypes:
        dtype = 'complex'
    elif 'float64' in src.dtypes:
        dtype = 'float64'
    elif 'float32' in src.dtypes:
        dtype = 'float32'
    elif 'int32' in src.dtypes:
        dtype = 'int32'
    elif 'uint32' in src.dtypes:
        dtype = 'uint32'
    elif 'int16' in src.dtypes:
        dtype = 'int16'
    elif 'uint16' in src.dtypes:
        dtype = 'uint16'
    elif 'uint16' in src.dtypes:
        dtype = 'uint16'
    elif 'bool' in src.dtypes:
        dtype = 'bool'

    return dtype


def predict(estimator, dataset, file_path, predict_type='raw', indexes=None,
            driver='GTiff', dtype='float32', nodata=-99999):
    """Apply prediction of a scikit learn model to a GDAL-supported
    raster dataset

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    dataset : rasterio.io.DatasetReader
        An opened Rasterio DatasetReader

    file_path : str
        Path to a GeoTiff raster for the classification results

    predict_type : str, optional (default='raw')
        'raw' for classification/regression
        'prob' for probabilities

    indexes : List, int, optional
        List of class indices to export

    driver : str, optional. Default is 'GTiff'
        Named of GDAL-supported driver for file export

    dtype : str, optional. Default is 'float32'
        Numpy data type for file export

    nodata : any number, optional. Default is -99999
        Nodata value for file export

    Returns
    -------
    rasterio.io.DatasetReader with predicted raster"""

    src = dataset

    # chose prediction function
    if predict_type == 'raw':
        predfun = _predfun
    elif predict_type == 'prob':
        predfun = _probfun

    # determine output count
    if predict_type == 'prob' and isinstance(indexes, int):
        indexes = range(indexes, indexes+1)

    elif predict_type == 'prob' and indexes is None:
        img = src.read(masked=True, window=(0, 0, 1, src.width))
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape(
            (n_samples, n_features))
        result = estimator.predict_proba(flat_pixels)
        indexes = range(result.shape[0])

    elif predict_type == 'raw':
        indexes = range(1)

    # open output file with updated metadata
    meta = src.meta
    meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)

    with rasterio.open(file_path, 'w', **meta) as dst:

        # define windows
        windows = [window for ij, window in dst.block_windows()]

        # generator gets raster arrays for each window
        # read all bands if single dtype
        if src.dtypes.count(src.dtypes[0]) == len(src.dtypes):
            data_gen = (src.read(window=window, masked=True) for window in windows)

        # else read each band separately
        else:
            def read(src, window):
                dtype = maximum_dtype(src)
                arr = np.ma.zeros((src.count, window.height, window.width), dtype=dtype)

                for band in range(src.count):
                    arr[band, :, :] = src.read(band+1, window=window, masked=True)
                return arr

            data_gen = (read(src=src, window=window) for window in windows)

        with tqdm(total=len(windows)) as pbar:
            for window, arr in zip(windows, data_gen):
                result = predfun(arr, estimator)
                dst.write(result[indexes, :, :].astype(dtype), window=window)
                pbar.update(1)

    return rasterio.open(file_path)


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
        if len(np.unique(response.geom_type)) > 1:
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

    # clip points to extent of raster
    extent = src.bounds

    valid_idx = np.where((xy[:, 0] > extent.left) &
                         (xy[:, 0] < extent.right) &
                         (xy[:, 1] > extent.bottom) &
                         (xy[:, 1] < extent.top))[0]

    xy = xy[valid_idx, :]
    y = y[valid_idx]

    # extract values at labelled indices
    X = _extract_points(xy, src)

    # summarize data and mask nodatavals in X, y, and xy
    y = np.ma.masked_where(X.mask.any(axis=1), y)
    Xmask_xy = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
    xy = np.ma.masked_where(Xmask_xy, xy)

    return X, y, xy


def _extract_points(xy, dataset):
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
