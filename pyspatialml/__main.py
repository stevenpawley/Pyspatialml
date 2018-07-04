import numpy as np
import rasterio
from tqdm import tqdm
from osgeo import gdal

def __predfun(img, estimator):
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


def __probfun(img, estimator):
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


def predict(estimator, raster, file_path, predict_type='raw', indexes=None,
            driver='GTiff', dtype='float32', nodata=-99999):
    """Prediction on list of GDAL rasters using a fitted scikit learn model
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    raster : list, comprising str
        List of paths to GDAL rasters that are to be used in the prediction.
        Note the order of the rasters in the list needs to be of the same
        order and length of the data that was used to train the estimator
    file_path : str
        Path to a GeoTiff raster for the classification results
    predict_type : str, optional (default='raw')
        'raw' for classification/regression
        'prob' for probabilities,
    indexes : List, int, optional
        List of class indices to export
    driver : str, optional. Default is 'GTiff'
        Named of GDAL-supported driver for file export
    dtype : str, optional. Default is 'float32'
        Numpy data type for file export
    nodata : any number, optional. Default is -99999
        Nodata value for file export"""

    src = rasterio.open(raster)

    # chose prediction function
    if predict_type == 'raw':
        predfun = __predfun
    elif predict_type == 'prob':
        predfun = __probfun

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
        data_gen = (src.read(window=window, masked=True) for window in windows)

        with tqdm(total=len(windows)) as pbar:
            for window, arr in zip(windows, data_gen):
                result = predfun(arr, estimator)
                dst.write(result[indexes, :, :].astype(dtype), window=window)
                pbar.update(1)

    src.close()
