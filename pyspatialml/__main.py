import math
import numpy as np
import rasterio


def print_progressbar(iteration, total, prefix = '', suffix = '', decimals = 1,
                     length = 100, fill = '█'):
    """Call in a loop to create terminal progress bar
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113

    Parameters
    ----------
    iteration: int
        Current iteration
    total: int
        Total iterations
    prefix: str, optional
        prefix string (Str)
    suffix: str, optional
        suffix
    decimals: int, optional
        Positive number of decimals in percent complete
    length: int, optional
        Character length of bar
    fill: str, optional
        bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '')
    # Print New Line on Complete
    if iteration == total:
        print()


def __predfun(img, **kwargs):
    estimator = kwargs['estimator']
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


def __probfun(img, **kwargs):
    estimator = kwargs['estimator']
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


def applier(src, func, file_path, indexes=None, driver='GTiff',
            dtype='float32', nodata=-99999, **kwargs):
    """Applies a function to a RasterStack object in image strips

    Parameters
    ----------
    src : rasterio.io.reader
    func : function to execute in block_windows
    file_path : str
        File path for output of processing operation
    indexes : List, int, optional
        List of class indices to export
    driver : str, optional. Default is 'GTiff'
        Named of GDAL-supported driver for file export
    dtype : str, optional. Default is 'float32'
        Numpy data type for file export
    nodata : any number, optional. Default is -99999
        Nodata value for file export
    kwargs : tuple
        Additional keyword arguments to pass to func"""

    # Loop through rasters strip-by-strip
    for i, window in src.block_windows():

        img = src.read(masked=True, window=window)
        result = func(img, **kwargs)

        if i[0] == 0:
            # get indexes from number of bands in result
            if indexes is None:
                indexes = range(result.shape[0])

            # define output file
            func_output = rasterio.open(
                file_path, mode='w', driver=driver,
                height=src.height, width=src.width,
                count=result.shape[0], dtype=dtype,
                crs=src.meta['crs'], transform=src.transform,
                nodata=nodata)

        # write data
        if indexes is not None:
            result = result[indexes, :, :]

        func_output.write(
            result.astype(dtype), window=window)

    func_output.close()

    return func_output


def predict(estimator, raster, file_path, predict_type='raw', indexes=None,
            driver='GTiff', dtype='float32', nodata=-99999):
    """Prediction on list of GDAL rasters using a fitted scikit learn model

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    predictor_rasters : list, comprising str
        List of paths to GDAL rasters that are to be used in the prediction.
        Note the order of the rasters in the list needs to be of the same
        order and length of the data that was used to train the estimator
    output : str
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

    # determine output count
    if predict_type == 'prob' and isinstance(indexes, int):
        indexes = range(indexes-1, indexes)

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

    # determine number of processing blocks
    n_blocks = len([i for i in src.block_windows()])

    with rasterio.open(file_path, 'w', **meta) as dst:
        for i, (index, window) in enumerate(src.block_windows()):
            print_progressbar(iteration=i, total=n_blocks)

            # read image data
            img = src.read(masked=True, window=window)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

            # reshape each image block matrix into a 2D matrix
            # first reorder into rows,cols,bands(transpose)
            # then resample into 2D array (rows=sample_n, cols=band_values)
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape(
                (n_samples, n_features))

            # create mask for NaN values and replace with number
            flat_pixels_mask = flat_pixels.mask.copy()

            if predict_type == 'raw':
                result = estimator.predict(flat_pixels)

                # replace mask and fill masked values with nodata value
                result = np.ma.masked_array(
                    result, mask=flat_pixels_mask.any(axis=1))
                result = np.ma.filled(result, fill_value=-99999)

                # reshape the prediction from a 1D into 3D array [band, row, col]
                result = result.reshape((1, rows, cols))

            elif predict_type == 'prob':
                result = estimator.predict_proba(flat_pixels)

                # reshape class probabilities back to 3D image [iclass, rows, cols]
                result = result.reshape(
                    (rows, cols, result.shape[1]))
                flat_pixels_mask = flat_pixels_mask.reshape((rows, cols, n_features))

                # flatten mask into 2d
                mask2d = flat_pixels_mask.any(axis=2)
                #mask2d = np.where(mask2d != mask2d.min(), True, False)
                mask2d = np.repeat(mask2d[:, :, np.newaxis],
                                   result.shape[2], axis=2)

                # convert proba to masked array using mask2d
                result = np.ma.masked_array(
                    result,
                    mask=mask2d,
                    fill_value=np.nan)
                result = np.ma.filled(
                    result, fill_value=-99999)

                # reshape band into rasterio format [band, row, col]
                result = result.transpose(2, 0, 1)

            # write data
            if indexes is not None:
                result = result[indexes, :, :]

            dst.write(result.astype(dtype), window=window)

    src.close()
    return dst


def predict_multi(estimator, raster, file_path, predict_type='raw', indexes=None,
            driver='GTiff', dtype='float32', nodata=-99999, n_jobs=-1):
    """Prediction on list of GDAL rasters using a fitted scikit learn model

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    predictor_rasters : list, comprising str
        List of paths to GDAL rasters that are to be used in the prediction.
        Note the order of the rasters in the list needs to be of the same
        order and length of the data that was used to train the estimator
    output : str
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

    # determine output count
    if predict_type == 'prob' and isinstance(indexes, int):
        indexes = range(indexes-1, indexes)

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

        # Define a generator for data, window pairs.
        def jobs():
            for ij, window in dst.block_windows():
                data = src.read(masked=True, window=window)
                result = np.zeros(data.shape, dtype=data.dtype)
                yield data, result, window

            # Submit the jobs to the thread pool executor.
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_jobs) as executor:

                # Map the futures returned from executor.submit()
                # to their destination windows.
                future_to_window = {
                    executor.submit(probfun, data, res): (res, window)
                    for data, res, window in jobs()}

                # As the processing jobs are completed, get the
                # results and write the data to the appropriate
                # destination window.
                for future in concurrent.futures.as_completed(future_to_window):
                    result, window = future_to_window[future]
                    dst.write(arr, window=window)

    src.close()
    return dst
