import numpy as np
import pandas as pd

def stack_constants(flat_pixels, constants):
    """Column stack any constant values into the flat_pixels array.

    Used to add additional constant features to the Raster object.

    Parameters
    ----------
    flat_pixels : list-like or 1d numpy.ndarray
        2d numpy array representing the flattened raster data in 
        (sample_n, band_values) format.
    
    constants : numpy.ndarray (optional)
        Array of constant values to be added to the flat_pixels array
        as additional features.
    """
    if isinstance(constants, (int, float)):
        constants = np.asarray([constants])
    elif isinstance(constants, list):
        constants = np.asarray(constants)
    elif isinstance(constants, np.ndarray):
        raise ValueError('constants must be a list or a numpy.ndarray')

    constants = np.broadcast_to(constants, (flat_pixels.shape[0], constants.shape[0]))
    flat_pixels = np.column_stack((flat_pixels, constants))
    return flat_pixels


def predict_output(img, estimator, constants=None):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the
        dimensions in order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
            
    constants: numpy.ndarray (optional)
        1d array of constant values to be added as features.
    
    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img
    img = np.ma.masked_invalid(img)

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values
    flat_pixels_mask = flat_pixels.mask.copy()

    # fill nans for prediction
    flat_pixels = flat_pixels.filled(0)
    
    # add constants
    if constants is not None:
        flat_pixels = stack_constants(flat_pixels, constants)
    
    # predict and replace mask
    result = estimator.predict(flat_pixels)
    result = np.ma.masked_array(
        data=result,
        mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result = result.reshape((1, window.height, window.width))

    return result


def predict_prob(img, estimator, constants=None):
    """Class probabilities function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the
        dimensions in order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    constants: numpy.ndarray (optional)
        1d array of constant values to be added as features.

    Returns
    -------
    numpy.ndarray
        Multi band raster as a 3d numpy array containing the
        probabilities associated with each class. ndarray dimensions
        are in the order of (class, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands (transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    img = np.ma.masked_invalid(img)
    mask2d = img.mask.any(axis=0)

    # then resample into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # fill mask with zeros for prediction
    flat_pixels = flat_pixels.filled(0)

    # add constants
    if constants is not None:
        flat_pixels = stack_constants(flat_pixels, constants)

    # predict probabilities
    result_proba = estimator.predict_proba(flat_pixels)

    # reshape class probabilities back to 3D array [class, rows, cols]
    result_proba = result_proba.reshape(
        (window.height, window.width, result_proba.shape[1])
    )

    # reshape band into rasterio format [band, row, col]
    result_proba = result_proba.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result_proba.shape[0],
        axis=0
    )

    # convert proba to masked array
    result_proba = np.ma.masked_array(result_proba, mask=mask3d, fill_value=np.nan)

    return result_proba


def predict_multioutput(img, estimator, constants=None):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the
        dimensions in order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    constants: numpy.ndarray (optional)
        1d array of constant values to be added as features.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result
        with the dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    img = np.ma.masked_invalid(img)
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)

    # add constants
    if constants is not None:
        flat_pixels = stack_constants(flat_pixels, constants)

    # predict probabilities
    result = estimator.predict(flat_pixels)

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(a=mask2d[np.newaxis, :, :], repeats=result.shape[0], axis=0)

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result
