import numpy as np
import pandas as pd


def stack_constants(flat_pixels, constants, names=None):
    """Column stack any constant values into the flat_pixels array.

    Used to add additional constant features to the Raster object.

    Parameters
    ----------
    flat_pixels : ndarray
        2d numpy array representing the flattened raster data in
        (sample_n, band_values) format.

    constants : list-like object, 1d array, or dict
        Array of constant values to be added to the flat_pixels array
        as additional features.

        If a dict is passed, the dict keys must refer to names of the
        features in the flat_pixels array, and the values will replace
        these features with constant values.

    names : list-like object (optional, default=None)
        Names of the raster layers.
    """
    if isinstance(constants, (int, float)):
        constants = [constants]

    if isinstance(constants, list):
        constants = np.asarray(constants)
        constants = np.broadcast_to(
            constants, (flat_pixels.shape[0], constants.shape[0])
        )
        flat_pixels = np.column_stack((flat_pixels, constants))

    elif isinstance(constants, dict):

        keys_not_in_raster = [i for i in constants.keys() if i not in names]

        if len(keys_not_in_raster) > 0:
            raise ValueError(
                "The following keys are not in the raster: {x}".format(
                    x=keys_not_in_raster
                )
            )

        flat_pixels = pd.DataFrame(flat_pixels, columns=names)

        for key, value in constants.items():
            flat_pixels[key] = value

        flat_pixels = flat_pixels.values

    elif isinstance(constants, np.ndarray):
        raise ValueError("constants must be a list or a numpy.ndarray")

    return flat_pixels


def flatten_valid_pixels(img, constants=None, names=None):
    """Flatten a raster window and return only pixels valid in every band.

    Returns the window, a 2D feature matrix for valid pixels, and a 1D mask
    where ``True`` marks invalid pixels in row-major order.
    """
    window, img = img
    img = np.ma.masked_invalid(img)
    n_features, rows, cols = img.shape
    invalid = np.ma.getmaskarray(img).any(axis=0).reshape(-1)
    flat = img.transpose(1, 2, 0).reshape((rows * cols, n_features))
    flat_pixels = flat[~invalid].filled(0)

    if constants is not None:
        flat_pixels = stack_constants(flat_pixels, constants, names)

    return window, flat_pixels, invalid


def predict_output(img, estimator, constants=None, names=None):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the
        dimensions in order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    constants : list-like object, 1d array, or dict
        Array of constant values to be added to the flat_pixels array
        as additional features.

        If a dict is passed, the dict keys must refer to names of the
        features in the flat_pixels array, and the values will replace
        these features with constant values.

    names : list-like object (optional, default=None)
        Names of the raster layers.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, flat_pixels, invalid = flatten_valid_pixels(img, constants, names)
    if flat_pixels.shape[0] == 0:
        return np.ma.masked_all(
            (1, window.height, window.width), dtype=np.float32
        )

    prediction = np.asarray(estimator.predict(flat_pixels)).reshape(-1)
    if prediction.size != (~invalid).sum():
        raise ValueError("estimator.predict returned an unexpected number of values")

    data = np.zeros(invalid.size, dtype=prediction.dtype)
    data[~invalid] = prediction
    return np.ma.array(
        data.reshape((1, window.height, window.width)),
        mask=invalid.reshape((1, window.height, window.width)),
    )


def predict_prob(img, estimator, constants=None, names=None, output_count=None):
    """Class probabilities function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the
        dimensions in order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    constants : list-like object, 1d array, or dict
        Array of constant values to be added to the flat_pixels array
        as additional features.

        If a dict is passed, the dict keys must refer to names of the
        features in the flat_pixels array, and the values will replace
        these features with constant values.

    names : list-like object (optional, default=None)
        Names of the raster layers.

    Returns
    -------
    numpy.ndarray
        Multi band raster as a 3d numpy array containing the
        probabilities associated with each class. ndarray dimensions
        are in the order of (class, row, column).
    """
    window, flat_pixels, invalid = flatten_valid_pixels(img, constants, names)
    if flat_pixels.shape[0] == 0:
        if output_count is None:
            classes = estimator.classes_
            if isinstance(classes, list):
                raise ValueError(
                    "output_count is required for multioutput probabilities"
                )
            output_count = len(classes)
        return np.ma.masked_all(
            (output_count, window.height, window.width), dtype=np.float32
        )

    valid_proba = np.asarray(estimator.predict_proba(flat_pixels))
    if valid_proba.ndim != 2 or valid_proba.shape[0] != (~invalid).sum():
        raise ValueError("estimator.predict_proba returned an unexpected shape")

    data = np.zeros((valid_proba.shape[1], invalid.size), dtype=valid_proba.dtype)
    data[:, ~invalid] = valid_proba.T
    mask = np.broadcast_to(invalid, data.shape)
    return np.ma.array(
        data.reshape((-1, window.height, window.width)),
        mask=mask.reshape((-1, window.height, window.width)),
        fill_value=np.nan,
    )


def predict_multioutput(
    img, estimator, constants=None, names=None, output_count=None
):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the
        dimensions in order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    constants : list-like object, 1d array, or dict
        Array of constant values to be added to the flat_pixels array
        as additional features.

        If a dict is passed, the dict keys must refer to names of the
        features in the flat_pixels array, and the values will replace
        these features with constant values.

    names : list-like object (optional, default=None)
        Names of the raster layers.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result
        with the dimensions in the order of (target, row, column).
    """
    window, flat_pixels, invalid = flatten_valid_pixels(img, constants, names)
    if flat_pixels.shape[0] == 0:
        if output_count is None:
            output_count = getattr(estimator, "n_outputs_", None)
        if output_count is None:
            raise ValueError(
                "output_count is required when a window has no valid pixels"
            )
        return np.ma.masked_all(
            (output_count, window.height, window.width), dtype=np.float32
        )

    valid_result = np.asarray(estimator.predict(flat_pixels))
    if valid_result.ndim == 1:
        valid_result = valid_result[:, np.newaxis]
    if valid_result.ndim != 2 or valid_result.shape[0] != (~invalid).sum():
        raise ValueError("estimator.predict returned an unexpected shape")

    data = np.zeros((valid_result.shape[1], invalid.size), dtype=valid_result.dtype)
    data[:, ~invalid] = valid_result.T
    mask = np.broadcast_to(invalid, data.shape)
    return np.ma.array(
        data.reshape((-1, window.height, window.width)),
        mask=mask.reshape((-1, window.height, window.width)),
    )
