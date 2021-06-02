from copy import deepcopy

import numpy as np
import rasterio
from scipy import ndimage

from .raster import Raster


def one_hot_encode(layer, file_path, categories=None, driver="GTiff"):
    """One-hot encoding of a RasterLayer.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer
        Containing categories to perform one-hot encoding on.

    file_path : str
        File path to save one-hot encoded raster.

    categories : list, ndarray, optional
        Optional list of categories to extract. Default performs one-hot
        encoding on all categorical values in the input layer.

    driver : str, options. Default is 'GTiff'
        GDAL-compatible driver.

    Returns
    -------
    pyspatialml.Raster
        Each categorical value is encoded as a layer with a Raster object.
    """
    arr = layer.read(masked=True)

    if categories is None:
        categories = np.unique(arr)
        categories = categories[~categories.mask]
        categories = categories.data.astype("int32")

    arr_ohe = np.ma.zeros((len(categories), arr.shape[0], arr.shape[1]), dtype="int32")
    names = []
    prefix = layer.names[0]

    for i, cat in enumerate(categories):
        enc = deepcopy(arr)
        enc[enc != cat] = 0
        enc[enc == cat] = 1
        arr_ohe[i, :, :] = enc

        names.append("_".join([prefix, "cat", str(cat)]))

    # create new stack
    meta = deepcopy(layer.ds.meta)
    meta["driver"] = driver
    meta["nodata"] = -99999
    meta["count"] = arr_ohe.shape[0]
    meta["dtype"] = "int32"

    with rasterio.open(file_path, mode="w", **meta) as dst:
        dst.write(arr_ohe)

    new_raster = Raster(file_path)
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    return new_raster


def xy_coordinates(layer, file_path, driver="GTiff"):
    """
    Fill 2d arrays with their x,y indices.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
        RasterLayer to use as a template.

    file_path : str
        File path to save to the resulting Raster object.s

    driver : str, options. Default is 'GTiff'
        GDAL driver to use to save raster.

    Returns
    -------
    pyspatialml.Raster object
    """

    arr = np.zeros(layer.shape, dtype=np.float32)
    arr = arr[np.newaxis, :, :]
    xyarrays = np.repeat(arr[0:1, :, :], 2, axis=0)
    xx, xy = np.meshgrid(np.arange(arr.shape[2]), np.arange(arr.shape[1]))
    xyarrays[0, :, :] = xx
    xyarrays[1, :, :] = xy

    # create new stack
    meta = deepcopy(layer.meta)
    meta["driver"] = driver
    meta["count"] = 2
    meta["dtype"] = xyarrays.dtype

    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(xyarrays)

    new_raster = Raster(file_path)
    names = ["x_coordinates", "y_coordinates"]
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    return new_raster


def rotated_coordinates(layer, file_path, n_angles=8, driver="GTiff"):
    """Generate 2d arrays with n_angles rotated coordinates.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
        RasterLayer to use as a template.

    n_angles : int, optional. Default is 8
        Number of angles to rotate coordinate system by.

    driver : str, optional. Default is 'GTiff'
        GDAL driver to use to save raster.

    Returns
    -------
    pyspatialml.Raster
    """
    # define x and y grid dimensions
    xmin, ymin, xmax, ymax = 0, 0, layer.shape[1], layer.shape[0]
    x_range = np.arange(start=xmin, stop=xmax, step=1)
    y_range = np.arange(start=ymin, stop=ymax, step=1, dtype=np.float32)

    X_var, Y_var, _ = np.meshgrid(x_range, y_range, n_angles)
    angles = np.deg2rad(np.linspace(0, 180, n_angles, endpoint=False))
    grids_directional = X_var + np.tan(angles) * Y_var

    # reorder to band, row, col order
    grids_directional = grids_directional.transpose((2, 0, 1))

    # create new stack
    meta = deepcopy(layer.meta)
    meta["driver"] = driver
    meta["count"] = n_angles
    meta["dtype"] = grids_directional.dtype
    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(grids_directional)

    new_raster = Raster(file_path)
    names = ["angle_" + str(i + 1) for i in range(n_angles)]
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    return new_raster


def distance_to_corners(layer, file_path, driver="GTiff"):
    """Generate buffer distances to corner and centre coordinates of raster
    extent.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader

    file_path : str
        File path to save to the resulting Raster object

    driver : str, optional. Default is 'GTiff'
        GDAL driver to use to save raster.

    Returns
    -------
    pyspatialml.Raster object
    """

    names = ["top_left", "top_right", "bottom_left", "bottom_right", "centre_indices"]

    rows = np.asarray(
        [0, 0, layer.shape[0] - 1, layer.shape[0] - 1, int(layer.shape[0] / 2)]
    )
    cols = np.asarray(
        [0, layer.shape[1] - 1, 0, layer.shape[1] - 1, int(layer.shape[1] / 2)]
    )

    # euclidean distances
    arr = _grid_distance(layer.shape, rows, cols)

    # create new stack
    meta = deepcopy(layer.meta)
    meta["driver"] = driver
    meta["count"] = 5
    meta["dtype"] = arr.dtype

    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(arr)

    new_raster = Raster(file_path)
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    return new_raster


def _grid_distance(shape, rows, cols):
    """Generate buffer distances to x,y coordinates.
    Parameters
    ----------
    shape : tuple
        shape of numpy array (rows, cols) to create buffer distances within.
    rows : 1d numpy array
        array of row indexes.
    cols : 1d numpy array
        array of column indexes.
    Returns
    -------
    ndarray
        3d numpy array of euclidean grid distances to each x,y coordinate pair
        [band, row, col].
    """

    # create buffer distances
    grids_buffers = np.zeros((shape[0], shape[1], rows.shape[0]), dtype=np.float32)

    for i, (y, x) in enumerate(zip(rows, cols)):
        # create 2d array (image) with pick indexes set to z
        point_arr = np.zeros((shape[0], shape[1]))
        point_arr[y, x] = 1
        buffer = ndimage.morphology.distance_transform_edt(1 - point_arr)
        grids_buffers[:, :, i] = buffer

    # reorder to band, row, column
    grids_buffers = grids_buffers.transpose((2, 0, 1))

    return grids_buffers


def distance_to_samples(layer, file_path, rows, cols, driver="GTiff"):
    """Generate buffer distances to x,y coordinates.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
        RasterLayer to use as a template.

    file_path : str
        File path to save to the resulting Raster object.

    rows : 1d numpy array
        array of row indexes.

    cols : 1d numpy array
        array of column indexes.

    driver : str, default='GTiff'
        GDAL driver to use to save raster.

    Returns
    -------
    pyspatialml.Raster object
    """
    # some checks
    if isinstance(rows, list):
        rows = np.asarray(rows)

    if isinstance(cols, list):
        cols = np.asarray(cols)

    if rows.shape != cols.shape:
        raise ValueError("rows and cols must have same dimensions")

    shape = layer.shape
    arr = _grid_distance(shape, rows, cols)

    # create new stack
    meta = deepcopy(layer.meta)
    meta["driver"] = driver
    meta["count"] = arr.shape[0]
    meta["dtype"] = arr.dtype

    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(arr)

    names = ["dist_sample" + str(i + 1) for i in range(len(rows))]
    new_raster = Raster(file_path)
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    return new_raster
