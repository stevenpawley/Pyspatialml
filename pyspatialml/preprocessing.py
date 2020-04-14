import os
from copy import deepcopy

import numpy as np
import rasterio
from rasterio.warp import reproject
from scipy import ndimage

from .raster import Raster
from .temporary_files import _file_path_tempfile


def one_hot_encode(layer, categories=None, file_path=None, driver='GTiff'):
    """One-hot encoding of a RasterLayer.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer
        Containing categories to perform one-hot encoding on.

    categories : list, ndarray, optional
        Optional list of categories to extract. Default performs one-hot encoding
        on all categorical values in the input layer.

    file_path : str, optional. Default is None
        File path to save one-hot encoded raster. If not supplied then data
        is written to a tempfile.

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
        categories = categories.data.astype('int32')

    arr_ohe = np.ma.zeros(
        (len(categories), arr.shape[0], arr.shape[1]), dtype='int32')
    names = []
    prefix = layer.names[0]

    for i, cat in enumerate(categories):
        enc = deepcopy(arr)
        enc[enc != cat] = 0
        enc[enc == cat] = 1
        arr_ohe[i, :, :] = enc

        names.append('_'.join([prefix, 'cat',  str(cat)]))

    # create new stack
    file_path, tfile = _file_path_tempfile(file_path)

    meta = deepcopy(layer.ds.meta)
    meta['driver'] = driver
    meta['nodata'] = -99999
    meta['count'] = arr_ohe.shape[0]
    meta['dtype'] = 'int32'

    with rasterio.open(file_path, mode='w', **meta) as dst:
        dst.write(arr_ohe)

    new_raster = Raster(file_path)
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    if tfile is not None:
        for layer in new_raster.iloc:
            layer.close = tfile.close

    return new_raster


def xy_coordinates(layer, file_path=None, driver='GTiff'):
    """
    Fill 2d arrays with their x,y indices.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
        RasterLayer to use as a template.
    
    file_path : str, optional. Default is None
        File path to save to the resulting Raster object. If not supplied then the
        raster is saved to a temporary file.
    
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
    file_path, tfile = _file_path_tempfile(file_path)

    meta = deepcopy(layer.meta)
    meta['driver'] = driver
    meta['count'] = 2
    meta['dtype'] = xyarrays.dtype

    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(xyarrays)
    
    new_raster = Raster(file_path)
    names = ['x_coordinates', 'y_coordinates']
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    if tfile is not None:
        for layer in new_raster.iloc:
            layer.close = tfile.close

    return new_raster


def rotated_coordinates(layer, n_angles=8, file_path=None, driver='GTiff'):
    """Generate 2d arrays with n_angles rotated coordinates.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
        RasterLayer to use as a template.

    n_angles : int, optional. Default is 8
        Number of angles to rotate coordinate system by.

    file_path : str, optional. Default is None
        File path to save to the resulting Raster object. If not supplied then the
        raster is saved to a temporary file.
    
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
    file_path, tfile = _file_path_tempfile(file_path)

    meta = deepcopy(layer.meta)
    meta['driver'] = driver
    meta['count'] = n_angles
    meta['dtype'] = grids_directional.dtype
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(grids_directional)
    
    new_raster = Raster(file_path)
    names = ['angle_' + str(i+1) for i in range(n_angles)]
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    if tfile is not None:
        for layer in new_raster.iloc:
            layer.close = tfile.close

    return new_raster


def distance_to_corners(layer, file_path=None, driver='GTiff'):
    """Generate buffer distances to corner and centre coordinates of raster extent.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
    
    file_path : str, optional. Default is None
        File path to save to the resulting Raster object. If not supplied then the
        raster is saved to a temporary file.
    
    driver : str, optional. Default is 'GTiff'
        GDAL driver to use to save raster.

    Returns
    -------
    pyspatialml.Raster object
    """

    names = [
        'top_left',
        'top_right',
        'bottom_left',
        'bottom_right',
        'centre_indices'
    ]

    rows = np.asarray(
        [0, 0, layer.shape[0]-1, layer.shape[0]-1, int(layer.shape[0]/2)])
    cols = np.asarray(
        [0, layer.shape[1]-1, 0, layer.shape[1]-1, int(layer.shape[1]/2)])

    # euclidean distances
    arr = _grid_distance(layer.shape, rows, cols)

    # create new stack
    file_path, tfile = _file_path_tempfile(file_path)

    meta = deepcopy(layer.meta)
    meta['driver'] = driver
    meta['count'] = 5
    meta['dtype'] = arr.dtype

    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(arr)
        
    new_raster = Raster(file_path)
    new_raster.rename({
        old: new for old, new in zip(new_raster.names, names)})

    if tfile is not None:
        for layer in new_raster.iloc:
            layer.close = tfile.close

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
    grids_buffers = np.zeros((shape[0], shape[1], rows.shape[0]),
                             dtype=np.float32)

    for i, (y, x) in enumerate(zip(rows, cols)):
        # create 2d array (image) with pick indexes set to z
        point_arr = np.zeros((shape[0], shape[1]))
        point_arr[y, x] = 1
        buffer = ndimage.morphology.distance_transform_edt(1 - point_arr)
        grids_buffers[:, :, i] = buffer

    # reorder to band, row, column
    grids_buffers = grids_buffers.transpose((2, 0, 1))

    return grids_buffers


def distance_to_samples(layer, rows, cols, file_path=None, driver='GTiff'):
    """Generate buffer distances to x,y coordinates.

    Parameters
    ----------
    layer : pyspatialml.RasterLayer, or rasterio.DatasetReader
    
    rows : 1d numpy array
        array of row indexes.

    cols : 1d numpy array
        array of column indexes.

    file_path : str, optional. Default=None
        File path to save to the resulting Raster object. If not supplied then the
        raster is saved to a temporary file.
    
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
        raise ValueError('rows and cols must have same dimensions')

    shape = layer.shape
    arr = _grid_distance(shape, rows, cols)

    # create new stack
    file_path, tfile = _file_path_tempfile(file_path)

    meta = deepcopy(layer.meta)
    meta['driver'] = driver
    meta['count'] = arr.shape[0]
    meta['dtype'] = arr.dtype

    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(arr)
        
    names = ['dist_sample' + str(i+1) for i in range(len(rows))]
    new_raster = Raster(file_path)
    new_raster.rename(
        {old: new for old, new in zip(new_raster.names, names)})

    if tfile is not None:
        for layer in new_raster.iloc:
            layer.close = tfile.close

    return new_raster


def reclass_nodata(input, output, src_nodata=None, dst_nodata=-99999,
                   intern=False):
    """
    Reclassify raster no data values and save to a new raster

    Args
    ----
    input : str
        File path to raster that is to be reclassified.

    output : str
        File path to output raster

    src_nodata : int or float, optional
        The source nodata value. Pixels with this value will be reclassified
        to the new dst_nodata value. If not set, it will default to the nodata value
        stored in the source image.

    dst_nodata : int or float, optional. Default is -99999
        The nodata value that the outout raster will receive after reclassifying
        pixels with the src_nodata value.

    itern : bool, optional. Default is False
        Optionally return the reclassified raster as a numpy array.

    Returns
    -------
    out: None, or 2D numpy array 
        Raster with reclassified nodata pixels.
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


def align_rasters(rasters, template, outputdir, method="Resampling.nearest"):
    """Aligns a list of rasters (paths to files) to a template raster.
    The rasters to that are to be realigned are assumed to represent
    single band raster files.

    Nodata values are also reclassified to the template raster's nodata values

    Args
    ----
    rasters : list, str
        List containing file paths to multiple rasters that are to be realigned.

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
