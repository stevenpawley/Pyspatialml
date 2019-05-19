from .__main import Raster, RasterLayer
import numpy as np
import tempfile
import rasterio
from copy import deepcopy

def one_hot_encode(layer, categories=None, file_path=None, driver='GTiff'):
    """
    One-hot encoding of a RasterLayer

    Parameters
    ----------
    layer : pyspatialml.RasterLayer
        Containing categories to perform one-hot encoding on

    categories : list, ndarray, optional
        Optional list of categories to extract. Default performs one-hot encoding
        on all categorical values in the input layer

    file_path : str, optional (default=None)
        File path to save one-hot encoded raster. If not supplied then data
        is written to a tempfile

    driver : str (default='GTiff')
        GDAL-compatible driver

    Returns
    -------
    pyspatialml.Raster
        Each categorical value is encoded as a layer with a Raster object
    """

    arr = layer.read(masked=True)

    if categories is None:
        categories = np.unique(arr)
        categories = categories[~categories.mask]
        categories = categories.data.astype('int32')

    arr_ohe = np.ma.zeros((len(categories), arr.shape[0], arr.shape[1]), dtype='int32')
    names = []
    prefix = layer.names[0]

    for i, cat in enumerate(categories):
        enc = deepcopy(arr)
        enc[enc != cat] = 0
        enc[enc == cat] = 1
        arr_ohe[i, :, :] = enc

        names.append('_'.join([prefix, 'cat',  str(cat)]))

    if file_path is None:
        file_path = tempfile.NamedTemporaryFile().name

    meta = layer.ds.meta
    meta['driver'] = driver
    meta['nodata'] = -99999
    meta['count'] = arr_ohe.shape[0]
    meta['dtype'] = 'int32'

    with rasterio.open(file_path, mode='w', **meta) as dst:
        dst.write(arr_ohe)

    new_raster = Raster(file_path)
    new_raster.rename({old: new for old, new in zip(new_raster.names, names)})

    return new_raster
