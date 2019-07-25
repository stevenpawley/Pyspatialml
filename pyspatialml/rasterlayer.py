import tempfile
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from .base import _get_nodata, _file_path_tempfile

import pyspatialml.base


class RasterLayer(pyspatialml.base.BaseRaster):
    """
    Represents a single raster band derived from a single or multi-band raster
    dataset.

    Simple wrapper around a rasterio.Band object with additional methods. 
    Used because the Rasterio.Band.ds.read method reads all bands from a
    multi-band dataset, whereas the RasterLayer read method only reads
    a single band.

    Methods encapsulated in RasterLayer objects represent those that typically
    would only be applied to a single-band of a raster, i.e. sieve-clump,
    distance to non-NaN pixels, or arithmetic operations on individual layers.
    """

    def __init__(self, band):

        # access inherited methods/attributes overriden by __init__
        super().__init__(band)

        # rasterlayer specific attributes
        self.bidx = band.bidx
        self.dtype = band.dtype
        self.nodata = band.ds.nodata
        self.file = band.ds.files[0]
        self.ds = band.ds
        self.driver = band.ds.meta['driver']
        self.meta = band.ds.meta
        self.cmap = 'viridis'
        self.names = [self._make_name(band.ds.files[0])]
        self.count = 1
        self.close = band.ds.close

    def _arith(self, function, other=None):
        """
        General method for performing arithmetic operations on RasterLayer
        objects.

        Parameters
        ----------
        function : function
            Custom function that takes either one or two arrays, and
            returns a single array following a pre-defined calculation.

        other : pyspatialml.RasterLayer (opt)
            If not specified, then a `function` should be provided that performs
            a calculation using only the selected RasterLayer. If `other` is
            specified, then a `function` should be supplied that takes to
            ndarrays as arguments and performs a calculation using both layers,
            i.e. layer1 - layer2.

        Returns
        -------
        pyspatialml.RasterLayer
            Returns a single RasterLayer containing the calculated result.
        """

        tfile = tempfile.NamedTemporaryFile()
        driver = self.driver

        # determine dtype of result based on calc on single pixel
        if other is not None:
            arr1 = self.read(masked=True, window=Window(0, 0, 1, 1))
            arr2 = other.read(masked=True, window=Window(0, 0, 1, 1))
            test = function(arr1, arr2)
            dtype = test.dtype
        else:
            dtype = self.dtype

        nodata = _get_nodata(dtype)

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=1, dtype=dtype, nodata=nodata)

        with rasterio.open(tfile.name, 'w', **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            self_gen = (self.read(window=w, masked=True) for w in windows)

            if other is not None:
                other_gen = (other.read(window=w, masked=True) for w in windows)
            else:
                other_gen = (None for w in windows)

            for window, arr1, arr2 in zip(windows, self_gen, other_gen):

                if other is not None:
                    result = function(arr1, arr2)
                else:
                    result = function(arr1)

                result = np.ma.filled(result, fill_value=nodata)
                dst.write(result.astype(dtype), window=window, indexes=1)

        # create RasterLayer from result
        src = rasterio.open(tfile.name)
        band = rasterio.band(src, 1)
        layer = pyspatialml.RasterLayer(band)

        # overwrite close attribute with close method from temporaryfilewrapper
        layer.close = tfile.close

        return layer

    def __add__(self, other):
        """
        Implements behaviour for addition of two RasterLayers, i.e.
        added_layer = layer1 + layer2.
        """
        def func(arr1, arr2):
            return arr1 + arr2

        return self._arith(func, other)

    def __sub__(self, other):
        """
        Implements behaviour for subtraction of two RasterLayers, i.e.
        subtracted_layer = layer1 - layer2.
        """
        def func(arr1, arr2):
            return arr1 - arr2

        return self._arith(func, other)
    
    def __mul__(self, other):
        """
        Implements behaviour for multiplication of two RasterLayers, i.e.
        product = layer1 * layer2.
        """
        def func(arr1, arr2):
            return arr1 * arr2

        return self._arith(func, other)

    def __truediv__(self, other):
        """
        Implements behaviour for division using `/` of two RasterLayers, i.e.
        div = layer1 / layer2.
        """
        def func(arr1, arr2):
            return arr1 / arr2

        return self._arith(func, other)

    def __and__(self, other):
        """
        Implements & operator. Equivalent to a intersection operation of self
        with other, i.e. intersected = layer1 & layer2.
        """
        def func(arr1, arr2):
            mask = np.logical_and(arr1, arr2).mask
            arr1.mask[mask] = True
            return arr1

        return self._arith(func, other)
    
    def __or__(self, other):
        """
        Implements | operator. Fills gaps in self with pixels from other.
        Equivalent to a union operation, i.e. union = layer1 | layer2.
        """
        def func(arr1, arr2):
            idx = np.logical_or(arr1, arr2.mask).mask
            arr1[idx] = arr2[idx]
            return arr1

        return self._arith(func, other)
    
    def __xor__(self, other):
        """
        Exclusive OR using ^.
        Equivalent to a symmetrical difference where the result
        comprises pixels that occur in self or other, but not both, i.e.
        xor = layer1 ^ layer2.

        """
        def func(arr1, arr2):
            mask = ~np.logical_xor(arr1, arr2)
            idx = np.logical_or(arr1, arr2.mask).mask
            arr1[idx] = arr2[idx]
            arr1.mask[np.nonzero(mask)] = True
            return arr1

        return self._arith(func, other)

    def __round__(self, ndigits):
        """
        Behaviour for round() function, i.e. round(layer).
        """
        def func(arr, ndigits):
            return np.round(arr, ndigits)

        func = partial(func, ndigits=ndigits)

        return self._arith(func)

    def __floor__(self):
        """
        Rounding down to the nearest integer using math.floor(), i.e.
        math.floor(layer).
        """
        def func(arr):
            return np.floor(arr)

        return self._arith(func)

    def __ceil__(self):
        """
        Rounding up to the nearest integer using math.ceil(), i.e.
        math.ceil(layer).
        """
        def func(arr):
            return np.ceil(arr)

        return self._arith(func)

    def __trunc__(self):
        """
        Truncating to an integral using math.trunc(), i.e. math.trunc(layer).
        """
        def func(arr):
            return np.trunc(arr)

        return self._arith(func)

    def __abs__(self):
        """
        abs() function as applied to a RasterLayer, i.e. abs(layer).
        """
        def func(arr):
            return np.abs(arr)

        return self._arith(func)

    def __pos__(self):
        """
        Unary positive, i.e. +layer1.
        """
        def func(arr):
            return np.positive(arr)

        return self._arith(func)

    def __neg__(self):
        """
        Unary negative, i.e. -layer1.
        """
        def func(arr):
            return np.negative(arr)

        return self._arith(func)

    def read(self, **kwargs):
        """
        Read method for a single RasterLayer.

        Reads the pixel values from a RasterLayer into a ndarray that always
        will have two dimensions in the order of (rows, columns).

        Parameters
        ----------
        **kwargs : named arguments that can be passed to the the
        rasterio.DatasetReader.read method.

        """
        if 'resampling' in kwargs.keys():
            resampling_methods = [i.name for i in rasterio.enums.Resampling]

            if kwargs['resampling'] not in resampling_methods:
                raise ValueError(
                    'Invalid resampling method.' +
                    'Resampling method must be one of {0}:'.format(
                        resampling_methods))

            kwargs['resampling'] = rasterio.enums.Resampling[
                kwargs['resampling']]
            
        return self.ds.read(indexes=self.bidx, **kwargs)

    def fill(self):
        raise NotImplementedError

    def sieve(self):
        raise NotImplementedError

    def clump(self):
        raise NotImplementedError

    def focal(self):
        raise NotImplementedError

    def distance(self, file_path=None, driver='GTiff', nodata=None):
        """
        Calculate euclidean grid distances to non-NaN pixels.

        Parameters
        ----------
        file_path : str (opt)
            Optional path to save calculated Raster object. If not
            specified then a tempfile is used.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        nodata : any number (opt)
            Nodata value for new dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            data type.

        Returns
        -------
        pyspatialml.RasterLayer
            Grid distance raster
        """
        arr = self.read(masked=True)
        arr = ndimage.distance_transform_edt(1 - arr)
        dtype = arr.dtype

        file_path, tfile = _file_path_tempfile(file_path)

        if nodata is None:
            nodata = _get_nodata(dtype)

        meta = self.ds.meta
        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['dtype'] = dtype

        with rasterio.open(file_path, mode='w', **meta) as dst:
            dst.write(arr[np.newaxis, :, :].astype(dtype))

        src = rasterio.open(file_path)
        band = rasterio.Band(src, 1)
        layer = pyspatialml.RasterLayer(band)

        # override RasterLayer close method if temp file is used
        if tfile is not None:
            layer.close = tfile.close

        return layer

    def plot(self, **kwargs):
        """
        Plot a RasterLayer using matplotlib.pyplot.imshow
        """
        fig, ax = plt.subplots(**kwargs)
        arr = self.read(masked=True)
        im = ax.imshow(arr,
                       extent=rasterio.plot.plotting_extent(self.ds),
                       cmap=self.cmap)
        plt.colorbar(im)
        
        return fig, ax

    def _extract_by_indices(self, rows, cols):
        """
        Spatial query of Raster object (by-band)
        """

        X = np.ma.zeros((len(rows), self.count), dtype='float32')
        arr = self.read(masked=True)
        X[:, 0] = arr[rows, cols]

        return X
