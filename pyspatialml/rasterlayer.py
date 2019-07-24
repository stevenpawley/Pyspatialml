import tempfile

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from functools import partial

import pyspatialml.base


class RasterLayer(pyspatialml.base.BaseRaster):
    """
    Represents a single rasterband derived from a single or multiband raster
    dataset

    Simple wrapper around a rasterio.Band object with additional methods. 
    Used because the Rasterio.Band.ds.read method reads all bands from a
    multiband dataset, whereas the RasterLayer read method only reads
    a single band

    Methods encapsulated in RasterLayer objects represent those that can only
    be applied to a single-band of a raster, i.e. sieve-clump, distance to
    non-NaN pixels etc.
    """

    def __init__(self, band):

        # access inherited methods/attributes overriden by __init__
        super().__init__(band)

        # rasterlayer specific attributes
        self.bidx = band.bidx
        self.dtype = band.dtype
        self.nodata = band.ds.nodata
        self.file = band.ds.files[0]
        self.driver = band.ds.meta['driver']
        self.meta = band.ds.meta
        self.ds = band.ds
        self.cmap = 'viridis'
        self.names = [self._make_name(band.ds.files[0])]
        self.count = 1
        # self.temporary_file = False

    def _arith(self, function, other=None):
        """
        General method for performing arithmetic operations on RasterLayer
        objects
        """

        file_path = tempfile.NamedTemporaryFile().name
        driver = self.driver

        # determine dtype of result based on calc on single pixel
        if other is not None:
            arr1 = self.read(masked=True, window=Window(0, 0, 1, 1))
            arr2 = other.read(masked=True, window=Window(0, 0, 1, 1))
            test = function(arr1, arr2)
            dtype = test.dtype
        else:
            dtype = self.dtype

        try:
            nodata = np.iinfo(dtype).min
        except ValueError:
            nodata = np.finfo(dtype).min

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=1, dtype=dtype, nodata=nodata)

        with rasterio.open(file_path, 'w', **meta) as dst:

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

        src = rasterio.open(file_path)
        band = rasterio.band(src, 1)

        return pyspatialml.RasterLayer(band)
    
    def __add__(self, other):
        """
        Implements behaviour for addition of two RasterLayers
        """
        def func(arr1, arr2):
            return arr1 + arr2

        return self._arith(func, other)

    def __sub__(self, other):
        """
        Implements behaviour for subtraction of two RasterLayers
        """
        def func(arr1, arr2):
            return arr1 - arr2

        return self._arith(func, other)
    
    def __mul__(self, other):
        """
        Implements behaviour for multiplication of two RasterLayers
        """
        def func(arr1, arr2):
            return arr1 * arr2

        return self._arith(func, other)

    def __truediv__(self, other):
        """
        Implements behaviour for division using `/` of two RasterLayers
        """
        def func(arr1, arr2):
            return arr1 / arr2

        return self._arith(func, other)

    def __and__(self, other):
        """
        Implements & operator. Equivalent to a intersection operation of self
        with other
        """
        def func(arr1, arr2):
            mask = np.logical_and(arr1, arr2).mask
            arr1.mask[mask] = True
            return arr1

        return self._arith(func, other)
    
    def __or__(self, other):
        """
        Implements | operator. Fills gaps in self with pixels from other.
        Equivalent to a union operation
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
        comprises pixels that occur in self or other, but not both

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
        Behaviour for round() function
        """
        def func(arr, ndigits):
            return np.round(arr, ndigits)

        func = partial(func, ndigits=ndigits)

        return self._arith(func)

    def __floor__(self):
        """
        Rounding down to the nearest integer using math.floor()
        """
        def func(arr):
            return np.floor(arr)

        return self._arith(func)

    def __ceil__(self):
        """
        Rounding up to the nearest integer using math.ceil()
        """
        def func(arr):
            return np.ceil(arr)

        return self._arith(func)

    def __trunc__(self):
        """
        Truncating to an integral using math.trunc()
        """
        def func(arr):
            return np.trunc(arr)

        return self._arith(func)

    def __abs__(self):
        """
        Abs() function as applied to a RasterLayer
        """
        def func(arr):
            return np.abs(arr)

        return self._arith(func)

    def __pos__(self):
        """
        Unary positive
        """
        def func(arr):
            return np.positive(arr)

        return self._arith(func)

    def __neg__(self):
        """
        Unary negative
        """
        def func(arr):
            return np.negative(arr)

        return self._arith(func)

    def read(self, **kwargs):
        
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

    def distance(self, file_path=None, driver='GTiff', nodata=-99999):
        """
        Calculate euclidean grid distances to non-NaN pixels

        Parameters
        ----------
        file_path : str, path to save distance raster, optional
            If not specified output is saved to a temporary file

        driver : str, default='GTiff'
            GDAL-supported driver format

        nodata : any number, optional. Default is -99999
            Value to use as the nodata value of the output raster

        Returns
        -------
        pyspatialml.RasterLayer
            Grid distance raster
        """
        arr = self.read(masked=True)
        arr = ndimage.distance_transform_edt(1 - arr)

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name

        meta = self.ds.meta
        meta['driver'] = driver
        meta['nodata'] = nodata

        with rasterio.open(file_path, mode='w', **meta) as dst:
            dst.write(arr[np.newaxis, :, :].astype('float32'))

        src = rasterio.open(file_path)
        return pyspatialml.rasterlayer.RasterLayer(rasterio.band(src, 1))

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
