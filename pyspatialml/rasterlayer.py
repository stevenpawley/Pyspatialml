from functools import partial

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import rasterio
from rasterio.features import sieve
from rasterio.fill import fillnodata
from rasterio.windows import Window
from scipy import ndimage

import pyspatialml.base

from .utils import _get_nodata
from .plotting import discrete_cmap
from .temporary_files import _file_path_tempfile


class RasterLayer(pyspatialml.base.BaseRaster):
    """Represents a single raster band derived from a single or multi-band raster
    dataset

    Simple wrapper around a rasterio.Band object with additional methods. Used because
    the Rasterio.Band.ds.read method reads all bands from a multi-band dataset, whereas
    the RasterLayer read method only reads a single band.

    Methods encapsulated in RasterLayer objects represent those that typically would
    only be applied to a single-band of a raster, i.e. sieve-clump, distance to non-NaN
    pixels, or arithmetic operations on individual layers.

    Attributes
    ----------
    bidx : int
        The band index of the RasterLayer within the file dataset.
    
    dtype : str
        The data type of the RasterLayer.
    
    nodata : any number
        The number that is used to represent nodata pixels in the RasterLayer.
    
    file : str
        The file path to the dataset.
    
    ds : rasterio.band
        The underlying rasterio.band object.

    driver : str
        The name of the GDAL format driver.
    
    meta : dict
        A python dict storing the RasterLayer metadata.
    
    cmap : str
        The name of matplotlib map, or a custom matplotlib.cm.LinearSegmentedColormap 
        or ListedColormap object.

    norm : matplotlib.colors.Normalize (opt)
        A matplotlib.colors.Normalize to apply to the RasterLayer.
        This overides the norm attribute of the RasterLayer.
    
     : int
        Number of layers; always equal to 1.
    
    Methods
    -------
    close :
    """

    def __init__(self, band):

        # access inherited methods/attributes overridden by __init__
        super().__init__(band)

        # rasterlayer specific attributes
        self.bidx = band.bidx
        self.dtype = band.dtype
        self.nodata = band.ds.nodata
        self.file = band.ds.files[0]
        self.ds = band.ds
        self.driver = band.ds.meta["driver"]
        self.meta = band.ds.meta
        self.cmap = "viridis"
        self.norm = None
        self.categorical = False
        self.names = [self._make_name(band.ds.files[0])]
        self.count = 1
        self._close = band.ds.close
    
    def close(self):
        """Close the RasterLayer for reading/writing
        """
        self._close

    def _arith(self, function, other=None):
        """General method for performing arithmetic operations on RasterLayer objects

        Parameters
        ----------
        function : function
            Custom function that takes either one or two arrays, and returns a single
            array following a pre-defined calculation.

        other : pyspatialml.RasterLayer (optional, default None)
            If not specified, then a `function` should be provided that performs a
            calculation using only the selected RasterLayer. If `other` is specified,
            then a `function` should be supplied that takes to ndarrays as arguments
            and performs a calculation using both layers, i.e. layer1 - layer2.

        Returns
        -------
        pyspatialml.RasterLayer
            Returns a single RasterLayer containing the calculated result.
        """

        _, tfile = _file_path_tempfile(None)
        driver = self.driver

        # determine dtype of result based on calc on single pixel
        if other is not None:
            arr1 = self.read(masked=True, window=Window(0, 0, 1, 1))

            try:
                arr2 = other.read(masked=True, window=Window(0, 0, 1, 1))
            except AttributeError:
                arr2 = other

            test = function(arr1, arr2)
            dtype = test.dtype
        else:
            dtype = self.dtype

        nodata = _get_nodata(dtype)

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=1, dtype=dtype, nodata=nodata)

        with rasterio.open(tfile.name, "w", **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            self_gen = (self.read(window=w, masked=True) for w in windows)

            if isinstance(other, RasterLayer):
                other_gen = (other.read(window=w, masked=True) for w in windows)
            else:
                other_gen = (other for w in windows)

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
        """Implements behaviour for addition of two RasterLayers,
        i.e. added_layer = layer1 + layer2
        """

        def func(arr1, arr2):
            return arr1 + arr2

        return self._arith(func, other)

    def __sub__(self, other):
        """Implements behaviour for subtraction of two RasterLayers, i.e.
        subtracted_layer = layer1 - layer2
        """

        def func(arr1, arr2):
            return arr1 - arr2

        return self._arith(func, other)

    def __mul__(self, other):
        """Implements behaviour for multiplication of two RasterLayers, i.e.
        product = layer1 * layer2
        """

        def func(arr1, arr2):
            return arr1 * arr2

        return self._arith(func, other)

    def __truediv__(self, other):
        """Implements behaviour for division using `/` of two RasterLayers, i.e.
        div = layer1 / layer2
        """

        def func(arr1, arr2):
            return arr1 / arr2

        return self._arith(func, other)

    def __and__(self, other):
        """Implements & operator

        Equivalent to a intersection operation of self
        with other, i.e. intersected = layer1 & layer2.
        """

        def func(arr1, arr2):
            mask = np.logical_and(arr1, arr2).mask
            arr1.mask[mask] = True
            return arr1

        return self._arith(func, other)

    def __or__(self, other):
        """Implements | operator

        Fills gaps in self with pixels from other. Equivalent to a union operation,
        i.e. union = layer1 | layer2.
        """

        def func(arr1, arr2):
            idx = np.logical_or(arr1, arr2.mask).mask
            arr1[idx] = arr2[idx]
            return arr1

        return self._arith(func, other)

    def __xor__(self, other):
        """Exclusive OR using ^

        Equivalent to a symmetrical difference where the result comprises pixels that
        occur in self or other, but not both, i.e. xor = layer1 ^ layer2.
        """

        def func(arr1, arr2):
            mask = ~np.logical_xor(arr1, arr2)
            idx = np.logical_or(arr1, arr2.mask).mask
            arr1[idx] = arr2[idx]
            arr1.mask[np.nonzero(mask)] = True
            return arr1

        return self._arith(func, other)

    def __round__(self, ndigits):
        """Behaviour for round() function, i.e. round(layer)
        """

        def func(arr, ndigits):
            return np.round(arr, ndigits)

        func = partial(func, ndigits=ndigits)

        return self._arith(func)

    def __floor__(self):
        """Rounding down to the nearest integer using math.floor(), i.e. math.floor(layer)
        """

        def func(arr):
            return np.floor(arr)

        return self._arith(func)

    def __ceil__(self):
        """Rounding up to the nearest integer using math.ceil(), i.e. math.ceil(layer)
        """

        def func(arr):
            return np.ceil(arr)

        return self._arith(func)

    def __trunc__(self):
        """Truncating to an integral using math.trunc(), i.e. math.trunc(layer)
        """

        def func(arr):
            return np.trunc(arr)

        return self._arith(func)

    def __abs__(self):
        """abs() function as applied to a RasterLayer, i.e. abs(layer)
        """

        def func(arr):
            return np.abs(arr)

        return self._arith(func)

    def __pos__(self):
        """Unary positive, i.e. +layer1
        """

        def func(arr):
            return np.positive(arr)

        return self._arith(func)

    def __neg__(self):
        """
        Unary negative, i.e. -layer1
        """

        def func(arr):
            return np.negative(arr)

        return self._arith(func)

    def read(self, **kwargs):
        """Read method for a single RasterLayer.

        Reads the pixel values from a RasterLayer into a ndarray that always
        will have two dimensions in the order of (rows, columns).

        Parameters
        ----------
        **kwargs : named arguments that can be passed to the the
        rasterio.DatasetReader.read method.
        """
        if "resampling" in kwargs.keys():
            resampling_methods = [i.name for i in rasterio.enums.Resampling]

            if kwargs["resampling"] not in resampling_methods:
                raise ValueError(
                    "Invalid resampling method."
                    + "Resampling method must be one of {0}:".format(resampling_methods)
                )

            kwargs["resampling"] = rasterio.enums.Resampling[kwargs["resampling"]]

        return self.ds.read(indexes=self.bidx, **kwargs)

    def _write(self, arr, file_path=None, driver="GTiff", dtype=None, nodata=None, **kwargs):
        """Internal function to write processed results to a file, usually a tempfile
        """

        # generate a file path to a temporary file is file_path is None
        file_path, tfile = _file_path_tempfile(file_path)

        if dtype is None:
            dtype = self.dtype

        if nodata is None:
            nodata = _get_nodata(dtype)

        meta = self.ds.meta
        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["dtype"] = dtype
        meta.update(kwargs)

        # mask any nodata values
        arr = np.ma.masked_equal(arr, self.nodata)
        
        # replace masked values with the user-specified nodata value
        arr = arr.filled(fill_value=nodata)

        # write to file
        with rasterio.open(file_path, mode="w", **meta) as dst:
            dst.write(arr.astype(dtype), 1)

        src = rasterio.open(file_path)
        band = rasterio.band(src, 1)
        layer = pyspatialml.RasterLayer(band)

        # override RasterLayer close method if temp file is used
        if tfile is not None:
            layer.close = tfile.close

        return layer

    def write(self, file_path, driver="GTiff", dtype=None, nodata=None, **kwargs):
        """Write method for a single RasterLayer.

        Parameters
        ----------
        file_path : str (opt)
            File path to save the dataset.
        
        driver : str
            GDAL-compatible driver used for the file format.
        
        dtype : str (opt)
            Numpy dtype used for the file. If omitted then the RasterLayer's dtype is
            used.
        
        nodata : any number (opt)
            A value used to represent the nodata pixels. If omitted then the RasterLayer's 
            nodata value is used (if assigned already).
        
        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.RasterLayer
        """
        arr = self.read()
        layer = self._write(arr, file_path, driver, dtype, nodata, **kwargs)
        return layer

    def fill(
        self,
        mask=None,
        max_search_distance=100,
        smoothing_iterations=0,
        file_path=None,
        driver="GTiff",
        dtype=None,
        nodata=None,
        **kwargs
    ):
        """Fill nodata gaps in a RasterLayer

        Thin wrapper around the rasterio.fill.fillnodata method.

        Parameters
        ----------
        mask : numpy.ndarray (optional, default None)
            Optionally provide a numpy array to indice which pixels to fill. Pixels
            designated to fill should have zero values in the mask, and values > 0 in
            the mask indicate pixels to use for interpolation.
        
        max_search_distance : float (default 100)
            The maximum number of pixels in all directions to use for interpolation.
        
        smoothing_iterations : integer (default 0)
            The number of 3x3 smoothing filter passes to run. The default is 0.
        
        file_path : str (optional, default None)
            Optional path to save calculated Raster object. If not specified then a
            tempfile is used.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type.

        dtype : str (optional, default None)
            Optionally specify a numpy compatible data type when saving to file. If not
            specified, a data type is set based on the data type of the RasterLayer.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.RasterLayer
            Filled RasterLayer
        """

        arr = rasterio.fill.fillnodata(
            image=self.read(masked=True),
            mask=mask,
            max_search_distance=max_search_distance,
            smoothing_iterations=smoothing_iterations,
        )

        layer = self._write(arr, file_path, driver, dtype, nodata, **kwargs)

        return layer

    def sieve(
        self,
        size=2,
        mask=None,
        connectivity=4,
        file_path=None,
        driver="GTiff",
        nodata=None,
        dtype=None,
        **kwargs,
    ):
        """Replace pixels with their largest neighbor

        Thin wrapper around the rasterio.features.sieve method.

        Parameters
        ----------
        size : integer (default 2)
            Minimum number of contigous pixels to retain
        
        mask : ndarray (optional, default None)
            Values of False or 0 will be excluded from the sieving process
        
        connectivity : integer (default 4)
            Use 4 or 8 pixel connectivity for grouping pixels into features.
            Default is 4.

        file_path : str (optional, default None)
            Optional path to save calculated Raster object. If not
            specified then a tempfile is used.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type.

        dtype : str (optional, default None)
            Optionally specify a numpy compatible data type when saving to file. If not
            specified, a data type is set based on the data type of the RasterLayer.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.RasterLayer
            Filled RasterLayer
        """
        arr = sieve(
            source=self.read(masked=True),
            size=size,
            mask=mask,
            connectivity=connectivity,
        )

        layer = self._write(arr, file_path, driver, dtype, nodata, **kwargs)

        return layer

    def distance(self, file_path=None, driver="GTiff", nodata=None, **kwargs):
        """Calculate euclidean grid distances to non-NaN pixels

        Parameters
        ----------
        file_path : str (optional, default None)
            Optional path to save calculated Raster object. If not specified then a
            tempfile is used.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type.
        
        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.RasterLayer
            Grid distance raster
        """
        arr = self.read(masked=True)
        arr = ndimage.distance_transform_edt(1 - arr)
        dtype = arr.dtype

        layer = self._write(arr, file_path, driver, dtype, nodata, **kwargs)

        return layer

    def plot(
        self,
        cmap=None,
        norm=None,
        ax=None,
        cax=None,
        figsize=None,
        categorical=None,
        legend=False,
        vmin=None,
        vmax=None,
        fig_kwds=None,
        legend_kwds=None,
    ):
        """Plot a RasterLayer using matplotlib.pyplot.imshow

        Parameters
        ----------
        cmap : str (default None)
            The name of a colormap recognized by matplotlib.
            Overrides the cmap attribute of the RasterLayer.
        
        norm : matplotlib.colors.Normalize (opt)
            A matplotlib.colors.Normalize to apply to the RasterLayer.
            This overides the norm attribute of the RasterLayer.
        
        ax : matplotlib.pyplot.Artist (optional, default None)
            axes instance on which to draw to plot.
        
        cax : matplotlib.pyplot.Artist (optional, default None)
            axes on which to draw the legend.
        
        figsize : tuple of integers (optional, default None)
            Size of the matplotlib.figure.Figure. If the ax argument is given
            explicitly, figsize is ignored.
                
        categorical : bool (optional, default False)
            if True then the raster values will be considered to represent discrete
            values, otherwise they are considered to represent continuous values. This
            overrides the  RasterLayer 'categorical' attribute. Setting the argument
            categorical to True is ignored if the RasterLayer.categorical is already
            True.
        
        legend : bool (optional, default False)
            Whether to plot the legend.

        vmin, xmax : scale (optional, default None)
            vmin and vmax define the data range that the colormap covers. By default,
            the colormap covers the complete value range of the supplied data. vmin,
            vmax are ignored if the norm parameter is used.
        
        fig_kwds : dict (optional, default None)
            Additional arguments to pass to the matplotlib.pyplot.figure call when
            creating the figure object. Ignored if ax is passed to the plot function.

        legend_kwds : dict (optional, default None)
            Keyword arguments to pass to matplotlib.pyplot.colorbar().

        Returns
        -------
        ax : matplotlib axes instance
        """

        # some checks
        if fig_kwds is None:
            fig_kwds = {}

        if ax is None:
            if cax is not None:
                raise ValueError("'ax' can not be None if 'cax' is not.")
            fig, ax = plt.subplots(figsize=figsize, **fig_kwds)
        
        ax.set_aspect("equal")

        if norm:
            if not isinstance(norm, mpl.colors.Normalize):
                raise AttributeError(
                    "norm argument should be a matplotlib.colors.Normalize object"
                )
        
        if cmap is None:
            cmap = self.cmap
        
        if norm is None:
            norm = self.norm

        if legend_kwds is None:
            legend_kwds = {}
        
        arr = self.read(masked=True)

        if categorical is True:
            if self.categorical is False:
                N = np.bincount(arr)
                cmap = discrete_cmap(N, base_cmap=cmap)
            vmin, vmax = None, None

        im = ax.imshow(
            X=arr,
            extent=rasterio.plot.plotting_extent(self.ds),
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )

        if legend is True:
            plt.colorbar(im, cax=cax, ax=ax, **legend_kwds)

        return ax

    def _extract_by_indices(self, rows, cols):
        """Spatial query of Raster object (by-band)
        """

        X = np.ma.zeros((len(rows), self.count), dtype="float32")
        arr = self.read(masked=True)
        X[:, 0] = arr[rows, cols]

        return X
