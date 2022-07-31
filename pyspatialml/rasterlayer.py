from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.io import MemoryFile

from ._plotting import discrete_cmap
from ._rasterbase import get_nodata_value, _make_name


class RasterLayer:
    """Represents a single raster band derived from a single or
    multi-band raster dataset

    Simple wrapper around a rasterio.Band object with additional
    methods. Used because the Rasterio.Band.ds.read method reads
    all bands from a multi-band dataset, whereas the RasterLayer read
    method only reads a single band.

    Methods encapsulated in RasterLayer objects represent those that
    typically would only be applied to a single-band of a raster, i.e.
    sieve-clump, distance to non-NaN pixels, or arithmetic operations
    on individual layers.

    Attributes
    ----------
    bidx : int
        The band index of the RasterLayer within the file dataset.

    dtype : str
        The data type of the RasterLayer.

    ds : rasterio.band
        The underlying rasterio.band object.

    name : str
        A syntactically valid name for the RasterLayer.

    file : str
        The file path to the dataset.

    nodata : any number
        The number that is used to represent nodata pixels in the
        RasterLayer.

    driver : str
        The name of the GDAL format driver.

    meta : dict
        A python dict storing the RasterLayer metadata.

    transform : affine.Affine object
        The affine transform parameters.

    count : int
        Number of layers; always equal to 1.

    shape: tuple
        Shape of RasterLayer in (rows, columns)

    width, height: int
        The width (cols) and height (rows) of the dataset.

    bounds : BoundingBox named tuple
        A named tuple with left, bottom, right and top coordinates of
        the dataset.

    cmap : str
        The name of matplotlib map, or a custom
        matplotlib.cm.LinearSegmentedColormap or ListedColormap object.

    norm : matplotlib.colors.Normalize (opt)
        A matplotlib.colors.Normalize to apply to the RasterLayer.
        This overides the norm attribute of the RasterLayer.
    """

    def __init__(self, band):
        """Initiate a RasterLayer object

        Parameters
        ----------
        band : a rasterio.Band object
        """
        self.bidx = band.bidx
        self.dtype = band.dtype
        self.ds = band.ds

        if len(band.ds.files) > 0:
            description = band.ds.descriptions[band.bidx-1]
            if description is not None:
                layer_name = _make_name(band.ds.descriptions[band.bidx-1])
            else:
                layer_name = _make_name(band.ds.files[0])

            self.name = layer_name
            self.file = band.ds.files[0]

        else:
            self.name = "in_memory"
            self.file = None

        self.nodata = band.ds.nodata
        self.driver = band.ds.meta["driver"]
        self.meta = band.ds.meta
        self.transform = band.ds.transform
        self.crs = band.ds.crs
        self.count = 1
        self.shape = band.shape
        self.width = band.ds.width
        self.height = band.ds.height
        self.bounds = band.ds.bounds
        self.in_memory = False

        self.cmap = "viridis"
        self.norm = None
        self.categorical = False

    def close(self):
        self.ds.close()

    def _arith(self, function, other=None):
        """General method for performing arithmetic operations on
        RasterLayer objects

        Parameters
        ----------
        function : function
            Custom function that takes either one or two arrays, and
            returns a single array following a pre-defined calculation.

        other : pyspatialml.RasterLayer (optional, default None)
            If not specified, then a `function` should be provided that
            performs a calculation using only the selected RasterLayer.
            If `other` is specified, then a `function` should be
            supplied that takes to ndarrays as arguments and performs a
            calculation using both layers, i.e. layer1 - layer2.

        Returns
        -------
        pyspatialml.RasterLayer
            Returns a single RasterLayer containing the calculated
            result.
        """

        driver = self.driver

        if isinstance(other, RasterLayer):
            result = function(self.read(masked=True), other.read(masked=True))
        else:
            result = function(self.read(masked=True))

        nodata = get_nodata_value(result.dtype)

        # open output file with updated metadata
        meta = self.meta.copy()
        meta.update(driver=driver, count=1, dtype=result.dtype, nodata=nodata)

        with MemoryFile() as memfile:
            dst = memfile.open(**meta)
            result = np.ma.filled(result, fill_value=nodata)
            dst.write(result, indexes=1)

        # create RasterLayer from result
        layer = RasterLayer(rasterio.band(dst, 1))

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
        """Implements behaviour for division using `/` of two RasterLayers,
        i.e. div = layer1 / layer2
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

        Fills gaps in self with pixels from other. Equivalent to a union
        operation, i.e. union = layer1 | layer2.
        """

        def func(arr1, arr2):
            idx = np.logical_or(arr1, arr2.mask).mask
            arr1[idx] = arr2[idx]
            return arr1

        return self._arith(func, other)

    def __xor__(self, other):
        """Exclusive OR using ^

        Equivalent to a symmetrical difference where the result comprises
        pixels that occur in self or other, but not both, i.e.
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
        """Behaviour for round() function, i.e. round(layer)"""

        def func(arr, ndigits):
            return np.round(arr, ndigits)

        func = partial(func, ndigits=ndigits)

        return self._arith(func)

    def __floor__(self):
        """Rounding down to the nearest integer using math.floor(),
        i.e. math.floor(layer)"""

        def func(arr):
            return np.floor(arr)

        return self._arith(func)

    def __ceil__(self):
        """Rounding up to the nearest integer using math.ceil(), i.e.
        math.ceil(layer)"""

        def func(arr):
            return np.ceil(arr)

        return self._arith(func)

    def __trunc__(self):
        """Truncating to an integral using math.trunc(), i.e.
        math.trunc(layer)"""

        def func(arr):
            return np.trunc(arr)

        return self._arith(func)

    def __abs__(self):
        """abs() function as applied to a RasterLayer, i.e. abs(layer)"""

        def func(arr):
            return np.abs(arr)

        return self._arith(func)

    def __pos__(self):
        """Unary positive, i.e. +layer1"""

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

    def _stats(self, max_pixels):
        """Take a sample of pixels from which to derive per-band
        statistics."""

        rel_width = self.shape[1] / max_pixels

        if rel_width > 1:
            col_scaling = round(max_pixels / rel_width)
            row_scaling = max_pixels - col_scaling
        else:
            col_scaling = round(max_pixels * rel_width)
            row_scaling = max_pixels - col_scaling

        out_shape = (row_scaling, col_scaling)
        arr = self.read(masked=True, out_shape=out_shape)
        arr = arr.flatten()
        return arr

    def min(self, max_pixels=10000):
        """Minimum value.

        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.

        Returns
        -------
        numpy.float32
            The minimum value of the object
        """
        arr = self._stats(max_pixels)
        return np.nanmin(arr)

    def max(self, max_pixels=10000):
        """Maximum value.

        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.

        Returns
        -------
        numpy.float32
            The maximum value of the object's pixels.
        """
        arr = self._stats(max_pixels)
        return np.nanmax(arr)

    def mean(self, max_pixels=10000):
        """Mean value

        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.

        Returns
        -------
        numpy.float32
            The mean value of the object's pixels.
        """
        arr = self._stats(max_pixels)
        return np.nanmean(arr)

    def median(self, max_pixels=10000):
        """Median value

        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.

        Returns
        -------
        numpy.float32
            The medium value of the object's pixels.
        """
        arr = self._stats(max_pixels)
        return np.nanmedian(arr)

    def stddev(self, max_pixels=10000):
        """Standard deviation

        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.

        Returns
        -------
        numpy.float32
            The standard deviation of the object's pixels.
        """
        arr = self._stats(max_pixels)
        return np.nanstd(arr)

    def read(self, **kwargs):
        """Read method for a single RasterLayer.

        Reads the pixel values from a RasterLayer into a ndarray that
        always will have two dimensions in the order of (rows, columns).

        Parameters
        ----------
        **kwargs : named arguments that can be passed to the the
        rasterio.DatasetReader.read method.
        """
        if "resampling" in kwargs.keys():
            resampling_methods = [i.name for i in rasterio.enums.Resampling]

            if kwargs["resampling"] not in resampling_methods:
                raise ValueError(
                    "Invalid resampling method. Resampling "
                    "method must be one of {0}:".format(resampling_methods)
                )

            kwargs["resampling"] = rasterio.enums.Resampling[kwargs["resampling"]]

        return self.ds.read(indexes=self.bidx, **kwargs)

    def write(self, file_path, driver="GTiff", dtype=None, nodata=None, **kwargs):
        """Write method for a single RasterLayer.

        Parameters
        ----------
        file_path : str (opt)
            File path to save the dataset.

        driver : str
            GDAL-compatible driver used for the file format.

        dtype : str (opt)
            Numpy dtype used for the file. If omitted then the
            RasterLayer's dtype is used.

        nodata : any number (opt)
            A value used to represent the nodata pixels. If omitted
            then the RasterLayer's nodata value is used (if assigned
            already).

        kwargs : opt
            Optional named arguments to pass to the format drivers.
            For example can be `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.RasterLayer
        """
        if dtype is None:
            dtype = self.dtype

        if nodata is None:
            nodata = get_nodata_value(dtype)

        meta = self.ds.meta
        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["dtype"] = dtype
        meta.update(kwargs)

        # mask any nodata values
        arr = np.ma.masked_equal(self.read(), self.nodata)
        arr = arr.filled(fill_value=nodata)

        # write to file
        with rasterio.open(file_path, mode="w", **meta) as dst:
            dst.write(arr.astype(dtype), 1)

        src = rasterio.open(file_path)
        band = rasterio.band(src, 1)
        layer = RasterLayer(band)

        return layer

    def _extract_by_indices(self, rows, cols):
        """Spatial query of Raster object (by-band)"""

        X = np.ma.zeros((len(rows), self.count), dtype="float32")
        arr = self.read(masked=True)
        X[:, 0] = arr[rows, cols]

        return X

    def plot(
        self,
        cmap=None,
        norm=None,
        ax=None,
        cax=None,
        figsize=None,
        out_shape=(500, 500),
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
            This overrides the norm attribute of the RasterLayer.

        ax : matplotlib.pyplot.Artist (optional, default None)
            axes instance on which to draw to plot.

        cax : matplotlib.pyplot.Artist (optional, default None)
            axes on which to draw the legend.

        figsize : tuple of integers (optional, default None)
            Size of the matplotlib.figure.Figure. If the ax argument is
            given explicitly, figsize is ignored.

        out_shape : tuple, default=(500, 500)
            Number of rows, cols to read from the raster datasets for
            plotting.

        categorical : bool (optional, default False)
            if True then the raster values will be considered to
            represent discrete values, otherwise they are considered to
            represent continuous values. This overrides the
            RasterLayer 'categorical' attribute. Setting the argument
            categorical to True is ignored if the
            RasterLayer.categorical is already True.

        legend : bool (optional, default False)
            Whether to plot the legend.

        vmin, xmax : scale (optional, default None)
            vmin and vmax define the data range that the colormap
            covers. By default, the colormap covers the complete value
            range of the supplied data. vmin, vmax are ignored if the
            norm parameter is used.

        fig_kwds : dict (optional, default None)
            Additional arguments to pass to the
            matplotlib.pyplot.figure call when creating the figure
            object. Ignored if ax is passed to the plot function.

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
                    "norm argument should be a " "matplotlib.colors.Normalize object"
                )

        if cmap is None:
            cmap = self.cmap

        if norm is None:
            norm = self.norm

        if legend_kwds is None:
            legend_kwds = {}

        arr = self.read(masked=True, out_shape=out_shape)

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
