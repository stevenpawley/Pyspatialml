from functools import partial

import os
import re
import numpy as np
import rasterio
from rasterio.io import MemoryFile

from ._plotting import RasterLayerPlotMixin
from ._rasterstats import RasterLayerStatsMixin
from ._utils import get_nodata_value


class RasterLayer(RasterLayerStatsMixin, RasterLayerPlotMixin):
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
                layer_name = self._make_name(band.ds.descriptions[band.bidx-1])
            else:
                layer_name = self._make_name(band.ds.files[0])

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

    @staticmethod
    def _make_name(name):
        """Converts a file basename to a valid class attribute name.

        Parameters
        ----------
        name : str
            File basename for converting to a valid class attribute name.

        Returns
        -------
        valid_name : str
            Syntactically correct name of layer so that it can form a class
            instance attribute.
        """
        basename = os.path.basename(name)
        sans_ext = os.path.splitext(basename)[0]

        valid_name = sans_ext.replace(" ", "_").replace("-", "_").replace(".", "_")

        if valid_name[0].isdigit():
            valid_name = "x" + valid_name

        valid_name = re.sub(r"[\[\]\(\)\{\}\;]", "", valid_name)
        valid_name = re.sub(r"_+", "_", valid_name)

        return valid_name

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
            supplied that takes two ndarrays as arguments and performs a
            calculation using both layers, i.e. layer1 - layer2.

        Returns
        -------
        pyspatialml.RasterLayer
            Returns a single RasterLayer containing the calculated
            result.
        """

        driver = self.driver

        # if other is a RasterLayer then use the read method to get the
        # array, otherwise assume other is a scalar or array
        if isinstance(other, RasterLayer):
            result = function(self.read(masked=True), other.read(masked=True))
        else:
            result = function(self.read(masked=True), other)

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
    
    def seek(self, offset, whence=None):
        return self
    
    def tell(self):
        return self

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
