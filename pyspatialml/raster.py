import os
import tempfile
from collections import OrderedDict
from collections import namedtuple
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor as Exec
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
from rasterio import features
from rasterio.io import MemoryFile
from rasterio.sample import sample_gen
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import Point
from tqdm import tqdm

from .plotting import RasterPlot
from .prediction import predict_output, predict_prob, predict_multioutput
from .rasterbase import (BaseRaster, get_nodata_value, get_num_workers,
                         _check_alignment, _fix_names)
from .rasterlayer import RasterLayer


class _LocIndexer(Mapping):
    """Access pyspatialml.RasterLayer objects by using a key.

    Represents a structure similar to a dict but allows access using a list of
    keys (not just a single key).

    Methods
    -------
    __get_item__ : str, or iterable of str
        Defines the subset method for the _LocIndexer. Allows the contained
        RasterLayer objects to be subset using a either single, or multiple
        labels corresponding to the names of each RasterLayer. Returns a
        RasterLayer if only a single item is subset, or a Raster if multiple
        items are subset.

    __set_item__ : key, value
        Allows a RasterLayer object to be assigned to a name within a Raster
        object. This automatically updates the indexer with the layer, and
        adds the RasterLayer's name as an attribute in the Raster.

    pop: key
        Return a single RasterLayer and delete it from the indexer. This
        also removes the attribute from the parent raster object that refers
        to the layer.

    rename : old, new
        Rename a RasterLayer from `old` to `new. This method renames the
        layer in the indexer and renames the equivalent attribute in the
        parent Raster object.
    """

    def __init__(self, raster, *args, **kw):
        """
        Initiate a _LocIndexer class

        Parameters
        ----------
        raster : pyspatialml.Raster object
            Pass the parent raster object to the _LocIndexer so that the
            parents attributes are kept up-to-date with change to the
            individual RasterLayers in the _LocIndexer.
        """
        self.raster = raster
        self._dict = OrderedDict(*args, **kw)

    def __getitem__(self, keys):
        if isinstance(keys, str):
            selected = self._dict[keys]
        else:
            selected = [self._dict[i] for i in keys]
            selected = Raster(selected)

        return selected

    def __setitem__(self, key, value):
        self._dict[key] = value
        setattr(self.raster, key, value)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def pop(self, key):
        pop = self._dict.pop(key)
        delattr(self.raster, key)
        return pop

    def rename(self, old, new):
        # rename the index
        self._dict = OrderedDict(
            [(new, v) if k == old else (k, v) for k, v in self._dict.items()]
        )

        # update the parent raster's attributes
        delattr(self.raster, old)
        setattr(self.raster, new, self._dict[new])

        # update the internal name of the RasterLayer
        self.raster[new].name = new


class _iLocIndexer(object):
    """Access pyspatialml.RasterLayer objects using an index position

    A wrapper around _LocIndexer to enable integer-based indexing of the items
    in the OrderedDict. Setting and getting items can occur using a single
    index position, a list or tuple of positions, or a slice of positions.

    Methods
    -------
    __getitem__ : index
        Subset RasterLayers using an integer index, a slice of indexes, or a
        list/tuple of indexes. Returns a RasterLayer is a single item is
        subset, or a Raster if multiple layers are subset.

    __setitem__ : index, value
        Assign a RasterLayer to a index position within the indexer. The index
        can be a single integer position, a slice of positions, or a list/tuple
        of positions. This method also updates the parent Raster object's
        attributes with the names of the new RasterLayers that were passed
        as the value.
    """

    def __init__(self, raster, loc_indexer):
        """Initiate a _iLocIndexer

        Parameters
        ----------
        raster : pyspatialml.Raster
            The parent Raster object. The _LocIndexer requires the parent
            Raster object so that the raster's attributes are kept up to date
            with changes to the RasterLayers in the dict.

        loc_indexer : pyspatialml.raster._LocIndexer
            An instance of a _LocIndexer.
        """
        self.raster = raster
        self._index = loc_indexer

    def __setitem__(self, index, value):
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            self._index[key] = value
            setattr(self.raster, key, value)

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if start is None:
                start = 0
            if stop is None:
                stop = self.raster.count
            if step is None:
                step = 1

            index = list(range(start, stop, step))

        if isinstance(index, (list, tuple)):
            for i, v in zip(index, value):
                key = list(self._index.keys())[i]
                self._index[key] = v
                setattr(self.raster, key, v)

    def __getitem__(self, index):
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            selected = self._index[key]

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if start is None:
                start = 0

            if stop is None:
                stop = self.raster.count

            if step is None:
                step = 1

            index = list(range(start, stop, step))

        if isinstance(index, (list, tuple)):
            key = []
            for i in index:
                key.append(list(self._index.keys())[i])
            selected = Raster([self._index[k] for k in key])

        return selected


class Raster(RasterPlot, BaseRaster):
    """Creates a collection of file-based GDAL-supported raster datasets that
    share a common coordinate reference system and geometry.

    Raster objects encapsulate RasterLayer objects, which represent single band
    raster datasets that can physically be represented by either separate
    single-band raster files, multi-band raster files, or any combination of
    individual bands from multi-band raster and single-band raster datasets.

    Methods defined in a Raster class comprise those that would typically
    applied to a stack of raster datasets. In addition, these methods always
    return a new Raster object.

    Attributes
    ----------
    loc : _LocIndexer object
        Access pyspatialml.RasterLayer objects within a Raster using a key
        or a list of keys.

    iloc : _ILocIndexer object
        Access pyspatialml.RasterLayer objects using an index position. A
        wrapper around _LocIndexer to enable integer-based indexing of the
        items in the OrderedDict. Setting and getting items can occur using
        a single index position, a list or tuple of positions, or a slice
        of positions.

    files : list
        A list of the raster dataset files that are used in the Raster.
        This does not have to be the same length as the number of
        RasterLayers because some files may have multiple bands.

    dtypes : list
        A list of numpy dtypes for each RasterLayer.

    nodatavals : list
        A list of the nodata values for each RasterLayer.

    count : int
        The number of RasterLayers in the Raster.

    res : tuple
        The resolution in (x, y) dimensions of the Raster.

    meta : dict
        A dict containing the raster metadata. The dict contains the
        following keys/values:

        crs : the crs object
        transform : the Affine.affine transform object
        width : width of the Raster in pixels
        height : height of the Raster in pixels
        count : number of RasterLayers within the Raster
        dtype : the numpy datatype that represents lowest common
                denominator of the different dtypes for all of the layers
                in the Raster.

    names : list
        A list of the RasterLayer names.

    block_shape : tuple
        The default block_shape in (rows, cols) for reading windows of data
        in the Raster for out-of-memory processing.

    tempdir : str, default is tempfile.tempdir
        Path to a directory to store temporary files that are produced
        during geoprocessing operations.

    in_memory : bool, default is False
        Whether to initiate the Raster from an array and store the data
        in-memory using Rasterio's in-memory files.
    """

    def __init__(self, src, crs=None, transform=None, nodata=None, mode="r",
                 file_path=None, driver=None, tempdir=tempfile.tempdir,
                 in_memory=False):
        """Initiate a new Raster object

        Parameters
        ----------
        src : file path, RasterLayer, rasterio dataset, or a ndarray
            Initiate a Raster object from any combination of a file path or
            list of file paths to GDAL-supported raster datasets, RasterLayer
            objects, or directly from a rasterio dataset or band object that is
            opened in 'r' or 'rw' mode.

            A Raster object can also be created directly from a numpy array
            in [band, rows, cols] order. The dditional arguments `crs` and
            `transform` should also be provided to supply spatial coordinate
            information.

        crs : rasterio.crs.CRS object (optional, default is None)
            CRS object containing projection information for data if provided
            by the associated `arr` parameter.

        transform : affine.Affine object (optional, default is None)
            Affine object containing transform information for data if provided
            by the associated `arr` parameter.

        nodata : any number (optional, default is None)
            Assign a nodata value to the Raster dataset when `src` is a
            ndarray. If a nodata value is not specified then it is
            determined based on the minimum permissible value for the array's
            data type.

        mode : str (default 'r')
            Mode used to open the raster datasets. Can be one of 'r', 'r+' or
            'w+'.

        file_path : str (optional, default None)
            Path to save new Raster object if created from an array.

        Returns
        -------
        pyspatialml.Raster
            Raster object containing the src layers stacked into a single
            object
        """
        super().__init__()

        self.loc = _LocIndexer(self)
        self.iloc = _iLocIndexer(self, self.loc)
        self.files = []
        self.dtypes = []
        self.nodatavals = []
        self.res = None
        self.tempdir = tempdir
        src_layers = []

        # check mode
        if mode not in ["r", "r+", "w", "w+"]:
            raise ValueError("mode must be one of 'r', 'r+', 'w', or 'w+'")

        # check mode for writing to file from ndarray
        if isinstance(src, np.ndarray) and in_memory is False and mode != "w+":
            raise ValueError(
                "mode must be 'w' or 'w+' to initiate from an ndarray")

        # get temporary file name if file_path is None
        if file_path is None:
            file_path, tfile = self._tempfile(file_path)
            driver = "GTiff"

        # initiate from numpy array
        if isinstance(src, np.ndarray):
            if src.ndim == 2:
                src = src[np.newaxis]

            count, height, width = src.shape

            if in_memory is True:
                memfile = MemoryFile()
                dst = memfile.open(height=height, width=width, count=count,
                                   driver=driver, dtype=src.dtype, crs=crs,
                                   transform=transform, nodata=nodata)
                dst.write(src)

            else:
                dst = rasterio.open(file_path, mode=mode, driver=driver,
                                    height=height, width=width,
                                    count=count, dtype=src.dtype, crs=crs,
                                    transform=transform, nodata=nodata)
                dst.write(src)

            for i in range(dst.count):
                band = rasterio.band(dst, i + 1)
                src_layers.append(RasterLayer(band))

            if tfile is not None and in_memory is False:
                for layer in src_layers:
                    layer._close = tfile.close

        # from a single file path
        elif isinstance(src, str):
            r = rasterio.open(src, mode=mode, driver=driver)
            for i in range(r.count):
                band = rasterio.band(r, i + 1)
                src_layers.append(RasterLayer(band))

        # from a list of file paths
        elif isinstance(src, list) and all(isinstance(x, str) for x in src):
            for f in src:
                r = rasterio.open(f, mode=mode, driver=driver)
                for i in range(r.count):
                    band = rasterio.band(r, i + 1)
                    src_layers.append(RasterLayer(band))

        # from a single RasterLayer
        elif isinstance(src, RasterLayer):
            src_layers = src

        # from a single or list of RasterLayer objects
        elif (isinstance(src, list) and
              all(isinstance(x, RasterLayer) for x in src)):
            src_layers = src

        # from a single rasterio.io.datasetreader/writer
        elif isinstance(src, (rasterio.io.DatasetReader, rasterio.io.DatasetWriter)):
            for i in range(src.count):
                band = rasterio.band(src, i + 1)
                src_layers.append(RasterLayer(band))

        # from a list of rasterio.io.datasetreader
        elif (isinstance(src, list) and 
              all(isinstance(x, (rasterio.io.DatasetReader, rasterio.io.DatasetWriter))
              for x in src)):
            for r in src:
                for i in range(r.count):
                    band = rasterio.band(r, i + 1)
                    src_layers.append(RasterLayer(band))

        # from a single rasterio.band objects
        elif isinstance(src, rasterio.Band):
            src_layers = RasterLayer(src)

        elif (isinstance(src, list) and
              all(isinstance(x, rasterio.Band) for x in src)):
            for band in src:
                src_layers.append(RasterLayer(band))

        else:
            raise ValueError("Cannot initiated a Raster from `src`")

        # call property with a list of rasterio.band objects
        self._layers = src_layers

    def __getitem__(self, key):
        """Subset the Raster object using a label or list of labels.

        Parameters
        ----------
        key : str, or list of str
            Key-based indexing of RasterLayer objects within the Raster.

        Returns
        -------
        Raster
            Returns a new Raster object with the with the subset selection of
            layers.
        """
        # return a RasterLayer if a single layer is selected
        if isinstance(key, str):
            selected = self.loc[key]

        # return a Raster object if multiple layers are subset
        else:
            selected = []
            for i in key:
                if i in self.names is False:
                    raise KeyError("key not present in Raster object")
                else:
                    selected.append(self.loc[i])
            selected = Raster(selected)
        return selected

    def __setitem__(self, key, value):
        """Replace a RasterLayer within the Raster object with a new
        RasterLayer.

        Note that this modifies the Raster object in place.

        Parameters
        ----------
        key : str
            Key-based index of layer to be replaced.

        value : pyspatialml.RasterLayer
            RasterLayer object to use for replacement.
        """
        if isinstance(value, RasterLayer):
            self.loc[key] = value
            self.iloc[self.names.index(key)] = value
            setattr(self, key, value)
        else:
            raise ValueError("value is not a RasterLayer object")

    def __iter__(self):
        return iter(self.loc.items())

    @property
    def _layers(self):
        return self.loc

    @_layers.setter
    def _layers(self, layers):
        # some checks
        if isinstance(layers, RasterLayer):
            layers = [layers]

        if all(isinstance(x, type(layers[0])) for x in layers) is False:
            raise ValueError(
                "Cannot create a Raster object from a mixture of input types")

        meta = _check_alignment(layers)

        if meta is False:
            raise ValueError(
                "Raster datasets do not have the same dimensions or transform")

        # reset existing attributes
        for name in self.names:
            delattr(self, name)

        self.loc = _LocIndexer(self)
        self.iloc = _iLocIndexer(self, self.loc)
        self.files = []
        self.dtypes = []
        self.nodatavals = []

        # update global Raster object attributes with new values
        self.count = len(layers)
        self.width = meta["width"]
        self.height = meta["height"]
        self.shape = (self.height, self.width)
        self.transform = meta["transform"]
        self.res = (abs(meta["transform"].a), abs(meta["transform"].e))
        self.crs = meta["crs"]

        bounds = rasterio.transform.array_bounds(self.height, self.width,
                                                 self.transform)
        BoundingBox = namedtuple("BoundingBox",
                                 ["left", "bottom", "right", "top"])
        self.bounds = BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])

        names = [i.name for i in layers]
        names = _fix_names(names)

        # update attributes per dataset
        for layer, name in zip(layers, names):
            self.dtypes.append(layer.dtype)
            self.nodatavals.append(layer.nodata)
            self.files.append(layer.file)
            layer.name = name
            self.loc[name] = layer
            setattr(self, name, self.loc[name])

        self.meta = dict(crs=self.crs, transform=self.transform,
                         width=self.width, height=self.height,
                         count=self.count,
                         dtype=np.find_common_type(self.dtypes, []))

    def read(self, masked=False, window=None, out_shape=None,
             resampling="nearest", as_df=False, **kwargs):
        """Reads data from the Raster object into a numpy array.

        Parameters
        ----------
        masked : bool (default False)
            Read data into a masked array.

        window : rasterio.window.Window object (optional, default None)
            Tuple of col_off, row_off, width, height of a window of data to
            read a chunk of data into a ndarray.

        out_shape : tuple (optional, default None)
            Shape of shape of array (rows, cols) to read data into using
            decimated reads.

        resampling : str (default 'nearest')
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max',
            'med', 'min', 'mode', 'q1', 'q3'.

        as_df : bool (default False)
            Whether to return the data as a pandas.DataFrame with columns
            named by the RasterLayer names.

        **kwargs : dict
            Other arguments to pass to rasterio.DatasetReader.read method

        Returns
        -------
        ndarray
            Raster values in 3d ndarray  with the dimensions in order of
            (band, row, and column).
        """

        dtype = self.meta["dtype"]

        # get window to read from window or height/width of dataset
        if window is None:
            width = self.width
            height = self.height
        else:
            width = window.width
            height = window.height

        # decimated reads using nearest neighbor resampling
        if out_shape:
            height, width = out_shape

        # read bands separately into numpy array
        if masked is True:
            arr = np.ma.zeros((self.count, height, width), dtype=dtype)
        else:
            arr = np.zeros((self.count, height, width), dtype=dtype)

        for i, layer in enumerate(self.iloc):
            arr[i, :, :] = layer.read(masked=masked, window=window,
                                      out_shape=out_shape,
                                      resampling=resampling, **kwargs)

            if masked is True:
                arr[i, :, :] = np.ma.MaskedArray(
                    data=arr[i, :, :], mask=np.isfinite(arr[i, :, :]).mask)

        if as_df is True:
            # reshape to rows, cols, bands
            arr = arr.transpose(1, 2, 0)
            arr_flat = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
            df = pd.DataFrame(data=arr_flat, columns=self.names)
            return df

        return arr

    def write(self, file_path, driver="GTiff", dtype=None, nodata=None,
              **kwargs):
        """Write the Raster object to a file.

        Overrides the write RasterBase class method, which is a partial
        function of the rasterio.DatasetReader.write method.

        Parameters
        ----------
        file_path : str
            File path used to save the Raster object.

        driver : str (default is 'GTiff').
            Name of GDAL driver used to save Raster data.

        dtype : str (opt, default None)
            Optionally specify a numpy compatible data type when saving to
            file. If not specified, a data type is selected based on the data
            types of RasterLayers in the Raster object.

        nodata : any number (opt, default None)
            Optionally assign a new nodata value when saving to file. If not
            specified a nodata value based on the minimum permissible value for
            the data types of RasterLayers in the Raster object is used. Note
            that this does not change the pixel nodata values of the raster,
            it only changes the metadata of what value represents a nodata
            pixel.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            New Raster object from saved file.
        """
        dtype = self._check_supported_dtype(dtype)

        if nodata is None:
            nodata = get_nodata_value(dtype)

        meta = self.meta.copy()
        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["dtype"] = dtype
        meta.update(kwargs)

        with rasterio.open(file_path, mode="w", **meta) as dst:

            for i, layer in enumerate(self.iloc):
                arr = layer.read()
                arr[arr == layer.nodata] = nodata
                dst.write(arr.astype(dtype), i + 1)

        return self._new_raster(file_path, self.names)

    def predict_proba(self, estimator, file_path=None, in_memory=False,
                      indexes=None, driver="GTiff", dtype=None, nodata=None,
                      progress=False, n_jobs=1, **kwargs):
        """Apply class probability prediction of a scikit learn model to a
        Raster.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str (optional, default None)
            Path to a GeoTiff raster for the prediction results. If not
            specified then the output is written to a temporary file.

        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        indexes : list of integers (optional, default None)
            List of class indices to export. In some circumstances, only a
            subset of the class probability estimations are desired, for
            instance when performing a binary classification only the
            probabilities for the positive class may be desired.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Optionally specify a GDAL compatible data type when saving to file.
            If not specified, a data type is set based on the data type of the
            prediction.

        nodata : any number (optional, default None)
            Nodata value for file export. If not specified then the nodata
            value is derived from the minimum permissible value for the given
            data type.

        n_jobs : int (default 1)
            Number of processing cores to use for parallel execution. Default
            is n_jobs=1. -1 is all cores; -2 is all cores -1.

        progress : bool (default False)
            Show progress bar for prediction.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster containing predicted class probabilities. Each predicted
            class is represented by a RasterLayer object. The RasterLayers are
            named `prob_n` for 1,2,3..n, with `n` based on the index position
            of the classes, not the number of the class itself.

            For example, a classification model predicting classes with integer
            values of 1, 3, and 5 would result in three RasterLayers named
            'prob_1', 'prob_2' and 'prob_3'.
        """
        # some checks
        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)

        n_jobs = get_num_workers(n_jobs)
        probfun = partial(predict_prob, estimator=estimator)

        if isinstance(indexes, int):
            indexes = range(indexes, indexes + 1)

        elif indexes is None:
            window = next(self.block_shapes(*self.block_shape))
            img = self.read(masked=True, window=window)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape(
                (n_samples, n_features))
            flat_pixels = flat_pixels.filled(0)
            result = estimator.predict_proba(flat_pixels)
            indexes = np.arange(0, result.shape[1])

        # check dtype and nodata
        if dtype is None:
            dtype = self._check_supported_dtype(result)
        else:
            dtype = self._check_supported_dtype(dtype)

        if nodata is None:
            nodata = get_nodata_value(dtype)

        # open output file with updated metadata
        meta = self.meta.copy()
        count = len(indexes)
        meta.update(driver=driver, count=count, dtype=dtype, nodata=nodata)
        meta.update(kwargs)

        # get windows
        windows = [w for w in self.block_shapes(*self.block_shape)]
        data_gen = ((w, self.read(window=w, masked=True)) for w in windows)

        # apply prediction function
        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                with Exec(max_workers=n_jobs) as executor:
                    for w, res, pbar in zip(
                            windows,
                            executor.map(probfun, data_gen),
                            tqdm(windows, disable=not progress)):
                        res = np.ma.filled(res, fill_value=nodata)
                        dst.write(res[indexes, :, :].astype(dtype), window=w)
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(
                    height=meta["height"],
                    width=meta["width"],
                    count=meta["count"],
                    dtype=meta["dtype"],
                    crs=meta["crs"],
                    transform=meta["transform"],
                    nodata=meta["nodata"],
                    driver=driver
                )

                with Exec(max_workers=n_jobs) as executor:
                    for w, res, pbar in zip(
                            windows,
                            executor.map(probfun, data_gen),
                            tqdm(windows, disable=not progress)):
                        res = np.ma.filled(res, fill_value=nodata)
                        dst.write(res[indexes, :, :].astype(dtype), window=w)

            file_path = dst

        # generate layer names
        prefix = "prob_"
        names = [prefix + str(i) for i in range(len(indexes))]

        # create new Raster object with the result
        new_raster = self._new_raster(file_path, names)

        # override close method
        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def predict(self, estimator, file_path=None, in_memory=False,
                driver="GTiff", dtype=None, nodata=None, n_jobs=1,
                progress=False, **kwargs):
        """Apply prediction of a scikit learn model to a Raster.

        The model can represent any scikit learn model or compatible api with a
        `fit` and `predict` method. These can consist of classification or
        regression models. Multi-class classifications and multi-target
        regressions are also supported.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str (optional, default None)
            Path to a GeoTiff raster for the prediction results. If not
            specified then the output is written to a temporary file.

        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export

        dtype : str (optional, default None)
            Optionally specify a GDAL compatible data type when saving to file.
            If not specified, np.float32 is assumed.

        nodata : any number (optional, default None)
            Nodata value for file export. If not specified then the nodata
            value is derived from the minimum permissible value for the given
            data type.

        n_jobs : int (default 1)
            Number of processing cores to use for parallel execution. Default
            is n_jobs=1. -1 is all cores; -2 is all cores -1.

        progress : bool (default False)
            Show progress bar for prediction.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster object containing prediction results as a RasterLayers. For
            classification and regression models, the Raster will contain a
            single RasterLayer, unless the model is multi-class or
            multi-target. Layers are named automatically as `pred_raw_n` with
            n = 1, 2, 3 ..n.
        """
        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)

        n_jobs = get_num_workers(n_jobs)

        # determine output count for multi-class or multi-target cases
        window = next(self.block_shapes(*self.block_shape))
        img = self.read(masked=True, window=window)
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
        flat_pixels = flat_pixels.filled(0)
        result = estimator.predict(flat_pixels)

        if result.ndim > 1:
            n_outputs = result.shape[result.ndim - 1]
        else:
            n_outputs = 1

        indexes = np.arange(0, n_outputs)

        # chose prediction function
        if len(indexes) == 1:
            predfun = partial(predict_output, estimator=estimator)
        else:
            predfun = partial(predict_multioutput, estimator=estimator)

        # check dtype and nodata
        if dtype is None:
            dtype = self._check_supported_dtype(result)
        else:
            dtype = self._check_supported_dtype(dtype)

        if nodata is None:
            nodata = get_nodata_value(dtype)

        # open output file with updated metadata
        meta = self.meta.copy()
        count = len(indexes)
        meta.update(driver=driver, count=count, dtype=dtype, nodata=nodata)
        meta.update(kwargs)

        # get windows
        windows = [w for w in self.block_shapes(*self.block_shape)]
        data_gen = ((w, self.read(window=w, masked=True)) for w in windows)

        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                with Exec(max_workers=n_jobs) as executor:
                    for w, res, pbar in zip(
                            windows,
                            executor.map(predfun, data_gen),
                            tqdm(windows, disable=not progress)):
                        res = np.ma.filled(res, fill_value=nodata)
                        dst.write(res[indexes, :, :].astype(dtype), window=w)
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(
                    height=meta["height"],
                    width=meta["width"],
                    count=meta["count"],
                    dtype=meta["dtype"],
                    crs=meta["crs"],
                    driver=driver,
                    transform=meta["transform"],
                    nodata=meta["nodata"]
                )
                with Exec(max_workers=n_jobs) as executor:
                    for w, res, pbar in zip(
                            windows,
                            executor.map(predfun, data_gen),
                            tqdm(windows, disable=not progress)):
                        res = np.ma.filled(res, fill_value=nodata)
                        dst.write(res[indexes, :, :].astype(dtype), window=w)
            file_path = dst

        # generate layer names
        prefix = "pred_raw_"
        names = [prefix + str(i) for i in range(len(indexes))]

        # create new Raster object with the result
        new_raster = self._new_raster(file_path, names)

        # override close method
        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def append(self, other, in_place=False):
        """Method to add new RasterLayers to a Raster object.

        Note that this modifies the Raster object in-place by default.

        Parameters
        ----------
        other : Raster object, or list of Raster objects
            Object to append to the Raster.

        in_place : bool (default False)
            Whether to change the Raster object in-place or leave original and
            return a new Raster object.

        Returns
        -------
        Raster
            Returned only if `in_place` is False
        """

        if isinstance(other, Raster):
            other = [other]

        for new_raster in other:

            if not isinstance(new_raster, Raster):
                raise AttributeError(
                    new_raster + " is not a pyspatialml.Raster object")

            # check that other raster does not result in duplicated names
            combined_names = self.names + new_raster.names
            combined_names = _fix_names(combined_names)

            # update layers and names
            combined_layers = list(self.loc.values()) + list(
                new_raster.loc.values())

            for layer, name in zip(combined_layers, combined_names):
                layer.names = [name]

            if in_place is True:
                self._layers = combined_layers
            else:
                new_raster = self._new_raster(self.files, self.names)
                new_raster._layers = combined_layers

                return new_raster

    def drop(self, labels, in_place=False):
        """Drop individual RasterLayers from a Raster object

        Note that this modifies the Raster object in-place by default.

        Parameters
        ---------
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or
            label, or a list of integers or labels.

        in_place : bool (default False)
            Whether to change the Raster object in-place or leave original and
            return a new Raster object.

        Returns
        -------
        pyspatialml.Raster
            Returned only if `in_place` is True
        """

        # convert single label to list
        if isinstance(labels, (str, int)):
            labels = [labels]

        # numerical index based subsetting
        if len([i for i in labels if isinstance(i, int)]) == len(labels):
            subset_layers = [v for (i, v) in enumerate(list(self.loc.values()))
                             if i not in labels]

        # str label based subsetting
        elif len([i for i in labels if isinstance(i, str)]) == len(labels):
            subset_layers = [v for (i, v) in enumerate(list(self.loc.values()))
                             if self.names[i] not in labels]

        else:
            raise ValueError(
                "Cannot drop layers based on mixture of indexes and labels")

        if in_place is True:
            self._layers = subset_layers
        else:
            new_raster = self._new_raster(self.files, self.names)
            new_raster._layers = subset_layers

            return new_raster

    def rename(self, names, in_place=False):
        """Rename a RasterLayer within the Raster object.

        Parameters
        ----------
        names : dict
            dict of old_name : new_name

        in_place : bool (default False)
            Whether to change names of the Raster object in-place or leave
            original and return a new Raster object.

        Returns
        -------
        pyspatialml.Raster
            Returned only if `in_place` is False
        """

        if in_place is True:
            for old_name, new_name in names.items():
                # change internal name of RasterLayer
                self.loc[old_name].name = new_name

                # change name of layer in stack
                self.loc.rename(old_name, new_name)
        else:
            new_raster = self._new_raster(self.files, self.names)
            for old_name, new_name in names.items():
                # change internal name of RasterLayer
                new_raster.loc[old_name].name = new_name

                # change name of layer in stack
                new_raster.loc.rename(old_name, new_name)

            return new_raster

    def _new_raster(self, file_path, names=None):
        """Return a new Raster object

        Parameters
        ----------
        file_path : str
            Path to files to create the new Raster object from.

        names : list (optional, default None)
            List to name the RasterLayer objects in the stack. If not supplied
            then the names will be generated from the file names.

        Returns
        -------
        pyspatialml.Raster
        """

        # some checks
        if not isinstance(file_path, list):
            file_path = [file_path]

        # create new raster from supplied file path
        raster = Raster(file_path)

        # rename and copy attributes
        if names is not None:
            rename = {old: new for old, new in zip(raster.names, names)}
            raster.rename(rename, in_place=True)

        for old_layer, new_layer in zip(self.iloc, raster.iloc):
            new_layer.cmap = old_layer.cmap
            new_layer.norm = old_layer.norm
            new_layer.categorical = old_layer.categorical

        raster.block_shape = self.block_shape

        return raster

    def copy(self):
        """Creates a shallow copy of a Raster object

        Note that shallow in the context of a Raster object means that an
        immutable copy of the object is made, however the on-disk file
        locations remain the same.

        Returns
        -------
        Raster
        """
        new_raster = Raster(self.files)
        rename = {old: new for old, new in zip(new_raster.names, self.names)}
        new_raster.rename(rename, in_place=True)

        for old_layer, new_layer in zip(self.iloc, new_raster.iloc):
            new_layer.cmap = old_layer.cmap
            new_layer.norm = old_layer.norm
            new_layer.categorical = old_layer.categorical

        new_raster.block_shape = self.block_shape

        return new_raster

    def mask(self, shapes, invert=False, crop=True, pad=False, file_path=None,
             in_memory=False, driver="GTiff", dtype=None, nodata=None, 
             **kwargs):
        """Mask a Raster object based on the outline of shapes in a
        geopandas.GeoDataFrame

        Parameters
        ----------
        shapes : geopandas.GeoDataFrame
            GeoDataFrame containing masking features.

        invert : bool (default False)
            If False then pixels outside shapes will be masked. If True then
            pixels inside shape will be masked.

        crop : bool (default True)
            Crop the raster to the extent of the shapes.

        pad : bool (default False)
            If True, the features will be padded in each direction by one half
            of a pixel prior to cropping raster.

        file_path : str (optional, default None)
            File path to save to resulting Raster. If not supplied then the
            resulting Raster is saved to a temporary file
        
        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then
            the cropped Raster is created using the existing dtype, which uses
            a dtype that can accommodate the data types of all of the
            individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for cropped dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            data type. Note that this changes the values of the pixels to the
            new nodata value, and changes the metadata of the raster.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.
        """

        # some checks
        if invert is True:
            crop = False

        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)
        
        meta = self.meta.copy()

        dtype = self._check_supported_dtype(dtype)

        if nodata is None:
            nodata = get_nodata_value(dtype)

        meta["dtype"] = dtype

        masked_ndarrays = []

        for layer in self.iloc:
            # set pixels outside of mask to raster band's nodata value
            masked_arr, transform = rasterio.mask.mask(
                dataset=layer.ds,
                shapes=[shapes.geometry.unary_union],
                filled=False,
                invert=invert,
                crop=crop,
                pad=pad,
            )

            if layer.ds.count > 1:
                masked_arr = masked_arr[layer.bidx - 1, :, :]

            else:
                masked_arr = np.ma.squeeze(masked_arr)

            masked_ndarrays.append(masked_arr)

        # stack list of 2d arrays into 3d array
        masked_ndarrays = np.ma.stack(masked_ndarrays)

        # write to file
        meta["transform"] = transform
        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["height"] = masked_ndarrays.shape[1]
        meta["width"] = masked_ndarrays.shape[2]
        meta.update(kwargs)
        masked_ndarrays = masked_ndarrays.filled(fill_value=nodata)

        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                dst.write(masked_ndarrays.astype(dtype))
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(**meta)
                dst.write(masked_ndarrays.astype(dtype))
            
            file_path = dst

        # create new Raster object with the result
        new_raster = self._new_raster(file_path, self.names)

        # override close method
        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def intersect(self, file_path=None, in_memory=False, driver="GTiff", 
                  dtype=None, nodata=None, **kwargs):
        """Perform a intersect operation on the Raster object.

        Computes the geometric intersection of the RasterLayers with the Raster
        object. This will cause nodata values in any of the rasters to be
        propagated through all of the output rasters.

        Parameters
        ----------
        file_path : str (optional, default None)
            File path to save to resulting Raster. If not supplied then the
            resulting Raster is saved to a temporary file.

        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new intersected Raster is created using the dtype of the
            existing Raster dataset, which uses a dtype that can accommodate
            the data types of all of the individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value
            is set based on the minimum permissible value of the Raster's data
            type. Note that this changes the values of the pixels that
            represent nodata to the new value.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster with layers that are masked based on a union of all masks in
            the suite of RasterLayers.
        """
        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)
        
        meta = self.meta.copy()
        dtype = self._check_supported_dtype(dtype)

        if nodata is None:
            nodata = get_nodata_value(dtype)

        arr = self.read(masked=True)
        mask_2d = arr.mask.any(axis=0)

        # repeat mask for n_bands
        mask_3d = np.repeat(
            a=mask_2d[np.newaxis, :, :], repeats=self.count, axis=0)

        intersected_arr = np.ma.masked_array(
            arr, mask=mask_3d, fill_value=nodata)
        intersected_arr = np.ma.filled(intersected_arr, fill_value=nodata)

        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["dtype"] = dtype
        meta.update(kwargs)

        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                dst.write(intersected_arr.astype(dtype))
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(**meta)
                dst.write(intersected_arr.astype(dtype))
            file_path = dst

        # create new Raster object with the result
        new_raster = self._new_raster(file_path, self.names)

        # override close method
        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def crop(self, bounds, file_path=None, in_memory=False, driver="GTiff", 
             dtype=None, nodata=None, **kwargs):
        """Crops a Raster object by the supplied bounds.

        Parameters
        ----------
        bounds : tuple
            A tuple containing the bounding box to clip by in the form of
            (xmin, ymin, xmax, ymax).

        file_path : str (optional, default None)
            File path to save to cropped raster. If not supplied then the
            cropped raster is saved to a temporary file.

        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff'). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new intersected Raster is created using the dtype of the
            existing Raster dataset, which uses a dtype that can accommodate
            the data types of all of the individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value
            is set based on the minimum permissible value of the Raster's data
            type. Note that this does not change the pixel nodata values of the
            raster, it only changes the metadata of what value represents a
            nodata pixel.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster cropped to new extent.
        """
        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)
        
        dtype = self._check_supported_dtype(dtype)
        if nodata is None:
            nodata = get_nodata_value(dtype)

        # get row, col positions for bounds
        xmin, ymin, xmax, ymax = bounds
        rows, cols = rasterio.transform.rowcol(
            transform=self.transform, xs=(xmin, xmax), ys=(ymin, ymax)
        )

        # create window covering the min/max rows and cols
        window = Window(
            col_off=min(cols),
            row_off=min(rows),
            width=max(cols) - min(cols),
            height=max(rows) - min(rows),
        )
        cropped_arr = self.read(masked=True, window=window)

        # calculate the new transform
        new_transform = rasterio.transform.from_bounds(
            west=xmin,
            south=ymin,
            east=xmax,
            north=ymax,
            width=cropped_arr.shape[2],
            height=cropped_arr.shape[1],
        )

        # update the destination meta
        meta = self.meta.copy()
        meta.update(
            transform=new_transform,
            width=cropped_arr.shape[2],
            height=cropped_arr.shape[1],
            driver=driver,
            nodata=nodata,
            dtype=dtype,
        )
        meta.update(kwargs)
        cropped_arr = cropped_arr.filled(fill_value=nodata)

        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                dst.write(cropped_arr.astype(dtype))
        
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(**meta)
                dst.write(cropped_arr.astype(dtype))
            file_path = dst

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def to_crs(self, crs, resampling="nearest", file_path=None, in_memory=False,
               driver="GTiff", nodata=None, n_jobs=1, warp_mem_lim=0, 
               progress=False, **kwargs):
        """Reprojects a Raster object to a different crs.

        Parameters
        ----------
        crs : rasterio.transform.CRS object, or dict
            Example: CRS({'init': 'EPSG:4326'})

        resampling : str (default 'nearest')
            Resampling method to use.  One of the following:
            nearest,
            bilinear,
            cubic,
            cubic_spline,
            lanczos,
            average,
            mode,
            max (GDAL >= 2.2),
            min (GDAL >= 2.2),
            med (GDAL >= 2.2),
            q1 (GDAL >= 2.2),
            q3 (GDAL >= 2.2)

        file_path : str (optional, default None)
            Optional path to save reprojected Raster object. If not specified
            then a tempfile is used.
        
        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then the existing
            nodata value of the Raster object is used, which can accommodate
            the dtypes of the individual layers in the Raster.

        n_jobs : int (default 1)
            The number of warp worker threads.

        warp_mem_lim : int (default 0)
            The warp operation memory limit in MB. Larger values allow the warp
            operation to be carried out in fewer chunks. The amount of memory
            required to warp a 3-band uint8 2000 row x 2000 col raster to a
            destination of the same size is approximately 56 MB. The default
            (0) means 64 MB with GDAL 2.2.

        progress : bool (default False)
            Optionally show progress of transform operations.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster following reprojection.
        """
        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)

        if nodata is None:
            nodata = get_nodata_value(self.meta["dtype"])

        resampling_methods = [i.name for i in rasterio.enums.Resampling]
        if resampling not in resampling_methods:
            raise ValueError(
                "Resampling method must be one of {}:".format(
                    resampling_methods))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs=self.crs,
            dst_crs=crs,
            width=self.width,
            height=self.height,
            left=self.bounds.left,
            right=self.bounds.right,
            bottom=self.bounds.bottom,
            top=self.bounds.top,
        )

        meta = self.meta.copy()
        meta["nodata"] = nodata
        meta["width"] = dst_width
        meta["height"] = dst_height
        meta["transform"] = dst_transform
        meta["crs"] = crs
        meta.update(kwargs)

        if progress is True:
            t = tqdm(total=self.count)

        if in_memory is False:
            with rasterio.open(file_path, "w", driver=driver, **meta) as dst:
                for i, layer in enumerate(self.iloc):
                    reproject(
                        source=rasterio.band(layer.ds, layer.bidx),
                        destination=rasterio.band(dst, i + 1),
                        resampling=rasterio.enums.Resampling[resampling],
                        num_threads=n_jobs,
                        warp_mem_lim=warp_mem_lim,
                    )

                    if progress is True:
                        t.update()
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(driver=driver, **meta)
                for i, layer in enumerate(self.iloc):
                    reproject(
                        source=rasterio.band(layer.ds, layer.bidx),
                        destination=rasterio.band(dst, i + 1),
                        resampling=rasterio.enums.Resampling[resampling],
                        num_threads=n_jobs,
                        warp_mem_lim=warp_mem_lim,
                    )

                    if progress is True:
                        t.update()
            file_path = dst

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def aggregate(self, out_shape, resampling="nearest", file_path=None,
                  in_memory=False, driver="GTiff", dtype=None, nodata=None, 
                  **kwargs):
        """Aggregates a raster to (usually) a coarser grid cell size.

        Parameters
        ----------
        out_shape : tuple
            New shape in (rows, cols).

        resampling : str (default 'nearest')
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max',
            'med', 'min', 'mode', 'q1', 'q3'.

        file_path : str (optional, default None)
            File path to save to cropped raster. If not supplied then the
            aggregated raster is saved to a temporary file.

        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new intersected Raster is created using the dtype of the
            existing Raster dataset, which uses a dtype that can accommodate
            the data types of all of the individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value
            is set based on the minimum permissible value of the Raster's
            dtype. Note that this does not change the pixel nodata values of
            the raster, it only changes the metadata of what value represents
            a nodata pixel.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster object aggregated to a new pixel size.
        """
        tfile = None

        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)
        
        rows, cols = out_shape

        arr = self.read(masked=True, out_shape=out_shape,
                        resampling=resampling)
        meta = self.meta.copy()

        dtype = self._check_supported_dtype(dtype)
        
        if nodata is None:
            nodata = get_nodata_value(dtype)

        arr = arr.filled(fill_value=nodata)

        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["height"] = rows
        meta["width"] = cols
        meta["dtype"] = dtype
        bnd = self.bounds
        meta["transform"] = rasterio.transform.from_bounds(
            west=bnd.left,
            south=bnd.bottom,
            east=bnd.right,
            north=bnd.top,
            width=cols,
            height=rows,
        )
        meta.update(kwargs)

        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                dst.write(arr.astype(dtype))
        
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(**meta)
                dst.write(arr.astype(dtype))
            file_path = dst

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def apply(self, function, file_path=None, in_memory=False, driver="GTiff",
              dtype=None, nodata=None, progress=False, n_jobs=1,
              function_args={}, **kwargs):
        """Apply user-supplied function to a Raster object.

        Parameters
        ----------
        function : function
            Function that takes an numpy array as a single argument.

        file_path : str (optional, default None)
            Optional path to save calculated Raster object. If not specified
            then a tempfile is used.

        in_memory : bool, default is False
            Whether to initiated the Raster from an array and store the data
            in-memory using Rasterio's in-memory files.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new Raster is created using the dtype of the calculation
            result.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value
            is set based on the minimum permissible value of the Raster's data
            type. Note that this changes the values of the pixels that
            represent nodata pixels.

        n_jobs : int (default 1)
            Number of processing cores to use for parallel execution. Default
            of -1 is all cores.

        progress : bool (default False)
            Optionally show progress of transform operations.

        function_args : dict (optional)
            Optionally pass arguments to the `function` as a dict or keyword
            arguments.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example
            can be `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            Raster containing the calculated result.
        """
        tfile = None
        
        if in_memory is False:
            file_path, tfile = self._tempfile(file_path)

        n_jobs = get_num_workers(n_jobs)
        function = partial(function, **function_args)

        # perform test calculation determine dimensions, dtype, nodata
        window = next(self.block_shapes(*self.block_shape))
        img = self.read(masked=True, window=window)
        arr = function(img, **function_args)

        if np.ndim(arr) > 2:
            indexes = np.arange(1, arr.shape[0] + 1)
            count = len(indexes)
        else:
            indexes = 1
            count = 1

        dtype = self._check_supported_dtype(dtype)
        
        if nodata is None:
            nodata = get_nodata_value(dtype)

        # open output file with updated metadata
        meta = self.meta.copy()
        meta.update(driver=driver, count=count, dtype=dtype, nodata=nodata)
        meta.update(kwargs)

        # get windows
        windows = [w for w in self.block_shapes(*self.block_shape)]
        data_gen = (self.read(window=w, masked=True) for w in windows)

        if in_memory is False:
            with rasterio.open(file_path, "w", **meta) as dst:
                with Exec(max_workers=n_jobs) as executor:
                    for w, res, pbar in zip(
                            windows,
                            executor.map(function, data_gen),
                            tqdm(windows, disable=not progress)):
                        res = np.ma.filled(res, fill_value=nodata)
                        dst.write(res.astype(dtype), window=w, indexes=indexes)
        else:
            with MemoryFile() as memfile:
                dst = memfile.open(**meta)
                with Exec(max_workers=n_jobs) as executor:
                    for w, res, pbar in zip(
                            windows,
                            executor.map(function, data_gen),
                            tqdm(windows, disable=not progress)):
                        res = np.ma.filled(res, fill_value=nodata)
                        dst.write(res.astype(dtype), window=w, indexes=indexes)
            file_path = dst

        # create new raster object with result
        new_raster = self._new_raster(file_path)

        # override close method
        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def to_pandas(self, max_pixels=None, resampling="nearest"):
        """Raster to pandas DataFrame.

        Parameters
        ----------
        max_pixels: int (default None)
            Maximum number of pixels to sample. By default all pixels are
            used.

        resampling : str (default 'nearest')
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max',
            'med', 'min', 'mode', 'q1', 'q3'.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing values of names of RasterLayers in the Raster
            as columns, and pixel values as rows.
        """

        # read dataset using decimated reads
        n_pixels = self.shape[0] * self.shape[1]

        if max_pixels is not None:
            scaling = max_pixels / n_pixels
        else:
            scaling = 1.0

        out_shape = (
            round(self.shape[0] * scaling), round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape,
                        resampling=resampling)
        bands, rows, cols = arr.shape
        nodatavals = self.nodatavals

        # x and y grid coordinate arrays
        x_range = np.linspace(start=self.bounds.left, stop=self.bounds.right,
                              num=cols)
        y_range = np.linspace(start=self.bounds.top, stop=self.bounds.bottom,
                              num=rows)
        xs, ys = np.meshgrid(x_range, y_range)

        arr = arr.reshape((bands, rows * cols))
        arr = arr.transpose()
        df = pd.DataFrame(
            data=np.column_stack((xs.flatten(), ys.flatten(), arr)),
            columns=["x", "y"] + self.names,
        )

        # set nodata values to nan
        for i, col_name in enumerate(self.names):
            df.loc[df[col_name] == nodatavals[i], col_name] = np.nan

        return df

    def sample(self, size, strata=None, return_array=False, random_state=None):
        """Generates a random sample of according to size, and samples the
        pixel values.

        Parameters
        ----------
        size : int
            Number of random samples or number of samples per strata if
            strategy='stratified'.

        strata : rasterio DatasetReader (opt)
            Whether to use stratified instead of random sampling. Strata can be
            supplied using an open rasterio DatasetReader object.

        return_array : bool (opt), default=False
            Optionally return extracted data as separate X, y and xy masked
            numpy arrays.

        random_state : int (opt)
            integer to use within random.seed.

        Returns
        -------
        tuple
            A tuple containing two elements:

            - numpy.ndarray
                Numpy array of extracted raster values, typically 2d.
            - numpy.ndarray
                2D numpy array of xy coordinates of extracted values.
        """

        # set the seed
        np.random.seed(seed=random_state)

        if not strata:
            # create np array to store randomly sampled data
            valid_samples = np.zeros((0, self.count))
            valid_coordinates = np.zeros((0, 2))

            # loop until target number of samples is satisfied
            satisfied = False

            n = size
            while satisfied is False:

                # generate random row and column indices
                Xsample = np.random.choice(range(0, self.width), n)
                Ysample = np.random.choice(range(0, self.height), n)

                # create 2d numpy array with sample locations set to 1
                sample_raster = np.empty((self.height, self.width))
                sample_raster[:] = np.nan
                sample_raster[Ysample, Xsample] = 1

                # get indices of sample locations
                rows, cols = np.nonzero(np.isnan(sample_raster) == False)

                # convert row, col indices to coordinates
                xy = np.transpose(
                    rasterio.transform.xy(self.transform, rows, cols))

                # sample at random point locations
                samples = self.extract_xy(xy, return_array=True)

                # append only non-masked data to each row of X_random
                samples = samples.astype("float32").filled(np.nan)
                invalid_ind = np.isnan(samples).any(axis=1)
                samples = samples[~invalid_ind, :]
                valid_samples = np.append(valid_samples, samples, axis=0)

                xy = xy[~invalid_ind, :]
                valid_coordinates = np.append(valid_coordinates, xy, axis=0)

                # check to see if target_nsamples has been reached
                if len(valid_samples) >= size:
                    satisfied = True
                else:
                    n = size - len(valid_samples)

        else:
            # get number of unique categories
            strata_arr = strata.read(1)
            categories = np.unique(strata_arr)
            categories = categories[np.nonzero(categories != strata.nodata)]
            categories = categories[~np.isnan(categories)]

            # store selected coordinates
            selected = np.zeros((0, 2))

            for cat in categories:

                # get row,col positions for cat strata
                ind = np.transpose(np.nonzero(strata_arr == cat))

                if size > ind.shape[0]:
                    msg = ("Sample size is greater than number of pixels in "
                           "strata {}").format(str(ind))

                    msg = os.linesep.join([msg, "Sampling using replacement"])
                    Warning(msg)

                # random sample
                sample = np.random.uniform(0, ind.shape[0], size).astype("int")
                xy = ind[sample, :]

                selected = np.append(selected, xy, axis=0)

            # convert row, col indices to coordinates
            x, y = rasterio.transform.xy(transform=self.transform,
                                         rows=selected[:, 0],
                                         cols=selected[:, 1])
            valid_coordinates = np.column_stack((x, y))

            # extract data
            valid_samples = self.extract_xy(valid_coordinates)

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            gdf = pd.DataFrame(valid_samples, columns=self.names)
            gdf["geometry"] = list(
                zip(valid_coordinates[:, 0], valid_coordinates[:, 1])
            )
            gdf["geometry"] = gdf["geometry"].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=self.crs)
            return gdf
        else:
            return valid_samples, valid_coordinates

    def extract_xy(self, xys, return_array=False, progress=False):
        """Samples pixel values using an array of xy locations.

        Parameters
        ----------
        xys : 2d array-like
            x and y coordinates from which to sample the raster
            (n_samples, xys).

        return_array : bool (opt), default=False
            By default the extracted pixel values are returned as a
            geopandas.GeoDataFrame. If `return_array=True` then the extracted
            pixel values are returned as a tuple of numpy.ndarrays.

        progress : bool (opt), default=False
            Show a progress bar for extraction.

        Returns
        -------
        geopandas.GeoDataframe
            Containing extracted data as point geometries if
            `return_array=False`.

        numpy.ndarray
            2d masked array containing sampled raster values (sample, bands) at
            the x,y locations.
        """

        # extract pixel values
        dtype = np.find_common_type([np.float32], self.dtypes)
        X = np.ma.zeros((xys.shape[0], self.count), dtype=dtype)

        for i, (layer, pbar) in enumerate(
                zip(self.iloc,
                    tqdm(self.iloc, total=self.count, disable=not progress))
        ):
            sampler = sample_gen(
                dataset=layer.ds, xy=xys, indexes=layer.bidx, masked=True
            )
            v = np.ma.asarray([i for i in sampler])
            X[:, i] = v.flatten()

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            gdf = pd.DataFrame(X, columns=self.names)
            gdf["geometry"] = list(zip(xys[:, 0], xys[:, 1]))
            gdf["geometry"] = gdf["geometry"].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=self.crs)
            return gdf

        return X

    def extract_vector(self, gdf, return_array=False, progress=False):
        """Sample a Raster/RasterLayer using a geopandas GeoDataframe
        containing points, lines or polygon features.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Containing either point, line or polygon geometries. Overlapping
            geometries will cause the same pixels to be sampled.

        return_array : bool (opt), default=False
            By default the extracted pixel values are returned as a
            geopandas.GeoDataFrame. If `return_array=True` then the extracted
            pixel values are returned as a tuple of numpy.ndarrays.

        progress : bool (opt), default=False
            Show a progress bar for extraction.

        Returns
        -------
        geopandas.GeoDataframe
            Containing extracted data as point geometries (one point per pixel)
            if `return_array=False`. The resulting GeoDataFrame is indexed
            using a named pandas.MultiIndex, with `pixel_idx` index referring
            to the index of each pixel that was sampled, and the `geometry_idx`
            index referring to the index of the each geometry in the supplied
            `gdf`. This makes it possible to keep track of how sampled pixel
            relates to the original geometries, i.e. multiple pixels being
            extracted within the area of a single polygon that can be referred
            to using the `geometry_idx`.

            The extracted data can subsequently be joined with the attribute
            table of the supplied `gdf` using:

            training_py = geopandas.read_file(nc.polygons)
            df = self.stack.extract_vector(gdf=training_py)
            df = df.dropna()

            df = df.merge(
                right=training_py.loc[:, ("id", "label")],
                left_on="polygon_idx",
                right_on="id",
                right_index=True
            )

        tuple
            A tuple (geodataframe index, extracted values, coordinates) of the
            extracted raster values as a masked array and the  coordinates of
            the extracted pixels if `as_gdf=False`.

        """

        # rasterize polygon and line geometries
        if all(gdf.geom_type == "Polygon") or all(
                gdf.geom_type == "LineString"):

            shapes = [(geom, val) for geom, val in
                      zip(gdf.geometry, gdf.index)]
            arr = np.ma.zeros((self.height, self.width))
            arr[:] = -99999

            arr = features.rasterize(
                shapes=shapes,
                fill=-99999,
                out=arr,
                transform=self.transform,
                all_touched=True,
            )

            ids = arr[np.nonzero(arr != -99999)]
            ids = ids.astype("int")
            rows, cols = np.nonzero(arr != -99999)
            xys = rasterio.transform.xy(transform=self.transform, rows=rows,
                                        cols=cols)
            xys = np.transpose(xys)

        elif all(gdf.geom_type == "Point"):
            ids = gdf.index.values
            xys = gdf.bounds.iloc[:, 2:].values

        # extract raster pixels
        dtype = np.find_common_type([np.float32], self.dtypes)
        X = np.ma.zeros((xys.shape[0], self.count), dtype=dtype)

        for i, (layer, pbar) in enumerate(zip(self.iloc,
                                              tqdm(self.iloc, total=self.count,
                                                   disable=not progress))):
            sampler = sample_gen(dataset=layer.ds, xy=xys, indexes=layer.bidx,
                                 masked=True)
            v = np.ma.asarray([i for i in sampler])
            X[:, i] = v.flatten()

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            X = pd.DataFrame(data=X, columns=self.names,
                             index=[pd.RangeIndex(0, X.shape[0]), ids])
            X.index.set_names(["pixel_idx", "geometry_idx"], inplace=True)
            X["geometry"] = list(zip(xys[:, 0], xys[:, 1]))
            X["geometry"] = X["geometry"].apply(Point)
            X = gpd.GeoDataFrame(X, geometry="geometry", crs=self.crs)
            return X

        return ids, X, xys

    def extract_raster(self, src, return_array=False, progress=False):
        """Sample a Raster object by an aligned raster of labelled pixels.

        Parameters
        ----------
        src: rasterio DatasetReader
            Single band raster containing labelled pixels as an open rasterio
            DatasetReader object.

        return_array : bool (opt), default=False
            By default the extracted pixel values are returned as a
            geopandas.GeoDataFrame. If `return_array=True` then the extracted
            pixel values are returned as a tuple of numpy.ndarrays.

        progress : bool (opt), default=False
            Show a progress bar for extraction.

        Returns
        -------
        geopandas.GeoDataFrame
            Geodataframe containing extracted data as point features if
            `return_array=False`

        tuple with three items if `return_array is True
            - numpy.ndarray
                Numpy masked array of extracted raster values, typically 2d.
            - numpy.ndarray
                1d numpy masked array of labelled sampled.
            - numpy.ndarray
                2d numpy masked array of row and column indexes of training
                pixels.
        """

        # open response raster and get labelled pixel indices and values
        arr = src.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)
        xys = np.transpose(rasterio.transform.xy(src.transform, rows, cols))
        ys = arr.data[rows, cols]

        # extract Raster object values at row, col indices
        dtype = np.find_common_type([np.float32], self.dtypes)
        X = np.ma.zeros((xys.shape[0], self.count), dtype=dtype)

        for i, (layer, pbar) in enumerate(zip(self.iloc,
                                              tqdm(self.iloc, total=self.count,
                                                   disable=not progress))):
            sampler = sample_gen(dataset=layer.ds, xy=xys, indexes=layer.bidx,
                                 masked=True)
            v = np.ma.asarray([i for i in sampler])
            X[:, i] = v.flatten()

        # summarize data
        if return_array is False:
            column_names = ["value"] + self.names
            gdf = pd.DataFrame(data=np.ma.column_stack((ys, X)),
                               columns=column_names)
            gdf["geometry"] = list(zip(xys[:, 0], xys[:, 1]))
            gdf["geometry"] = gdf["geometry"].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=self.crs)
            return gdf

        return X, ys, xys
