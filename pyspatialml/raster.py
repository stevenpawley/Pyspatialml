from __future__ import print_function

import concurrent.futures
import math
import tempfile
from collections import Counter, OrderedDict, namedtuple
from collections.abc import Mapping
from copy import deepcopy
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from tqdm import tqdm

from .base import BaseRaster
from .rasterlayer import RasterLayer
from .temporary_files import _file_path_tempfile
from .utils import _get_nodata, _get_num_workers


class _LocIndexer(Mapping):
    """Access pyspatialml.RasterLayer objects by using a key.

    Represents a structure similar to a dict but allows access using a list of keys
    (not just a single key).

    Parameters
    ----------
    parent : pyspatialml.Raster
        The parent Raster object. The _LocIndexer requires the parent Raster object
        so that the raster's attributes are kept up to date with changes to the
        RasterLayers in the dict.
    """

    def __init__(self, parent, *args, **kw):
        self.parent = parent
        self._dict = OrderedDict(*args, **kw)

    def __getitem__(self, keys):
        if isinstance(keys, str):
            selected = self._dict[keys]
        else:
            selected = [self._dict[i] for i in keys]
        return selected

    def __str__(self):
        return str(self._dict)

    def __setitem__(self, key, value):
        self._dict[key] = value
        setattr(self.parent, key, value)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def pop(self, key):
        pop = self._dict.pop(key)
        delattr(self.parent, key)
        return pop

    def rename(self, old, new):
        self._dict = OrderedDict(
            [(new, v) if k == old else (k, v) for k, v in self._dict.items()]
        )
        delattr(self.parent, old)
        setattr(self.parent, new, self._dict[new])


class _iLocIndexer(object):
    """Access pyspatialml.RasterLayer objects using an index position

    A wrapper around _LocIndexer to enable integer-based indexing of the items in the
    OrderedDict. Setting and getting items can occur using a single index position, a
    list or tuple of positions, or a slice of positions.

    Parameters
    ----------
    parent : pyspatialml.Raster
        The parent Raster object. The _LocIndexer requires the parent Raster object
        so that the raster's attributes are kept up to date with changes to the
        RasterLayers in the dict.

    loc_indexer : pyspatialml.raster._LocIndexer
        An instance of a _LocIndexer.
    """

    def __init__(self, parent, loc_indexer):
        self.parent = parent
        self._index = loc_indexer

    def __setitem__(self, index, value):
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            self._index[key] = value
            setattr(self.parent, key, value)

        if isinstance(index, slice):
            index = list(range(index.start, index.stop))

        if isinstance(index, (list, tuple)):
            for i, v in zip(index, value):
                key = list(self._index.keys())[i]
                self._index[key] = v
                setattr(self.parent, key, v)

    def __getitem__(self, index):
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            selected = self._index[key]

        if isinstance(index, slice):
            start = index.start
            stop = index.stop

            if start is None:
                start = 0

            if stop is None:
                stop = self.parent.count

            index = list(range(start, stop))

        if isinstance(index, (list, tuple)):
            key = []
            for i in index:
                key.append(list(self._index.keys())[i])
            selected = [self._index[k] for k in key]
            selected = Raster(selected)

        return selected


class Raster(BaseRaster):
    """Flexible class that represents a collection of file-based GDAL-supported raster
    datasets which share a common coordinate reference system and geometry.

    Raster objects encapsulate RasterLayer objects, which represent single band raster
    datasets that can physically be represented by either separate single-band raster
    files, multi-band raster files, or any combination of individual bands from
    multi-band raster and single-band raster datasets.

    Methods defined in a Raster class comprise those that would typically applied to
    a stack of raster datasets. In addition, these methods always return a new Raster
    object.
    """

    def __init__(
        self,
        src=None,
        arr=None,
        crs=None,
        transform=None,
        nodata=None,
        mode="r",
        file_path=None,
    ):
        """Initiate a new Raster object

        Parameters
        ----------
        src : file path, RasterLayer, or rasterio dataset (opt, default is None)
            Initiate a Raster object from any combination of a file path or list of
            file paths to GDAL-supported raster datasets, RasterLayer objects, or
            directly from a rasterio dataset or band object that is opened in 'r' or
            'rw' mode.

        arr : numpy.ndarray (optional, default is None)
            Whether to initiate a Raster object from a numpy.ndarray. Additional
            arguments `crs` and `transform` should also be provided to supply spatial
            coordinate information. Parameters `arr` and `src` are mutually-exclusive.

        crs : rasterio.crs.CRS object (optional, default is None)
            CRS object containing projection information for data if provided by the
            associated `arr` parameter.

        transform : affine.Affine object (optional, default is None)
            Affine object containing transform information for data if provided by the
            associated `arr` parameter.

        nodata : any number (optional, default is None)
            Assign a nodata value to the Raster dataset when `arr` is used for
            initiation. If a nodata value is not specified then it is determined based
            on the minimum permissible value for the array's data type.

        mode : str (default 'r')
            Mode used to open the raster datasets. Can be one of 'r', 'r+' or 'rw'.

        file_path : str (optional, default None)
            Path to save new Raster object if created from `arr`.
        
        Attributes
        ----------
        loc : _LocIndexer object
            Access pyspatialml.RasterLayer objects within a Raster using a key or a
            list of keys.
        
        iloc : _ILocIndexer object
            Access pyspatialml.RasterLayer objects using an index position. A wrapper
            around _LocIndexer to enable integer-based indexing of the items in the
            OrderedDict. Setting and getting items can occur using a single index
            position, a list or tuple of positions, or a slice of positions.

        files : list
            A list of the raster dataset files that are used in the Raster. This does
            not have to be the same length as the number of RasterLayers because some
            files may have multiple bands.
        
        dtypes : list
            A list of numpy dtypes for each RasterLayer.
        
        nodatavals : list
            A list of the nodata values for each RasterLayer.
        
        count : int
            The number of RasterLayers in the Raster.
        
        res : tuple
            The resolution in (x, y) dimensions of the Raster.
        
        meta : dict
            A dict containing the raster metadata. The dict contains the following
            keys/values:
            crs : the crs object
            transform : the Affine.affine transform object
            width : width of the Raster in pixels
            height : height of the Raster in pixels
            count : number of RasterLayers within the Raster
            dtype : the numpy datatype that represents lowest common denominator of the 
            different dtypes for all of the layers in the Raster.
        
        names : list
            A list of the RasterLayer names.

        block_shape : tuple
            The default block_shape in (rows, cols) for reading windows of data in the
            Raster for out-of-memory processing.
        
        Returns
        -------
        pyspatialml.Raster
            Raster object containing the src layers stacked into a single
            object
        """
        
        # class attributes
        self.loc = _LocIndexer(self)
        self.iloc = _iLocIndexer(self, self.loc)
        self.files = []
        self.dtypes = []
        self.nodatavals = []
        self.count = 0
        self.res = None
        self.meta = None
        self._block_shape = (256, 256)

        # some checks
        if src and arr:
            raise ValueError("Arguments src and arr are mutually exclusive")

        if mode not in ["r", "r+", "w"]:
            raise ValueError("mode must be one of 'r', 'r+', or 'w'")

        # initiate from array
        if arr is not None:

            if file_path is None:
                file_path = tempfile.NamedTemporaryFile().name

            with rasterio.open(
                fp=file_path,
                mode="w",
                driver="GTiff",
                height=arr.shape[1],
                width=arr.shape[2],
                count=arr.shape[0],
                dtype=arr.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata,
            ) as dst:
                dst.write(arr)

            src = [file_path]

        if not isinstance(src, list):
            src = [src]

        src_layers = []

        # initiated from file paths
        if all(isinstance(x, str) for x in src):
            for f in src:
                r = rasterio.open(f, mode=mode)

                for i in range(r.count):
                    band = rasterio.band(r, i + 1)
                    src_layers.append(RasterLayer(band))

        # initiate from RasterLayer objects
        elif all(isinstance(x, RasterLayer) for x in src):
            src_layers = src

        # initiate from rasterio.io.datasetreader
        elif all(isinstance(x, rasterio.io.DatasetReader) for x in src):
            for r in src:
                for i in range(r.count):
                    band = rasterio.band(r, i + 1)
                    src_layers.append(RasterLayer(band))

        # initiate from rasterio.band objects
        elif all(isinstance(x, rasterio.Band) for x in src):
            for band in src:
                src_layers.append(RasterLayer(band))

        # otherwise raise error
        elif all(isinstance(x, type(x[0])) for x in src):
            raise ValueError(
                "Cannot initiated a Raster from a list of different type " "objects"
            )

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
        pyspatialml.RasterLayer or pyspatialml.Raster
            Returns a Raster with the subset selection of layers. If only one layer is
            subset then a RasterLayer object is returned.
        """

        # return a RasterLayer if a single layer are subset
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
        """Replace a RasterLayer within the Raster object with a new RasterLayer.
        
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
        """Iterate over RasterLayers.
        """
        return iter(self.loc.items())

    def close(self):
        """Close all of the RasterLayer objects in the Raster.

        Note that this will cause any rasters based on temporary files to be removed.
        This is intended as a method of clearing temporary files that may have
        accumulated during an analysis session.
        """
        for layer in self.iloc:
            layer.close()

    @staticmethod
    def _check_alignment(layers):
        """Check that a list of raster datasets are aligned with the same pixel
        dimensions and geotransforms.

        Parameters
        ----------
        layers : list
            List of pyspatialml.RasterLayer objects.

        Returns
        -------
        dict or False
            Dict of metadata if all layers are spatially aligned, otherwise
            returns False.
        """

        src_meta = []
        for layer in layers:
            src_meta.append(layer.ds.meta.copy())

        if not all(i["crs"] == src_meta[0]["crs"] for i in src_meta):
            Warning(
                "crs of all rasters does not match, " "possible unintended consequences"
            )

        if not all(
            [
                i["height"] == src_meta[0]["height"]
                or i["width"] == src_meta[0]["width"]
                or i["transform"] == src_meta[0]["transform"]
                for i in src_meta
            ]
        ):
            return False

        else:
            return src_meta[0]

    @staticmethod
    def _fix_names(combined_names):
        """Adjusts the names of pyspatialml.RasterLayer objects within the Raster when
        appending new layers.
        
        This avoids the Raster object containing duplicated names in the case that
        multiple RasterLayer's are appended with the same name.

        In the case of duplicated names, the RasterLayer names are appended
        with a `_n` with n = 1, 2, 3 .. n.

        Parameters
        ----------
        combined_names : list
            List of str representing names of RasterLayers. Any duplicates with
            have a suffix appended to them.

        Returns
        -------
        list
            List with adjusted names
        """

        counts = Counter(combined_names)

        for s, num in counts.items():
            if num > 1:
                for suffix in range(1, num + 1):
                    if s + "_" + str(suffix) not in combined_names:
                        combined_names[combined_names.index(s)] = s + "_" + str(suffix)
                    else:
                        i = 1
                        while s + "_" + str(i) in combined_names:
                            i += 1
                        combined_names[combined_names.index(s)] = s + "_" + str(i)

        return combined_names

    @property
    def block_shape(self):
        """Return the windows size used for raster calculations, specified as a tuple
        (rows, columns).

        Returns
        -------
        tuple
            Block window shape that is currently set for the Raster as a tuple in the
            format of (n_rows, n_columns) in pixels.
        """
        return self._block_shape

    @block_shape.setter
    def block_shape(self, value):
        """Set the windows size used for raster calculations, specified as a tuple
        (rows, columns).

        Parameters
        ----------
        value : tuple
            Tuple of integers for default block shape to read and write data from the
            Raster object for memory-safe calculations. Specified as (n_rows,n_columns).
        """
        if not isinstance(value, tuple):
            raise ValueError(
                "block_shape must be set using an integer tuple " "as (rows, cols)"
            )

        rows, cols = value

        if not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError(
                "tuple must consist of integer values referring "
                "to number of rows, cols"
            )

        self._block_shape = (rows, cols)

    @property
    def names(self):
        """Return the names of the RasterLayers in the Raster object

        Returns
        -------
        list
            List of names of RasterLayer objects
        """
        return list(self.loc.keys())

    @property
    def _layers(self):
        """Getter method

        Returns
        -------
        pyspatialml.indexing._LocIndexer
            Returns a dict of key-value pairs of names and RasterLayers.
        """
        return self.loc

    @_layers.setter
    def _layers(self, layers):
        """Setter method for the files attribute in the Raster object

        Parameters
        ----------
        layers : RasterLayer or list of RasterLayers
            RasterLayers used to initiate a Raster object.
        """

        # some checks
        if isinstance(layers, RasterLayer):
            layers = [layers]

        if all(isinstance(x, type(layers[0])) for x in layers) is False:
            raise ValueError(
                "Cannot create a Raster object from a mixture of input types"
            )

        meta = self._check_alignment(layers)

        if meta is False:
            raise ValueError(
                "Raster datasets do not all have the same dimensions or " "transform"
            )

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

        bounds = rasterio.transform.array_bounds(
            self.height, self.width, self.transform
        )
        BoundingBox = namedtuple("BoundingBox", ["left", "bottom", "right", "top"])
        self.bounds = BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])

        names = [i.names[0] for i in layers]
        names = self._fix_names(names)

        # update attributes per dataset
        for layer, name in zip(layers, names):
            self.dtypes.append(layer.dtype)
            self.nodatavals.append(layer.nodata)
            self.files.append(layer.file)
            layer.names = [name]
            self.loc[name] = layer
            setattr(self, name, self.loc[name])

        self.meta = dict(
            crs=self.crs,
            transform=self.transform,
            width=self.width,
            height=self.height,
            count=self.count,
            dtype=np.find_common_type(self.dtypes, []),
        )
    
    def _check_supported_dtype(self, dtype):
        if dtype is None:
            dtype = self.meta["dtype"]
        else:
            if rasterio.dtypes.check_dtype(dtype) is False:
                raise AttributeError(
                    "{dtype} is not a support GDAL dtype".format(dtype=dtype)
                )
        return dtype

    def read(
        self,
        masked=False,
        window=None,
        out_shape=None,
        resampling="nearest",
        as_df=False,
        **kwargs
    ):
        """Reads data from the Raster object into a numpy array.

        Overrides read BaseRaster class read method and replaces it with a method that
        reads from multiple RasterLayer objects.

        Parameters
        ----------
        masked : bool (default False)
            Read data into a masked array.

        window : rasterio.window.Window object (optional, default None)
            Tuple of col_off, row_off, width, height of a window of data to read a chunk
            of data into a ndarray.

        out_shape : tuple (optional, default None)
            Shape of shape of array (rows, cols) to read data into using decimated reads.

        resampling : str (default 'nearest')
            Resampling method to use when applying decimated reads when out_shape is
            specified. Supported methods are: 'average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'q1', 'q3'.
        
        as_df : bool (default False)
            Whether to return the data as a pandas.DataFrame with columns named by the
            RasterLayer names. Can be useful when using scikit-learn ColumnTransformer
            to select columns based on names rather than keeping track of indexes.

        **kwargs : dict
            Other arguments to pass to rasterio.DatasetReader.read method

        Returns
        -------
        ndarray
            Raster values in 3d ndarray  with the dimensions in order of (band, row,
            and column).
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
            arr[i, :, :] = layer.read(
                masked=masked,
                window=window,
                out_shape=out_shape,
                resampling=resampling,
                **kwargs
            )

            if masked is True:
                arr[i, :, :] = np.ma.MaskedArray(
                    data=arr[i, :, :], mask=np.isfinite(arr[i, :, :]).mask
                )

        if as_df is True:
            arr = arr.transpose(1, 2, 0) # rehape to rows, cols, bands
            arr_flat = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
            df = pd.DataFrame(data=arr_flat, columns=self.names)

            return df

        return arr

    def write(self, file_path, driver="GTiff", dtype=None, nodata=None, **kwargs):
        """Write the Raster object to a file.

        Overrides the write RasterBase class method, which is a partial function of the
        rasterio.DatasetReader.write method.

        Parameters
        ----------
        file_path : str
            File path used to save the Raster object.

        driver : str (default is 'GTiff'). 
            Name of GDAL driver used to save Raster data.

        dtype : str (opt, default None)
            Optionally specify a numpy compatible data type when saving to file. If not
            specified, a data type is selected based on the data types of RasterLayers
            in the Raster object.

        nodata : any number (opt, default None)
            Optionally assign a new nodata value when saving to file. If not specified
            a nodata value based on the minimum permissible value for the data types of
            RasterLayers in the Raster object is used. Note that this does not change
            the pixel nodata values of the raster, it only changes the metadata of what
            value represents a nodata pixel.
        
        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        Raster
            New Raster object from saved file.
        """
        dtype = self._check_supported_dtype(dtype)
        if nodata is None:
            nodata = _get_nodata(dtype)

        meta = self.meta
        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["dtype"] = dtype
        meta.update(kwargs)

        with rasterio.open(file_path, mode="w", **meta) as dst:

            for i, layer in enumerate(self.iloc):
                arr = layer.read()
                arr[arr == layer.nodata] = nodata
                dst.write(arr.astype(dtype), i + 1)

        raster = self._new_raster(file_path, self.names)

        return raster

    def predict_proba(
        self,
        estimator,
        file_path=None,
        indexes=None,
        driver="GTiff",
        dtype=None,
        nodata=None,
        as_df=False,
        progress=False,
        **kwargs,
    ):
        """Apply class probability prediction of a scikit learn model to a Raster.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str (optional, default None)
            Path to a GeoTiff raster for the prediction results. If not specified then
            the output is written to a temporary file.

        indexes : list of integers (optional, default None)
            List of class indices to export. In some circumstances, only a subset of
            the class probability estimations are desired, for instance when performing
            a binary classification only the probabilities for the positive class may
            be desired.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Optionally specify a GDAL compatible data type when saving to file. If not
            specified, a data type is set based on the data type of the prediction.

        nodata : any number (optional, default None)
            Nodata value for file export. If not specified then the nodata value is
            derived from the minimum permissible value for the given data type.
    
        as_df : bool (default is False)
            Whether to read the raster data via pandas before prediction. This can be
            useful if transformers are being used as part of a pipeline and you want
            to refer to column names rather than indices.

        progress : bool (default False)
            Show progress bar for prediction.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.Raster
            Raster containing predicted class probabilities. Each predicted class is
            represented by a RasterLayer object. The RasterLayers are named `prob_n`
            for 1,2,3..n, with `n` based on the index position of the classes, not the
            number of the class itself.

            For example, a classification model predicting classes with integer values
            of 1, 3, and 5 would result in three RasterLayers named prob_1, prob_2 and
            prob_3.
        """
        file_path, tfile = _file_path_tempfile(file_path)
        predfun = self._probfun

        # determine output count
        if isinstance(indexes, int):
            indexes = range(indexes, indexes + 1)

        elif indexes is None:
            indexes = np.arange(0, estimator.n_classes_)

        if dtype is None:
            dtype = np.float32
        
        if rasterio.dtypes.check_dtype(dtype) is False:
            raise AttributeError(
                "{dtype} is not a support GDAL dtype".format(dtype=dtype)
            )

        if nodata is None:
            nodata = _get_nodata(dtype)
            
        if progress is True:
            disable_tqdm = False
        else:
            disable_tqdm = True

        # open output file with updated metadata
        meta = deepcopy(self.meta)
        meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)
        meta.update(kwargs)

        with rasterio.open(file_path, "w", **meta) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            data_gen = ((window, self.read(window=window, masked=True, as_df=as_df)) for window in windows)

            for window, arr, pbar in zip(windows, data_gen, tqdm(windows, disable=disable_tqdm)):
                result = predfun(arr, estimator)
                result = np.ma.filled(result, fill_value=nodata)
                dst.write(result[indexes, :, :].astype(dtype), window=window)

        # generate layer names
        prefix = "prob_"
        names = [prefix + str(i) for i in range(len(indexes))]

        # create new raster object
        new_raster = self._new_raster(file_path, names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def predict(
        self,
        estimator,
        file_path=None,
        driver="GTiff",
        dtype=None,
        nodata=None,
        as_df=False,
        n_jobs=-1,
        progress=False,
        **kwargs,
    ):
        """Apply prediction of a scikit learn model to a Raster.
        
        The model can represent any scikit learn model or compatible api with a `fit`
        and `predict` method. These can consist of classification or regression models.
        Multi-class classifications and multi-target regressions are also supported.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str (optional, default None)
            Path to a GeoTiff raster for the prediction results. If not specified then
            the output is written to a temporary file.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export

        dtype : str (optional, default None)
            Optionally specify a GDAL compatible data type when saving to file. If not
            specified, np.float32 is assumed.

        nodata : any number (optional, default None)
            Nodata value for file export. If not specified then the nodata value is
            derived from the minimum permissible value for the given data type.
        
        as_df : bool (default is False)
            Whether to read the raster data via pandas before prediction. This can be
            useful if transformers are being used as part of a pipeline and you want
            to refer to column names rather than indices.
        
        n_jobs : int (default -1)
            Number of processing cores to use for parallel execution. Default is
            n_jobs=1. -1 is all cores; -2 is all cores -1. 

        progress : bool (default False)
            Show progress bar for prediction.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatial.Raster
            Raster object containing prediction results as a RasterLayers. For
            classification and regression models, the Raster will contain a single
            RasterLayer, unless the model is multi-class or multi-target. Layers
            are named automatically as `pred_raw_n` with n = 1, 2, 3 ..n.
        """
        file_path, tfile = _file_path_tempfile(file_path)
        n_jobs = _get_num_workers(n_jobs)

        # determine output count for multi output cases
        indexes = np.arange(0, estimator.n_outputs_)

        # chose prediction function
        if len(indexes) == 1:
            predfun = partial(self._predfun, estimator=estimator)
        else:
            predfun = partial(self._predfun_multioutput, estimator=estimator)

        if dtype is None:
            dtype = np.float32

        if rasterio.dtypes.check_dtype(dtype) is False:
            raise AttributeError(
                "{dtype} is not a support GDAL dtype".format(dtype=dtype)
            )
                
        if nodata is None:
            nodata = _get_nodata(dtype)
        
        if progress is True:
            disable_tqdm = False
        else:
            disable_tqdm = True

        # open output file with updated metadata
        meta = deepcopy(self.meta)
        meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)
        meta.update(kwargs)

        with rasterio.open(file_path, "w", **meta) as dst:
            windows = [window for window in self.block_shapes(*self._block_shape)]

            # generator gets raster arrays for each window
            data_gen = ((window, self.read(window=window, masked=True, as_df=as_df)) for window in windows)

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                for window, result, pbar in zip(windows, executor.map(predfun, data_gen), tqdm(windows, disable=disable_tqdm)):
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(dtype), window=window)
        
        # generate layer names
        prefix = "pred_raw_"
        names = [prefix + str(i) for i in range(len(indexes))]

        # create new raster object
        new_raster = self._new_raster(file_path, names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def _predfun(self, img, estimator):
        """Prediction function for classification or regression response.

        Parameters
        ----
        img : tuple (window, numpy.ndarray)
            A window object, and a 3d ndarray of raster data with the dimensions in
            order of (band, rows, columns).

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        Returns
        -------
        numpy.ndarray
            2d numpy array representing a single band raster containing the
            classification or regression result.
        """
        window, img = img

        if not isinstance(img, pd.DataFrame):
            # reshape each image block matrix into a 2D matrix
            # first reorder into rows, cols, bands(transpose)
            # then resample into 2D array (rows=sample_n, cols=band_values)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

            # create mask for NaN values and replace with number
            flat_pixels_mask = flat_pixels.mask.copy()
            flat_pixels = flat_pixels.filled(0)
        
        else:
            flat_pixels = img
            flat_pixels_mask = pd.isna(flat_pixels).values
            flat_pixels = flat_pixels.fillna(0)
            flat_pixels = flat_pixels.values

        # predict and replace mask
        result_cla = estimator.predict(flat_pixels)
        result_cla = np.ma.masked_array(
            data=result_cla, mask=flat_pixels_mask.any(axis=1)
        )

        # reshape the prediction from a 1D into 3D array [band, row, col]
        result_cla = result_cla.reshape((1, window.height, window.width))

        return result_cla

    @staticmethod
    def _probfun(img, estimator):
        """Class probabilities function.

        Parameters
        ----------
        img : tuple (window, numpy.ndarray)
            A window object, and a 3d ndarray of raster data with the dimensions in
            order of (band, rows, columns).

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        Returns
        -------
        numpy.ndarray
            Multi band raster as a 3d numpy array containing the probabilities
            associated with each class. ndarray dimensions are in the order of
            (class, row, column).
        """
        window, img = img

        if not isinstance(img, pd.DataFrame):
            # reshape each image block matrix into a 2D matrix
            # first reorder into rows, cols, bands (transpose)
            # then resample into 2D array (rows=sample_n, cols=band_values)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            mask2d = img.mask.any(axis=0)
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
            flat_pixels = flat_pixels.filled(0)
        
        else:
            flat_pixels = img
            mask2d = pd.isna(flat_pixels).values
            mask2d = mask2d.reshape((window.height, window.width, flat_pixels.shape[1]))
            mask2d = mask2d.any(axis=2)
            flat_pixels = flat_pixels.fillna(0)
            flat_pixels = flat_pixels.values

        # predict probabilities
        result_proba = estimator.predict_proba(flat_pixels)

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result_proba = result_proba.reshape((window.height, window.width, result_proba.shape[1]))

        # reshape band into rasterio format [band, row, col]
        result_proba = result_proba.transpose(2, 0, 1)

        # repeat mask for n_bands
        mask3d = np.repeat(
            a=mask2d[np.newaxis, :, :], repeats=result_proba.shape[0], axis=0
        )

        # convert proba to masked array
        result_proba = np.ma.masked_array(result_proba, mask=mask3d, fill_value=np.nan)

        return result_proba

    @staticmethod
    def _predfun_multioutput(img, estimator):
        """Multi-target prediction function.

        Parameters
        ----------
        img : tuple (window, numpy.ndarray)
            A window object, and a 3d ndarray of raster data with the dimensions in
            order of (band, rows, columns).

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        Returns
        -------
        numpy.ndarray
            3d numpy array representing the multi-target prediction result with the
            dimensions in the order of (target, row, column).
        """
        window, img = img

        if not isinstance(img, pd.DataFrame):
            # reshape each image block matrix into a 2D matrix
            # first reorder into rows, cols, bands(transpose)
            # then resample into 2D array (rows=sample_n, cols=band_values)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            mask2d = img.mask.any(axis=0)
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
            flat_pixels = flat_pixels.filled(0)
        
        else:
            flat_pixels = img
            mask2d = pd.isna(flat_pixels).values
            mask2d = mask2d.reshape((window.height, window.width, flat_pixels.shape[1]))
            mask2d = mask2d.any(axis=2)
            flat_pixels = flat_pixels.fillna(0)
            flat_pixels = flat_pixels.values

        # predict probabilities
        result = estimator.predict(flat_pixels)

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result = result.reshape((window.height, window.width, result.shape[1]))

        # reshape band into rasterio format [band, row, col]
        result = result.transpose(2, 0, 1)

        # repeat mask for n_bands
        mask3d = np.repeat(a=mask2d[np.newaxis, :, :], repeats=result.shape[0], axis=0)

        # convert proba to masked array
        result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

        return result

    def append(self, other, in_place=True):
        """Method to add new RasterLayers to a Raster object.
        
        Note that this modifies the Raster object in-place by default.

        Parameters
        ----------
        other : Raster object, or list of Raster objects
            Object to append to the Raster.
        
        in_place : bool (default True)
            Whether to change the Raster object in-place or leave original and return
            a new Raster object.

        Returns
        -------
        pyspatialml.Raster
            Returned only if `in_place` is True
        """

        if isinstance(other, Raster):
            other = [other]

        for new_raster in other:

            if not isinstance(new_raster, Raster):
                raise AttributeError(new_raster + " is not a pyspatialml.Raster object")

            # check that other raster does not result in duplicated names
            combined_names = self.names + new_raster.names
            combined_names = self._fix_names(combined_names)

            # update layers and names
            combined_layers = list(self.loc.values()) + list(new_raster.loc.values())

            for layer, name in zip(combined_layers, combined_names):
                layer.names = [name]

            if in_place is True:
                self._layers = combined_layers
            else:
                new_raster = self._new_raster(self.files, self.names)
                new_raster._layers = combined_layers

                return new_raster

    def drop(self, labels, in_place=True):
        """Drop individual RasterLayers from a Raster object
        
        Note that this modifies the Raster object in-place by default.
        
        Parameters
        ---------
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or label, or a
            list of integers or labels.
        
        in_place : bool (default True)
            Whether to change the Raster object in-place or leave original and return
            a new Raster object.

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

            subset_layers = [
                v for (i, v) in enumerate(list(self.loc.values())) if i not in labels
            ]

        # str label based subsetting
        elif len([i for i in labels if isinstance(i, str)]) == len(labels):

            subset_layers = [
                v
                for (i, v) in enumerate(list(self.loc.values()))
                if self.names[i] not in labels
            ]

        else:
            raise ValueError(
                "Cannot drop layers based on mixture of indexes and labels"
            )

        if in_place is True:
            self._layers = subset_layers
        else:
            new_raster = self._new_raster(self.files, self.names)
            new_raster._layers = subset_layers

            return new_raster

    def rename(self, names, in_place=True):
        """Rename a RasterLayer within the Raster object.
        
        Note that by default this modifies the Raster object in-place.

        Parameters
        ----------
        names : dict
            dict of old_name : new_name
        
        in_place : bool (default True)
            Whether to change names of the Raster object in-place or leave original
            and return a new Raster object.

        Returns
        -------
        pyspatialml.Raster
            Returned only if `in_place` is True
        """

        if in_place is True:
            for old_name, new_name in names.items():
                # change internal name of RasterLayer
                self.loc[old_name].names = [new_name]

                # change name of layer in stack
                self.loc.rename(old_name, new_name)
        else:
            new_raster = self._new_raster(self.files, self.names)
            for old_name, new_name in names.items():
                # change internal name of RasterLayer
                new_raster.loc[old_name].names = [new_name]

                # change name of layer in stack
                new_raster.loc.rename(old_name, new_name)

            return new_raster

    def plot(
        self,
        cmap=None,
        norm=None,
        figsize=None,
        out_shape=(100, 100),
        title_fontsize=8,
        label_fontsize=6,
        legend_fontsize=6,
        names=None,
        fig_kwds=None,
        legend_kwds=None,
        subplots_kwds=None,
    ):
        """Plot a Raster object as a raster matrix

        Parameters
        ----------
        cmap : str (opt), default=None
            Specify a single cmap to apply to all of the RasterLayers.
            This overides the cmap attribute of each RasterLayer.
        
        norm :  matplotlib.colors.Normalize (opt), default=None
            A matplotlib.colors.Normalize to apply to all of the RasterLayers.
            This overides the norm attribute of each RasterLayer.
            
        figsize : tuple (opt), default=None
            Size of the resulting matplotlib.figure.Figure.

        out_shape : tuple, default=(100, 100)
            Number of rows, cols to read from the raster datasets for plotting.

        title_fontsize : any number, default=8
            Size in pts of titles.

        label_fontsize : any number, default=6
            Size in pts of axis ticklabels.

        legend_fontsize : any number, default=6
            Size in pts of legend ticklabels.

        names : list (opt), default=None
            Optionally supply a list of names for each RasterLayer to override the
            default layer names for the titles.

        fig_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.figure call when
            creating the figure object.

        legend_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.colorbar call when
            creating the colorbar object.

        subplots_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.subplots_adjust
            function. These are used to control the spacing and position of each
            subplot, and can include
            {left=None, bottom=None, right=None, top=None, wspace=None, hspace=None}.

        Returns
        -------
        axs : numpy.ndarray
            array of matplotlib.axes._subplots.AxesSubplot or a single
            matplotlib.axes._subplots.AxesSubplot if Raster object contains only a
            single layer.
        """

        # some checks
        if norm:
            if not isinstance(norm, mpl.colors.Normalize):
                raise AttributeError(
                    "norm argument should be a matplotlib.colors.Normalize object"
                )

        if cmap:
            cmaps = [cmap for i in self.iloc]
        else:
            cmaps = [i.cmap for i in self.iloc]
        
        if norm:
            norms = [norm for i in self.iloc]
        else:
            norms = [i.norm for i in self.iloc]

        if names is None:
            names = self.names
        else:
            if len(names) != self.count:
                raise AttributeError(
                    "arguments 'names' needs to be the same length as the number of RasterLayer objects"
                )

        if fig_kwds is None:
            fig_kwds = {}

        if legend_kwds is None:
            legend_kwds = {}

        if subplots_kwds is None:
            subplots_kwds = {}
        
        if figsize:
            fig_kwds["figsize"] = figsize

        # plot a single layer
        if self.count == 1:
            return self.iloc[0].plot(
                cmap=cmap, 
                norm=norm, 
                figsize=figsize, 
                fig_kwds=fig_kwds,
                legend_kwds=legend_kwds,
                legend=True
            )
        
        # estimate required number of rows and columns in figure
        rows = int(np.sqrt(self.count))
        cols = int(math.ceil(np.sqrt(self.count)))

        if rows * cols < self.count:
            rows += 1

        fig, axs = plt.subplots(rows, cols, **fig_kwds)

        # axs.flat is an iterator over the row-order flattened axs array
        for ax, n, cmap, norm, name in zip(
            axs.flat, range(self.count), cmaps, norms, names
        ):

            arr = self.iloc[n].read(masked=True, out_shape=out_shape)

            ax.set_title(name, fontsize=title_fontsize, y=1.00)

            im = ax.imshow(
                arr,
                extent=[
                    self.bounds.left,
                    self.bounds.right,
                    self.bounds.bottom,
                    self.bounds.top,
                ],
                cmap=cmap,
                norm=norm,
            )

            divider = make_axes_locatable(ax)

            if "orientation" not in legend_kwds.keys():
                legend_kwds["orientation"] = "vertical"

            if legend_kwds["orientation"] == "vertical":
                legend_pos = "right"

            elif legend_kwds["orientation"] == "horizontal":
                legend_pos = "bottom"

            cax = divider.append_axes(legend_pos, size="10%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax, **legend_kwds)
            cbar.ax.tick_params(labelsize=legend_fontsize)

            # hide tick labels by default when multiple rows or cols
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

            # show y-axis tick labels on first subplot
            if n == 0 and rows > 1:
                ax.set_yticklabels(
                    ax.yaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                )
            if n == 0 and rows == 1:
                ax.set_xticklabels(
                    ax.xaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                )
                ax.set_yticklabels(
                    ax.yaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                )
            if rows > 1 and n == (rows * cols) - cols:
                ax.set_xticklabels(
                    ax.xaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                )

        for ax in axs.flat[axs.size - 1 : self.count - 1 : -1]:
            ax.set_visible(False)

        plt.subplots_adjust(**subplots_kwds)

        return axs

    def _new_raster(self, file_path, names=None):
        """Return a new Raster object

        Parameters
        ----------
        file_path : str
            Path to files to create the new Raster object from.

        names : list (optional, default None)
            List to name the RasterLayer objects in the stack. If not supplied then the
            names will be generated from the file names.

        Returns
        -------
        pyspatialml.Raster
        """

        # some checks
        if isinstance(file_path, str):
            file_path = [file_path]

        # create new raster from supplied file path
        raster = Raster(file_path)

        # rename and set cmaps
        if names is not None:
            rename = {old: new for old, new in zip(raster.names, names)}
            raster.rename(rename)

        for old_layer, new_layer in zip(self.iloc, raster.iloc):
            new_layer.cmap = old_layer.cmap

        return raster

    def mask(
        self,
        shapes,
        invert=False,
        crop=True,
        pad=False,
        file_path=None,
        driver="GTiff",
        dtype=None,
        nodata=None,
        **kwargs,
    ):
        """Mask a Raster object based on the outline of shapes in a
        geopandas.GeoDataFrame

        Parameters
        ----------
        shapes : geopandas.GeoDataFrame
            GeoDataFrame containing masking features.

        invert : bool (default False)
            If False then pixels outside shapes will be masked. If True then pixels
            inside shape will be masked.

        crop : bool (default True)
            Crop the raster to the extent of the shapes.

        pad : bool (default False)
            If True, the features will be padded in each direction by one half of a
            pixel prior to cropping raster.

        file_path : str (optional, default None)
            File path to save to resulting Raster. If not supplied then the resulting
            Raster is saved to a temporary file

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then the
            cropped Raster is created using the existing dtype, which uses a dtype that
            can accommodate the data types of all of the individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for cropped dataset. If not specified then a nodata value is
            set based on the minimum permissible value of the Raster's data type. Note
            that this changes the values of the pixels to the new nodata value, and changes
            the metadata of the raster.
        
        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.
        """

        # some checks
        if invert is True:
            crop = False

        file_path, tfile = _file_path_tempfile(file_path)
        meta = deepcopy(self.meta)

        dtype = self._check_supported_dtype(dtype)
        if nodata is None:
            nodata = _get_nodata(dtype)

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

        with rasterio.open(file_path, "w", **meta) as dst:
            dst.write(masked_ndarrays.astype(dtype))

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def intersect(self, file_path=None, driver="GTiff", dtype=None, nodata=None, **kwargs):
        """Perform a intersect operation on the Raster object.

        Computes the geometric intersection of the RasterLayers with the Raster object.
        This will cause nodata values in any of the rasters to be propagated through
        all of the output rasters.

        Parameters
        ----------
        file_path : str (optional, default None)
            File path to save to resulting Raster. If not supplied then the resulting
            Raster is saved to a temporary file.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then the new
            intersected Raster is created using the dtype of the existing Raster dataset,
            which uses a dtype that can accommodate the data types of all of the
            individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type. Note that
            this changes the values of the pixels that represent nodata to the new
            value.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.Raster
            Raster with layers that are masked based on a union of all masks in the
            suite of RasterLayers.
        """
        file_path, tfile = _file_path_tempfile(file_path)
        meta = deepcopy(self.meta)

        dtype = self._check_supported_dtype(dtype)

        if nodata is None:
            nodata = _get_nodata(dtype)

        arr = self.read(masked=True)
        mask_2d = arr.mask.any(axis=0)

        # repeat mask for n_bands
        mask_3d = np.repeat(a=mask_2d[np.newaxis, :, :], repeats=self.count, axis=0)

        intersected_arr = np.ma.masked_array(arr, mask=mask_3d, fill_value=nodata)

        intersected_arr = np.ma.filled(intersected_arr, fill_value=nodata)

        meta["driver"] = driver
        meta["nodata"] = nodata
        meta["dtype"] = dtype
        meta.update(kwargs)

        with rasterio.open(file_path, "w", **meta) as dst:
            dst.write(intersected_arr.astype(dtype))

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def crop(self, bounds, file_path=None, driver="GTiff", dtype=None, nodata=None, **kwargs):
        """Crops a Raster object by the supplied bounds.

        Parameters
        ----------
        bounds : tuple
            A tuple containing the bounding box to clip by in the form of
            (xmin, ymin, xmax, ymax).

        file_path : str (optional, default None)
            File path to save to cropped raster. If not supplied then the cropped raster
            is saved to a temporary file.

        driver : str (default 'GTiff'). Default is 'GTiff'
            Named of GDAL-supported driver for file export.
        
        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then the new
            intersected Raster is created using the dtype of the existing Raster
            dataset, which uses a dtype that can accommodate the data types of all of
            the individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type. Note that
            this does not change the pixel nodata values of the raster, it only changes
            the metadata of what value represents a nodata pixel.
        
        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.Raster
            Raster cropped to new extent.
        """

        file_path, tfile = _file_path_tempfile(file_path)
        dtype = self._check_supported_dtype(dtype)
        if nodata is None:
            nodata = _get_nodata(dtype)

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
            height=cropped_arr.shape[1]
        )

        # update the destination meta
        meta = self.meta.copy()
        meta.update(
            transform=new_transform,
            width=cropped_arr.shape[2], 
            height=cropped_arr.shape[1],
            driver=driver,
            nodata=nodata,
            dtype=dtype
        )
        meta.update(kwargs)

        cropped_arr = cropped_arr.filled(fill_value=nodata)

        with rasterio.open(file_path, "w", **meta) as dst:
            dst.write(cropped_arr.astype(dtype))

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def to_crs(
        self,
        crs,
        resampling="nearest",
        file_path=None,
        driver="GTiff",
        nodata=None,
        n_jobs=1,
        warp_mem_lim=0,
        progress=False,
        **kwargs,
    ):
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
            Optional path to save reprojected Raster object. If not specified then a
            tempfile is used.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then the existing nodata
            value of the Raster object is used, which can accommodate the dtypes of the
            individual layers in the Raster.

        n_jobs : int (default 1)
            The number of warp worker threads.

        warp_mem_lim : int (default 0)
            The warp operation memory limit in MB. Larger values allow the warp
            operation to be carried out in fewer chunks. The amount of memory required
            to warp a 3-band uint8 2000 row x 2000 col raster to a destination of the
            same size is approximately 56 MB. The default (0) means 64 MB with GDAL 2.2.

        progress : bool (default False)
            Optionally show progress of transform operations.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.Raster
            Raster following reprojection.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        if nodata is None:
            nodata = _get_nodata(self.meta["dtype"])

        resampling_methods = [i.name for i in rasterio.enums.Resampling]
        if resampling not in resampling_methods:
            raise ValueError(
                "Invalid resampling method."
                + "Resampling method must be one of {0}:".format(resampling_methods)
            )

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

        meta = deepcopy(self.meta)
        meta["nodata"] = nodata
        meta["width"] = dst_width
        meta["height"] = dst_height
        meta["transform"] = dst_transform
        meta["crs"] = crs
        meta.update(kwargs)

        if progress is True:
            t = tqdm(total=self.count)

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

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def aggregate(
        self,
        out_shape,
        resampling="nearest",
        file_path=None,
        driver="GTiff",
        dtype=None,
        nodata=None,
        **kwargs,
    ):
        """Aggregates a raster to (usually) a coarser grid cell size.

        Parameters
        ----------
        out_shape : tuple
            New shape in (rows, cols).

        resampling : str (default 'nearest')
            Resampling method to use when applying decimated reads when out_shape is
            specified. Supported methods are: 'average', 'bilinear', 'cubic', 'cubic_spline',
            'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'q1', 'q3'.

        file_path : str (optional, default None)
            File path to save to cropped raster. If not supplied then the aggregated
            raster is saved to a temporary file.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.
        
        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then the new
            intersected Raster is created using the dtype of the existing Raster dataset,
            which uses a dtype that can accommodate the data types of all of the
            individual RasterLayers.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's dtype. Note that
            this does not change the pixel nodata values of the raster, it only changes
            the metadata of what value represents a nodata pixel.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.Raster
            Raster object aggregated to a new pixel size.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        rows, cols = out_shape

        arr = self.read(masked=True, out_shape=out_shape, resampling=resampling)

        meta = deepcopy(self.meta)

        dtype = self._check_supported_dtype(dtype)
        if nodata is None:
            nodata = _get_nodata(dtype)

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

        with rasterio.open(file_path, "w", **meta) as dst:
            dst.write(arr.astype(dtype))

        new_raster = self._new_raster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def apply(
        self,
        function,
        file_path=None,
        driver="GTiff",
        dtype=None,
        nodata=None,
        progress=False,
        n_jobs=-1,
        **kwargs,
    ):
        """Apply user-supplied function to a Raster object.

        Parameters
        ----------
        function : function
            Function that takes an numpy array as a single argument.

        file_path : str (optional, default None)
            Optional path to save calculated Raster object. If not specified then a
            tempfile is used.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        dtype : str (optional, default None)
            Coerce RasterLayers to the specified dtype. If not specified then the new
            Raster is created using the dtype of the calculation result.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type. Note that
            this changes the values of the pixels that represent nodata pixels.
        
        n_jobs : int (default -1)
            Number of processing cores to use for parallel execution. Default of -1 is all cores.

        progress : bool (default False)
            Optionally show progress of transform operations.

        kwargs : opt
            Optional named arguments to pass to the format drivers. For example can be
            `compress="deflate"` to add compression.

        Returns
        -------
        pyspatialml.Raster
            Raster containing the calculated result.
        """

        file_path, tfile = _file_path_tempfile(file_path)
        n_jobs = _get_num_workers(n_jobs)

        # perform test calculation determine dimensions, dtype, nodata
        window = Window(0, 0, self.width, 1)
        img = self.read(masked=True, window=window)
        arr = function(img)

        if np.ndim(arr) > 2:
            indexes = np.arange(1, arr.shape[0] + 1)
            count = len(indexes)
        else:
            indexes = 1
            count = 1

        dtype = self._check_supported_dtype(dtype)
        if nodata is None:
            nodata = _get_nodata(dtype)

        # open output file with updated metadata
        meta = deepcopy(self.meta)
        meta.update(driver=driver, count=count, dtype=dtype, nodata=nodata)
        meta.update(kwargs)

        with rasterio.open(file_path, "w", **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            data_gen = (self.read(window=window, masked=True) for window in windows)

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                if progress is True:
                    for window, result, pbar in zip(
                        windows, executor.map(function, data_gen), tqdm(windows)
                    ):

                        result = np.ma.filled(result, fill_value=nodata)
                        dst.write(result.astype(dtype), window=window, indexes=indexes)
                else:
                    for window, result in zip(
                        windows, executor.map(function, data_gen)
                    ):

                        result = np.ma.filled(result, fill_value=nodata)
                        dst.write(result.astype(dtype), window=window, indexes=indexes)

        new_raster = self._new_raster(file_path)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer._close = tfile.close

        return new_raster

    def block_shapes(self, rows, cols):
        """Generator for windows for optimal reading and writing based on the raster
        format Windows are returns as a tuple with xoff, yoff, width, height.

        Parameters
        ----------
        rows : int
            Height of window in rows.

        cols : int
            Width of window in columns.
        """

        for i in range(0, self.width, rows):
            if i + rows < self.width:
                num_cols = rows
            else:
                num_cols = self.width - i

            for j in range(0, self.height, cols):
                if j + cols < self.height:
                    num_rows = rows
                else:
                    num_rows = self.height - j

                yield Window(i, j, num_cols, num_rows)

    def astype(self, dtype, file_path=None, driver="GTiff", nodata=None, **kwargs):
        """Coerce Raster to a different dtype.
        
        Parameters
        ----------
        dtype : str or np.dtype
            Datatype to coerce Raster object
        
        file_path : str (optional, default None)
            Optional path to save calculated Raster object. If not specified then a
            tempfile is used.

        driver : str (default 'GTiff')
            Named of GDAL-supported driver for file export.

        nodata : any number (optional, default None)
            Nodata value for new dataset. If not specified then a nodata value is set
            based on the minimum permissible value of the Raster's data type. Note that
            this changes the values of the pixels that represent nodata pixels.
        
        Returns
        -------
        pyspatialml.Raster
        """

        raise NotImplementedError
