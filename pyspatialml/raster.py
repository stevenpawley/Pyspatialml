from __future__ import print_function

import math
import tempfile
from collections import Counter
from collections import namedtuple

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

from .base import BaseRaster, _file_path_tempfile, _get_nodata
from .indexing import ExtendedDict, LinkedList
from .rasterlayer import RasterLayer


class Raster(BaseRaster):
    """Flexible class that represents a collection of file-based GDAL-supported
    raster datasets which share a common coordinate reference system and
    geometry.

    Raster objects encapsulate RasterLayer objects, which represent single band
    raster datasets that can physically be represented by either separate
    single-band raster files, multi-band raster files, or any combination of
    individual bands from multi-band raster and single-band raster datasets.

    Methods defined in a Raster class comprise those that would typically
    applied to a stack of raster datasets. In addition, these methods always
    return a new Raster object.
    """

    def __init__(self, src=None, arr=None, crs=None, transform=None,
                 nodata=None, mode='r', file_path=None):
        """
        Initiate a new Raster object

        Parameters
        ----------
        src : file path, RasterLayer, or rasterio dataset (opt)
            Initiate a Raster object from any combination of a file path or list
            of file paths to GDAL-supported raster datasets, RasterLayer
            objects, or directly from a rasterio dataset or band object that is
            opened in 'r' or 'rw' mode.

        arr : ndarray (opt)
            Whether to initiate a Raster object from a numpy.ndarray. Additional
            arguments `crs` and `transform` should also be provided to supply
            spatial coordinate information. Parameters `arr` and `src` are
            mutually-exclusive.

        crs : rasterio.crs.CRS object (opt)
            CRS object containing projection information for data if provided by
            the associated `arr` parameter.

        transform : affine.Affine object (opt)
            Affine object containing transform information for data if provided
            by the associated `arr` parameter.

        nodata : any number (opt)
            Assign a nodata value to the Raster dataset when `arr` is used
            for initiation. If a nodata value is not specified then it is
            determined based on the minimum permissible value for the array's
            data type.

        mode : str (opt). Default is 'r'
            Mode used to open the raster datasets. Can be one of 'r', 'r+' or
            'rw'.

        file_path : str (opt)
            Path to save new Raster object if created from `arr`.

        Returns
        -------
        pyspatialml.Raster
            Raster object containing the src layers stacked into a single object
        """

        self.loc = ExtendedDict(self)
        self.iloc = LinkedList(self, self.loc)
        self.files = []
        self.dtypes = []
        self.nodatavals = []
        self.count = 0
        self.res = None
        self.meta = None
        self._block_shape = (256, 256)
        self.max_memory = 5e+09

        # some checks
        if src and arr:
            raise ValueError('Arguments src and arr are mutually exclusive')

        if mode not in ['r', 'r+', 'w']:
            raise ValueError("mode must be one of 'r', 'r+', or 'w'")

        # initiate from array
        if arr is not None:

            if file_path is None:
                file_path = tempfile.NamedTemporaryFile().name

            with rasterio.open(
                fp=file_path, mode='w', driver='GTiff', height=arr.shape[1],
                width=arr.shape[2], count=arr.shape[0],
                dtype=arr.dtype, crs=crs, transform=transform,
                nodata=nodata) as dst:
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
                    band = rasterio.band(r, i+1)
                    src_layers.append(RasterLayer(band))

        # initiate from RasterLayer objects
        elif all(isinstance(x, RasterLayer)
                 for x in src):
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
                'Cannot initiated a Raster from a list of different type '
                'objects')

        # call property with a list of rasterio.band objects
        self._layers = src_layers

    def __getitem__(self, key):
        """
        Subset the Raster object using a label or list of labels.
        
        Parameters
        ----------
        key : str or list
            Key-based indexing of RasterLayer objects within the Raster.
            
        Returns
        -------
        Raster
            A new Raster object only containing the subset of layers specified
            in the label argument.
        """

        if isinstance(key, str):
            key = [key]

        subset_layers = []

        for i in key:

            if i in self.names is False:
                raise KeyError('layername not present in Raster object')
            else:
                subset_layers.append(self.loc[i])

        subset_raster = Raster(subset_layers)

        return subset_raster

    def __setitem__(self, key, value):
        """
        Replace a RasterLayer within the Raster object with a new RasterLayer.
        
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
            raise ValueError('value is not a RasterLayer object')

    def __iter__(self):
        """
        Iterate over RasterLayers
        """
        return iter(self.loc.items())

    def close(self):
        """
        Close all of the RasterLayer objects in the Raster.

        Note that this will cause any rasters based on temporary files to be
        removed. This is intended as a method of clearing temporary files that
        may have accumulated during an analysis session.
        """
        for layer in self.iloc:
            layer.close()

    @staticmethod
    def _check_alignment(layers):
        """
        Check that a list of raster datasets are aligned with the same pixel
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

        if not all(i['crs'] == src_meta[0]['crs'] for i in src_meta):
            Warning('crs of all rasters does not match, '
                    'possible unintended consequences')

        if not all([i['height'] == src_meta[0]['height'] or
                    i['width'] == src_meta[0]['width'] or
                    i['transform'] == src_meta[0]['transform']
                    for i in src_meta]):
            return False

        else:
            return src_meta[0]

    @staticmethod
    def _fix_names(combined_names):
        """
        Adjusts the names of pyspatialml.RasterLayer objects within the Raster
        when appending new layers. This avoids the Raster object containing
        duplicated names in the case that multiple RasterLayer's are appended
        with the same name.

        In the case of duplicated names, the RasterLayer names are appended with
        a `_n` with n = 1, 2, 3 .. n.

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
                        combined_names[combined_names.index(s)] = (
                                s + "_" + str(suffix))
                    else:
                        i = 1
                        while s + "_" + str(i) in combined_names:
                            i += 1
                        combined_names[combined_names.index(s)] = (
                                s + "_" + str(i))

        return combined_names

    @property
    def block_shape(self):
        """
        Return the windows size used for raster calculations,
        specified as a tuple (rows, columns).

        Returns
        -------
        tuple
            Block window shape that is currently set for the Raster as a tuple
            in the format of (n_rows, n_columns) in pixels.
        """
        return self._block_shape

    @block_shape.setter
    def block_shape(self, value):
        """
        Set the windows size used for raster calculations,
        specified as a tuple (rows, columns).

        Parameters
        ----------
        value : tuple
            Tuple of integers for default block shape to read and write
            data from the Raster object for memory-safe calculations.
            Specified as (n_rows, n_columns).
        """
        if not isinstance(value, tuple):
            raise ValueError('block_shape must be set using an integer tuple '
                             'as (rows, cols)')

        rows, cols = value

        if not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError('tuple must consist of integer values referring '
                             'to number of rows, cols')

        self._block_shape = (rows, cols)

    @property
    def names(self):
        """
        Return the names of the RasterLayers in the Raster object.

        Returns
        -------
        list
            List of names of RasterLayer objects
        """
        return list(self.loc.keys())

    @property
    def _layers(self):
        """
        Getter method.

        Returns
        -------
        pyspatialml.indexing.ExtendedDict
            Returns a dict of key-value pairs of names and RasterLayers
        """
        return self.loc

    @_layers.setter
    def _layers(self, layers):
        """
        Setter method for the files attribute in the Raster object.

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
                'Cannot create a Raster object from a mixture of input types')

        meta = self._check_alignment(layers)
        if meta is False:
            raise ValueError(
                'Raster datasets do not all have the same dimensions or '
                'transform')

        # reset existing attributes
        for name in self.names:
            delattr(self, name)

        self.loc = ExtendedDict(self)
        self.iloc = LinkedList(self, self.loc)
        self.files = []
        self.dtypes = []
        self.nodatavals = []

        # update global Raster object attributes with new values
        self.count = len(layers)
        self.width = meta['width']
        self.height = meta['height']
        self.shape = (self.height, self.width)
        self.transform = meta['transform']
        self.res = (abs(meta['transform'].a), abs(meta['transform'].e))
        self.crs = meta['crs']

        bounds = rasterio.transform.array_bounds(
            self.height,
            self.width,
            self.transform)
        BoundingBox = namedtuple(
            'BoundingBox', ['left', 'bottom', 'right', 'top'])
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

        self.meta = dict(crs=self.crs,
                         transform=self.transform,
                         width=self.width,
                         height=self.height,
                         count=self.count,
                         dtype=np.find_common_type([], self.dtypes))

    def read(self, masked=False, window=None, out_shape=None,
             resampling='nearest', **kwargs):
        """
        Reads data from the Raster object into a numpy array.

        Overrides read BaseRaster class read method and replaces it with a
        method that reads from multiple RasterLayer objects.

        Parameters
        ----------
        masked : bool (opt). Default is False
            Read data into a masked array.

        window : rasterio.window.Window object (opt)
            Tuple of col_off, row_off, width, height of a window of data
            to read a chunk of data into a ndarray.

        out_shape : tuple (opt)
            Shape of shape of array (rows, cols) to read data into using
            decimated reads.

        resampling : str (opt). Default is 'nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'.

        **kwargs : dict
            Other arguments to pass to rasterio.DatasetReader.read method

        Returns
        -------
        ndarray
            Raster values in 3d ndarray  with the dimensions in order of
            (band, row, and column).
        """

        dtype = self.meta['dtype']

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
                **kwargs)
            
            if masked is True:
                arr[i, :, :] = np.ma.MaskedArray(
                    data=arr[i, :, :], 
                    mask=np.isfinite(arr[i, :, :]).mask)
                
        return arr

    def write(self, file_path, driver="GTiff", dtype=None, nodata=None):
        """
        Write the Raster object to a file.

        Overrides the write RasterBase class method, which is a partial
        function of the rasterio.DatasetReader.write method.

        Parameters
        ----------
        file_path : str
            File path used to save the Raster object.

        driver : str (opt). Default is 'GTiff'
            Name of GDAL driver used to save Raster data.

        dtype : str (opt)
            Optionally specify a numpy compatible data type when saving to file.
            If not specified, a data type is selected based on the data types of
            RasterLayers in the Raster object.

        nodata : any number (opt)
            Optionally assign a new nodata value when saving to file. If not
            specified a nodata value based on the minimum permissible value for
            the data types of RasterLayers in the Raster object is used.

        Returns
        -------
        Raster
            New Raster object from saved file.
        """
        if dtype is None:
            dtype = self.meta['dtype']

        if nodata is None:
            nodata = _get_nodata(dtype)

        meta = self.meta
        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['dtype'] = dtype

        with rasterio.open(file_path, mode='w', **meta) as dst:

            for i, layer in enumerate(self.iloc):
                arr = layer.read()
                arr[arr == layer.nodata] = nodata
                dst.write(arr.astype(dtype), i+1)

        raster = self._newraster(file_path, self.names)

        return raster

    def to_pandas(self, max_pixels=50000, resampling='nearest'):
        """
        Raster to pandas DataFrame.

        Parameters
        ----------
        max_pixels: int (opt). Default is 50000
            Maximum number of pixels to sample.

        resampling : str (opt). Default is 'nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'/

        Returns
        -------
        pandas.DataFrame
            DataFrame containing values of names of RasterLayers in the Raster
            as columns, and pixel values as rows.
        """

        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels

        # read dataset using decimated reads
        out_shape = (round(self.shape[0] * scaling),
                     round(self.shape[1] * scaling))
        arr = self.read(masked=True,
                        out_shape=out_shape,
                        resampling=resampling)
        
        # x and y grid coordinate arrays
        x_range = np.linspace(start=self.bounds.left,
                              stop=self.bounds.right,
                              num=arr.shape[2])
        y_range = np.linspace(start=self.bounds.top,
                              stop=self.bounds.bottom,
                              num=arr.shape[1])
        xs, ys = np.meshgrid(x_range, y_range)

        arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        arr = arr.transpose()
        df = pd.DataFrame(
            data=np.column_stack((xs.flatten(), ys.flatten(), arr)),
            columns=['x', 'y'] + self.names)

        # set nodata values to nan
        for i, col_name in enumerate(self.names):
            df.loc[df[col_name] == self.nodatavals[i], col_name] = np.nan

        return df

    def predict_proba(self, estimator, file_path=None, indexes=None,
                      driver='GTiff', dtype=None, nodata=None,
                      progress=True):
        """
        Apply class probability prediction of a scikit learn model to a Raster.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str (opt)
            Path to a GeoTiff raster for the prediction results. If not
            specified then the output is written to a temporary file.

        indexes : list of integers (opt)
            List of class indices to export. In some circumstances, only a
            subset of the class probability estimations are desired, for
            instance when performing a binary classification only the
            probabilities for the positive class may be desired.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        dtype : str (opt)
            Optionally specify a numpy compatible data type when saving to file.
            If not specified, a data type is set based on the data type of
            the prediction.

        nodata : any number (opt)
            Nodata value for file export. If not specified then the nodata value
            is derived from the minimum permissible value for the given
            data type.

        progress : bool (opt). Default is True
            Show progress bar for prediction.

        Returns
        -------
        pyspatialml.Raster
            Raster containing predicted class probabilities. Each predicted
            class is represented by a RasterLayer object. The RasterLayers are
            named `prob_n` for 1,2,3..n, with `n` based on the index position
            of the classes, not the number of the class itself.

            For example, a classification model predicting classes with integer
            values of 1, 3, and 5 would result in three RasterLayers named
            prob_1, prob_2 and prob_3.
        """
        file_path, tfile = _file_path_tempfile(file_path)
        predfun = self._probfun

        # determine output count
        if isinstance(indexes, int):
            indexes = range(indexes, indexes + 1)

        elif indexes is None:
            window = Window(0, 0, self.width, 1)
            img = self.read(masked=True, window=window)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(
                1, 2, 0).reshape((n_samples, n_features))
            result = estimator.predict_proba(flat_pixels)
            indexes = np.arange(0, result.shape[1])

        if dtype is None:
            dtype = result.dtype

        if nodata is None:
            nodata = _get_nodata(dtype)

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=len(
            indexes), dtype=dtype, nodata=nodata)

        with rasterio.open(file_path, 'w', **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            data_gen = (self.read(window=window, masked=True)
                        for window in windows)

            if progress is True:
                for window, arr, pbar in zip(windows, data_gen, tqdm(windows)):
                    result = predfun(arr, estimator)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(
                        dtype), window=window)
            else:
                for window, arr in zip(windows, data_gen):
                    result = predfun(arr, estimator)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(
                        dtype), window=window)

        # generate layer names
        prefix = "prob_"
        names = [prefix + str(i) for i in range(len(indexes))]

        # create new raster object
        new_raster = self._newraster(file_path, names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close

        return new_raster

    def predict(self, estimator, file_path=None, driver='GTiff', dtype=None,
                nodata=None, progress=True):
        """
        Apply prediction of a scikit learn model to a Raster. The model can
        represent any scikit learn model or compatible api with a `fit` and
        `predict` method. These can consist of classification or regression
        models. Multi-class classifications and multi-target regressions are
        also supported.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str (opt)
            Path to a GeoTiff raster for the prediction results. If not
            specified then the output is written to a temporary file.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export

        dtype : str (opt)
            Optionally specify a numpy compatible data type when saving to file.
            If not specified, a data type is set based on the data types of
            the prediction.

        nodata : any number (opt)
            Nodata value for file export. If not specified then the nodata value
            is derived from the minimum permissible value for the given data
            type.

        progress : bool (opt). Default is True
            Show progress bar for prediction.

        Returns
        -------
        pyspatial.Raster
            Raster object containing prediction results as a RasterLayers.
            For classification and regression models, the Raster will contain a
            single RasterLayer, unless the model is multi-class or multi-target.
            Layers are named automatically as `pred_raw_n` with n = 1, 2, 3 ..n.
        """
        file_path, tfile = _file_path_tempfile(file_path)

        # determine output count for multi output cases
        window = Window(0, 0, self.width, 1)
        img = self.read(masked=True, window=window)
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
        result = estimator.predict(flat_pixels)

        if result.ndim > 1:
            n_outputs = result.shape[result.ndim-1]
        else:
            n_outputs = 1
        
        indexes = np.arange(0, n_outputs)

        # chose prediction function
        if len(indexes) == 1:
            predfun = self._predfun
        else:
            predfun = self._predfun_multioutput

        if dtype is None:
            dtype = result.dtype

        if nodata is None:
            nodata = _get_nodata(dtype)

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=len(
            indexes), dtype=dtype, nodata=nodata)

        with rasterio.open(file_path, 'w', **meta) as dst:
            # define windows
            # windows = [window for ij, window in dst.block_windows()]
            windows = [window for window in self.block_shapes(*self._block_shape)]

            # generator gets raster arrays for each window
            data_gen = (self.read(window=window, masked=True)
                        for window in windows)

            if progress is True:
                for window, arr, pbar in zip(windows, data_gen, tqdm(windows)):
                    result = predfun(arr, estimator)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(
                        dtype), window=window)
            else:
                for window, arr in zip(windows, data_gen):
                    result = predfun(arr, estimator)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(
                        dtype), window=window)

        # generate layer names
        prefix = "pred_raw_"
        names = [prefix + str(i) for i in range(len(indexes))]

        # create new raster object
        new_raster = self._newraster(file_path, names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close

        return new_raster

    def _predfun(self, img, estimator):
        """
        Prediction function for classification or regression response.

        Parameters
        ----
        img : numpy.ndarray
            3d ndarray of raster data with the dimensions in order of
            (band, rows, columns).

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        Returns
        -------
        numpy.ndarray
            2d numpy array representing a single band raster containing the
            classification or regression result.
        """

        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

        # create mask for NaN values and replace with number
        flat_pixels_mask = flat_pixels.mask.copy()

        # prediction
        result_cla = estimator.predict(flat_pixels)

        # replace mask
        result_cla = np.ma.masked_array(
            data=result_cla, mask=flat_pixels_mask.any(axis=1))

        # reshape the prediction from a 1D into 3D array [band, row, col]
        result_cla = result_cla.reshape((1, rows, cols))

        return result_cla

    @staticmethod
    def _probfun(img, estimator):
        """
        Class probabilities function.

        Parameters
        ----------
        img : numpy.ndarray
            3d numpy array of raster data with the dimensions in order of
            (band, row, column).

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        Returns
        -------
        numpy.ndarray
            Multi band raster as a 3d numpy array containing the probabilities
            associated with each class. ndarray dimensions are in the order of
            (class, row, column).
        """

        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        mask2d = img.mask.any(axis=0)

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

        # predict probabilities
        result_proba = estimator.predict_proba(flat_pixels)

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result_proba = result_proba.reshape((rows, cols, result_proba.shape[1]))

        # reshape band into rasterio format [band, row, col]
        result_proba = result_proba.transpose(2, 0, 1)

        # repeat mask for n_bands
        mask3d = np.repeat(a=mask2d[np.newaxis, :, :],
                           repeats=result_proba.shape[0], axis=0)

        # convert proba to masked array
        result_proba = np.ma.masked_array(
            result_proba,
            mask=mask3d,
            fill_value=np.nan)

        return result_proba

    @staticmethod
    def _predfun_multioutput(img, estimator):
        """
        Multi-target prediction function.

        Parameters
        ----------
        img : numpy.ndarray
            3d numpy array of raster data with the dimensions in order of
            (band, row, column).

        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        Returns
        -------
        numpy.ndarray
            3d numpy array representing the multi-target prediction result with
            the dimensions in the order of (target, row, column).
        """

        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        mask2d = img.mask.any(axis=0)

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

        # predict probabilities
        result = estimator.predict(flat_pixels)

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result = result.reshape((rows, cols, result.shape[1]))

        # reshape band into rasterio format [band, row, col]
        result = result.transpose(2, 0, 1)

        # repeat mask for n_bands
        mask3d = np.repeat(a=mask2d[np.newaxis, :, :],
                           repeats=result.shape[0], axis=0)

        # convert proba to masked array
        result = np.ma.masked_array(
            result,
            mask=mask3d,
            fill_value=np.nan)

        return result

    def append(self, other, in_place=True):
        """
        Method to add new RasterLayers to a Raster object.
        
        Note that this modifies the Raster object in-place by default.

        Parameters
        ----------
        other : Raster object, or list of Raster objects
            Object to append to the Raster.
        
        in_place : bool (opt). Default is True
            Whether to change the Raster object in-place or leave original and
            return a new Raster object.

        Returns
        -------
        pyspatialml.Raster
            Returned only if `in_place` is True
        """

        if isinstance(other, Raster):
            other = [other]

        for new_raster in other:

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
                new_raster = self._newraster(self.files, self.names)
                new_raster._layers = combined_layers
                
                return new_raster

    def drop(self, labels, in_place=True):
        """
        Drop individual RasterLayers from a Raster object.
        
        Note that this modifies the Raster object in-place by default.
        
        Parameters
        ---------
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or label,
            or a list of integers or labels.
        
        in_place : bool (opt). Default is True
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

            subset_layers = [v for (i, v) in enumerate(
                list(self.loc.values())) if i not in labels]

        # str label based subsetting
        elif len([i for i in labels if isinstance(i, str)]) == len(labels):

            subset_layers = [v for (i, v) in enumerate(
                list(self.loc.values())) if self.names[i] not in labels]

        else:
            raise ValueError(
                'Cannot drop layers based on mixture of indexes and labels')

        if in_place is True:
            self._layers = subset_layers
        else:
            new_raster = self._newraster(self.files, self.names)
            new_raster._layers = subset_layers
            
            return new_raster

    def rename(self, names, in_place=True):
        """
        Rename a RasterLayer within the Raster object.
        
        Note that by default this modifies the Raster object in-place.

        Parameters
        ----------
        names : dict
            dict of old_name : new_name
        
        in_place : bool (opt). Default is True
            Whether to change names of the Raster object in-place or leave
            original and return a new Raster object.

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
                self.loc[new_name] = self.loc.pop(old_name)
        else:
            new_raster = self._newraster(self.files, self.names)
            for old_name, new_name in names.items():
                # change internal name of RasterLayer
                new_raster.loc[old_name].names = [new_name]

                # change name of layer in stack
                new_raster.loc[new_name] = new_raster.loc.pop(old_name)
                
            return new_raster

    def plot(self, out_shape=(100, 100), label_fontsize=8, title_fontsize=8,
             names=None, **kwargs):
        """
        Plot a Raster object as a raster matrix.

        Parameters
        ----------
        out_shape : tuple (opt). Default is (100, 100)
            Number of rows, cols to read from the raster datasets for plotting.

        label_fontsize : any number (opt). Default is 8
            Size in pts of labels.

        title_fontsize : any number (opt). Default is 8
            Size in pts of titles.

        names : list (opt)
            Optionally supply a list of names for each RasterLayer to override
            the default layer names for the titles.

        **kwargs : dict
            Additional arguments to pass when creating the figure object.

        Returns
        -------
        tuple

        Two items
            matplotlib.figure.Figure
            matplotlib.axes.Axes
        """

        if self.count == 1:
            fig, ax = self.iloc[0].plot()
            return fig, ax

        # estimate required number of rows and columns in figure
        rows = int(np.sqrt(self.count))
        cols = int(math.ceil(np.sqrt(self.count)))

        if rows * cols < self.count:
            rows += 1

        cmaps = [i.cmap for i in self.iloc]

        if names is None:
            names = self.names

        fig, axs = plt.subplots(rows, cols, **kwargs)

        # axs.flat is an iterator over the row-order flattened axs array
        for ax, n, cmap, name in zip(axs.flat, range(self.count), cmaps, names):

            arr = self.iloc[n].read(masked=True, out_shape=out_shape)

            ax.set_title(name, fontsize=title_fontsize, y=1.00)
            im = ax.imshow(
                arr,
                extent=[self.bounds.left, self.bounds.right,
                        self.bounds.bottom, self.bounds.top],
                cmap=cmap)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=label_fontsize)

            # hide tick labels by default when multiple rows or cols
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])

            # show y-axis tick labels on first subplot
            if n == 0 and rows > 1:
                ax.set_yticklabels(
                    ax.yaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)
            if n == 0 and rows == 1:
                ax.set_xticklabels(
                    ax.xaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)
                ax.set_yticklabels(
                    ax.yaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)
            if rows > 1 and n == (rows * cols) - cols:
                ax.set_xticklabels(
                    ax.xaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)

        for ax in axs.flat[axs.size - 1:self.count - 1:-1]:
            ax.set_visible(False)

        plt.subplots_adjust()

        return fig, axs

    def _newraster(self, file_path, names=None):
        """
        Return a new Raster object.

        Parameters
        ----------
        file_path : str
            Path to files to create the new Raster object from.

        names : list (opt)
            List to name the RasterLayer objects in the stack. If not supplied
            then the names will be generated from the file names.

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

    def mask(self, shapes, invert=False, crop=True, filled=True,
             pad=False, file_path=None, driver='GTiff', dtype=None,
             nodata=None):
        """
        Mask a Raster object based on the outline of shapes in a
        geopandas.GeoDataFrame.

        Parameters
        ----------
        shapes : geopandas.GeoDataFrame
            GeoDataFrame containing masking features.

        invert : bool (opt). Default is False
            If False then pixels outside shapes will be masked. If True then
            pixels inside shape will be masked.

        crop : bool (opt). Default is True
            Crop the raster to the extent of the shapes.

        filled : bool (opt). Default is True
            If True, the pixels outside the features will be set to nodata.
            If False, the output array will contain the original pixel data,
            and only the mask will be based on shapes.

        pad : bool (opt). Default is False
            If True, the features will be padded in each direction by one half
            of a pixel prior to cropping raster.

        file_path : str (opt). Default is None
            File path to save to resulting Raster. If not supplied then the
            resulting Raster is saved to a temporary file

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        dtype : str (opt)
            Coerce RasterLayers to the specified dtype. If not specified then
            the cropped Raster is created using the existing dtype, which uses a
            dtype that can accommodate the data types of all of the individual
            RasterLayers.

        nodata : any number (opt)
            Nodata value for cropped dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            data type.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        masked_ndarrays = []

        for layer in self.iloc:
            masked_arr, transform = rasterio.mask.mask(
                dataset=layer.ds, shapes=[shapes.geometry.unary_union],
                filled=filled, invert=invert, crop=crop, pad=pad)

            if layer.ds.count > 1:
                masked_arr = masked_arr[layer.bidx - 1, :, :]

            else:
                masked_arr = np.squeeze(masked_arr)

            masked_ndarrays.append(masked_arr)

        # stack list of 2d arrays into 3d array
        masked_ndarrays = np.stack(masked_ndarrays)

        # write to file
        meta = self.meta
        meta['transform'] = transform
        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['height'] = masked_ndarrays.shape[1]
        meta['width'] = masked_ndarrays.shape[2]
        
        if dtype is None:
            dtype = meta['dtype']

        if nodata is None:
            nodata = _get_nodata(dtype)
        
        meta['dtype'] = dtype

        masked_ndarrays = np.ma.filled(masked_ndarrays, fill_value=nodata)

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(masked_ndarrays.astype(dtype))

        new_raster = self._newraster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close
        
        return new_raster

    def intersect(self, file_path=None, driver='GTiff', dtype=None,
                  nodata=None):
        """
        Perform a intersect operation on the Raster object.

        Computes the geometric intersection of the RasterLayers with the
        Raster object. This will cause nodata values in any of the rasters
        to be propagated through all of the output rasters.

        Parameters
        ----------
        file_path : str (opt)
            File path to save to resulting Raster. If not supplied then the
            resulting Raster is saved to a temporary file.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        dtype : str (opt)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new intersected Raster is created using the dtype of the
            existing Raster dataset, which uses a dtype that can accommodate
            the data types of all of the individual RasterLayers.

        nodata : any number (opt)
            Nodata value for new dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            data type.

        Returns
        -------
        pyspatialml.Raster
            Raster with layers that are masked based on a union of all masks
            in the suite of RasterLayers.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        arr = self.read(masked=True)
        mask_2d = arr.mask.any(axis=0)

        # repeat mask for n_bands
        mask_3d = np.repeat(a=mask_2d[np.newaxis, :, :],
                           repeats=self.count, axis=0)

        intersected_arr = np.ma.masked_array(
            arr, mask=mask_3d, fill_value=nodata)

        intersected_arr = np.ma.filled(intersected_arr, fill_value=nodata)

        meta = self.meta

        if dtype is None:
            dtype = meta['dtype']

        if nodata is None:
            nodata = _get_nodata(dtype)

        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['dtype'] = dtype

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(intersected_arr.astype(dtype))

        new_raster = self._newraster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close

        return new_raster

    def crop(self, bounds, file_path=None, driver='GTiff', dtype=None,
             nodata=None):
        """
        Crops a Raster object by the supplied bounds.

        Parameters
        ----------
        bounds : tuple
            A tuple containing the bounding box to clip by in the
            form of (xmin, xmax, ymin, ymax).

        file_path : str (opt)
            File path to save to cropped raster. If not supplied then the
            cropped raster is saved to a temporary file.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.
        
        dtype : str (opt)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new intersected Raster is created using the dtype of the
            existing Raster dataset, which uses a dtype that can accommodate
            the data types of all of the individual RasterLayers.

        nodata : any number (opt)
            Nodata value for new dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            data type.

        Returns
        -------
        pyspatialml.Raster
            Raster cropped to new extent.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        xmin, ymin, xmax, ymax = bounds

        rows, cols = rasterio.transform.rowcol(
            transform=self.transform,
            xs=(xmin, xmax),
            ys=(ymin, ymax))

        window = Window(col_off=min(cols),
                        row_off=min(rows),
                        width=max(cols)-min(cols),
                        height=max(rows)-min(rows))

        cropped_arr = self.read(masked=True, window=window)
        meta = self.meta
        aff = self.transform

        if dtype is None:
            dtype = meta['dtype']

        if nodata is None:
            nodata = _get_nodata(dtype)

        meta['width'] = max(cols) - min(cols)
        meta['height'] = max(rows) - min(rows)
        meta['transform'] = Affine(aff.a, aff.b, xmin, aff.d, aff.e, ymin)
        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['dtype'] = dtype

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(cropped_arr.astype(dtype))

        new_raster = self._newraster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close
        
        return new_raster

    def to_crs(self, crs, resampling='nearest', file_path=None, driver='GTiff',
               nodata=None, n_jobs=1, warp_mem_lim=0, progress=False):
        """
        Reprojects a Raster object to a different crs.

        Parameters
        ----------
        crs : rasterio.transform.CRS object, or dict
            Example: CRS({'init': 'EPSG:4326'})

        resampling : str (opt)
            Resampling method to use.  One of the following:
            Resampling.nearest,
            Resampling.bilinear,
            Resampling.cubic,
            Resampling.cubic_spline,
            Resampling.lanczos,
            Resampling.average,
            Resampling.mode,
            Resampling.max (GDAL >= 2.2),
            Resampling.min (GDAL >= 2.2),
            Resampling.med (GDAL >= 2.2),
            Resampling.q1 (GDAL >= 2.2),
            Resampling.q3 (GDAL >= 2.2)

        file_path : str (opt)
            Optional path to save reprojected Raster object. If not
            specified then a tempfile is used.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        nodata : any number (opt)
            Nodata value for new dataset. If not specified then the existing
            nodata value of the Raster object is used, which can accommodate
            the dtypes of the individual layers in the Raster.

        n_jobs : int (opt). Default is 1
            The number of warp worker threads.

        warp_mem_lim : int (opt). Default is 0
            The warp operation memory limit in MB. Larger values allow the
            warp operation to be carried out in fewer chunks. The amount of
            memory required to warp a 3-band uint8 2000 row x 2000 col
            raster to a destination of the same size is approximately
            56 MB. The default (0) means 64 MB with GDAL 2.2.

        progress : bool (opt). Default is False
            Optionally show progress of transform operations.

        Returns
        -------
        pyspatialml.Raster
            Raster following reprojection.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        resampling_methods = [i.name for i in rasterio.enums.Resampling]
        if resampling not in resampling_methods:
            raise ValueError(
                'Invalid resampling method.' +
                'Resampling method must be one of {0}:'.format(
                    resampling_methods))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs=self.crs,
            dst_crs=crs,
            width=self.width,
            height=self.height,
            left=self.bounds.left,
            right=self.bounds.right,
            bottom=self.bounds.bottom,
            top=self.bounds.top)

        meta = self.meta
        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['width'] = dst_width
        meta['height'] = dst_height
        meta['transform'] = dst_transform
        meta['crs'] = crs

        with rasterio.open(file_path, 'w', **meta) as dst:
            if progress is True:
                t = tqdm(total=self.count)
                for i, layer in enumerate(self.iloc):
                    reproject(
                        source=rasterio.band(layer.ds, layer.bidx),
                        destination=rasterio.band(dst, i+1),
                        dst_transform=dst_transform,
                        dst_crs=crs,
                        dst_nodata=nodata,
                        resampling=rasterio.enums.Resampling[resampling],
                        num_threads=n_jobs,
                        warp_mem_lim=warp_mem_lim)
                    if progress is True:
                        t.update()

        new_raster = self._newraster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close
        
        return new_raster

    def aggregate(self, out_shape, resampling='nearest', file_path=None, 
                  driver='GTiff', dtype=None, nodata=None):
        """
        Aggregates a raster to (usually) a coarser grid cell size.

        Parameters
        ----------
        out_shape : tuple
            New shape in (rows, cols).

        resampling : str, optional. Default is 'nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'.

        file_path : str (opt)
            File path to save to cropped raster. If not supplied then the
            aggregated raster is saved to a temporary file.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.
        
        dtype : str (opt)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new intersected Raster is created using the dtype of the
            existing Raster dataset, which uses a dtype that can accommodate
            the data types of all of the individual RasterLayers.

        nodata : any number (opt)
            Nodata value for new dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            dtype.

        Returns
        -------
        pyspatialml.Raster
            Raster object aggregated to a new pixel size.
        """

        file_path, tfile = _file_path_tempfile(file_path)

        rows, cols = out_shape

        arr = self.read(masked=True, out_shape=out_shape,
                        resampling=resampling)

        meta = self.meta

        if dtype is None:
            dtype = meta['dtype']

        if nodata is None:
            nodata = _get_nodata(dtype)

        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['height'] = rows
        meta['width'] = cols
        meta['dtype'] = dtype

        bnd = self.bounds
        meta['transform'] = rasterio.transform.from_bounds(
            west=bnd.left, south=bnd.bottom, east=bnd.right, north=bnd.top,
            width=cols, height=rows)

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(arr.astype(dtype))
            
        new_raster = self._newraster(file_path, self.names)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close

        return new_raster

    def calc(self, function, file_path=None, driver='GTiff', dtype=None,
             nodata=None, progress=False):
        """
        Apply user-supplied function to a Raster object.

        Parameters
        ----------
        function : function
            Function that takes an numpy array as a single argument.

        file_path : str (opt)
            Optional path to save calculated Raster object. If not
            specified then a tempfile is used.

        driver : str (opt). Default is 'GTiff'
            Named of GDAL-supported driver for file export.

        dtype : str (opt)
            Coerce RasterLayers to the specified dtype. If not specified then
            the new Raster is created using the dtype of the calculation
            result.

        nodata : any number (opt)
            Nodata value for new dataset. If not specified then a nodata
            value is set based on the minimum permissible value of the Raster's
            data type.

        progress : bool (opt). Default is False
            Optionally show progress of transform operations.

        Returns
        -------
        pyspatialml.Raster
            Raster containing the calculated result.
        """
        
        file_path, tfile = _file_path_tempfile(file_path)

        # determine output dimensions
        window = Window(0, 0, 1, self.width)
        img = self.read(masked=True, window=window)
        arr = function(img)

        if len(arr.shape) > 2:
            indexes = range(arr.shape[0])
            count = len(indexes)
        else:
            indexes = 1
            count = 1

        # perform test calculation
        arr = self.read(masked=True, window=Window(0, 0, 1, 1))
        arr = function(arr)

        if dtype is None:
            dtype = arr.dtype

        if nodata is None:
            nodata = _get_nodata(dtype)

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=count, dtype=dtype, nodata=nodata)

        with rasterio.open(file_path, 'w', **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            data_gen = (self.read(window=window, masked=True)
                        for window in windows)

            if progress is True:
                for window, arr, pbar in zip(windows, data_gen, tqdm(windows)):
                    result = function(arr)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result.astype(dtype), window=window)
            else:
                for window, arr in zip(windows, data_gen):
                    result = function(arr)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result.astype(dtype), window=window)

        new_raster = self._newraster(file_path)

        if tfile is not None:
            for layer in new_raster.iloc:
                layer.close = tfile.close
        
        return new_raster

    def block_shapes(self, rows, cols):
        """
        Generator for windows for optimal reading and writing based on the
        raster format Windows are returns as a tuple with xoff, yoff, width,
        height.

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

    def _extract_by_indices(self, rows, cols):
        """
        Spatial query of Raster object (by-band).
        """
        X = np.ma.zeros((len(rows), self.count), dtype='float32')

        for i, layer in enumerate(self.iloc):
            arr = layer.read(masked=True)
            X[:, i] = arr[rows, cols]

        return X
