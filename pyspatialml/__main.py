import os
import tempfile
from collections import namedtuple
from itertools import chain
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Point
from tqdm import tqdm


class RasterStack:
    def __init__(self, files):
        """Constructor to create a RasterStack Class

        Args
        ----
        files : list, str
            List of file paths of individual raster datasets to be stacked"""

        self.loc = {}          # name-based indexing
        self.iloc = []         # index-based indexing
        self.names = []        # short-names of datasets with appended band number for multi-band datasets
        self.attr_names = []   # class attribute names, short-names of rasterio.io.DatasetReader objects
        self.dtypes = []       # dtypes of stacked raster datasets and bands
        self.nodatavals = []   # no data values of stacked raster datasets and bands
        self.width = None      # width of aligned raster datasets in pixels
        self.height = None     # height of aligned raster datasets in pixels
        self.shape = None      # shape of aligned raster datasets (rows, cols)
        self.count = 0         # number of bands in stacked raster datasets
        self.transform = None  # transform of aligned raster datasets
        self.res = None        # (x, y) resolution of aligned raster datasets
        self.crs = None        # crs of aligned raster datasets
        self.bounds = None     # BoundingBox class (namedtuple) ('left', 'bottom', 'right', 'top')
        self.meta = None       # dict containing 'crs', 'transform', 'width', 'height', 'count', 'dtype'

        self._files = None     # set proxy for self._files
        self.files = files     # call property

    def __getitem__(self, x):
        return getattr(self, x)

    def __del__(self):
        """Deconstructor for RasterLayer class to close files"""

        self.close()

    def open(self):
        """Open raster datasets contained within the RasterStack object"""

        self.files = self.files

    def close(self):
        """Deconstructor for RasterLayer class to close files"""

        for src in self.iloc:
            src.close()

    def append(self, other, inplace=False):
        """Setter method to add new raster datasets to the RasterStack object

        Args
        ----
        other : str, list-like, or RasterStack object
            File path or list of file paths to GDAL-supported raster datasets to add
            to the RasterStack object. Also supports appending another RasterStack object
        inplace : bool, default = False
            Modify the RasterStack in place"""

        if isinstance(other, str):
            other = [other]

        elif isinstance(other, RasterStack):
            other = other.files

        if inplace is True:
            self.files = self.files + other
        else:
            new_stack = RasterStack(self.files + other)
            return new_stack

    def drop(self, labels, inplace=False):
        """Drop raster datasets from the RasterStack object

        Args
        ----
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or label,
            or a list of integers or labels
        inplace : bool, default = False
            Modify the RasterStack in place"""

        if isinstance(labels, (str, int)):
            labels = [labels]

        existing_files = deepcopy(self.files)

        # index-based method
        if len([i for i in labels if isinstance(i, int)]) == len(labels):
            existing_files = [existing_files[i] for i in range(len(existing_files)) if i not in labels]

        # label-based method
        elif len([i for i in labels if isinstance(i, str)]) == len(labels):
            for fp in labels:
                existing_files = [i for i in existing_files if fp not in i]
        else:
            raise ValueError('Cannot drop layers based on mixture of indexes and labels')

        if inplace is True:
            self.files = existing_files
        else:
            new_stack = RasterStack(existing_files)
            return new_stack

    @property
    def files(self):
        """Getter method for file names within the RasterStack object"""

        return self._files

    @files.setter
    def files(self, values):
        """Setter method for the files attribute in the RasterStack object

        Performs checks that layers are spatially aligned and sets
        metadata attributes

        Args
        ----
        values : list-like
            List of file paths to GDAL-supported raster datasets"""

        try:
            values, labels = values
        except ValueError:
            labels = None

        if isinstance(values, str):
            values = [values]

        meta = self._check_alignment(values)
        if meta is False:
            raise ValueError(
                'Raster datasets do not all have the same dimensions or transform')
        else:
            # reset existing attributes
            for name in self.attr_names:
                delattr(self, name)

            self.iloc = []
            self.names = []
            self.attr_names = []
            self.dtypes = []
            self.nodatavals = []

            # update attributes with new values
            self.width = meta['width']
            self.height = meta['height']
            self.shape = (self.height, self.width)
            self.count = 0
            self.transform = meta['transform']
            self.res = (abs(meta['transform'].a),
                        abs(meta['transform'].e))
            self.crs = meta['crs']

            bounds = rasterio.transform.array_bounds(
                self.height, self.width, self.transform)
            BoundingBox = namedtuple('BoundingBox', ['left', 'bottom', 'right', 'top'])
            self.bounds = BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])

            self._files = values

            # attach datasets to attributes
            for i, fp in enumerate(values):

                # generate attribute names from file names
                if labels is None:
                    valid_name = self._make_name(fp)
                    self.attr_names.append(valid_name)
                else:
                    valid_name = labels[i]

                src = rasterio.open(fp)
                self.count += src.count
                self.nodatavals += src.nodatavals

                if src.count > 1:
                    self.dtypes += src.dtypes
                    self.names += ['.'.join([valid_name, str(i+1)])
                                   for i in range(src.count)]
                else:
                    self.dtypes.append(src.meta['dtype'])
                    self.names.append(valid_name)

                self.loc.update({valid_name: src})
                self.iloc.append(src)
                setattr(self, valid_name, src)

            self.meta = dict(crs=self.crs,
                             transform=self.transform,
                             width=self.width,
                             height=self.height,
                             count=self.count,
                             dtype=self._maximum_dtype())

    def read(self, masked=False, window=None):
        """Reads data from the RasterStack object into a numpy array

        Args
        ----
        masked : bool, default = False
            Read data into a masked array

        window : rasterio.window.Window object
            Tuple of col_off, row_off, width, height of a window of data
            to read"""

        dtype = self._maximum_dtype()

        # get window to read from window or height/width of dataset
        if window is None:
            width = self.width
            height = self.height
        else:
            width = window.width
            height = window.height

        # read masked or non-masked data
        if masked is True:
            arr = np.ma.zeros((self.count, height, width), dtype=dtype)
        else:
            arr = np.zeros((self.count, height, width), dtype=dtype)

        # read bands separately into numpy array
        for i, src in enumerate(self.iloc):
            if src.count == 1:
                arr[i, :, :] = src.read(1, masked=masked, window=window)
            else:
                for j in range(src.count):
                    arr[i+j, :, :] = src.read(j+1, masked=masked, window=window)

        return arr

    def _maximum_dtype(self):
        """Returns a single dtype that is large enough to store data
        within all raster bands"""

        if 'complex128' in self.dtypes:
            dtype = 'complex128'
        elif 'complex64' in self.dtypes:
            dtype = 'complex64'
        elif 'complex' in self.dtypes:
            dtype = 'complex'
        elif 'float64' in self.dtypes:
            dtype = 'float64'
        elif 'float32' in self.dtypes:
            dtype = 'float32'
        elif 'int32' in self.dtypes:
            dtype = 'int32'
        elif 'uint32' in self.dtypes:
            dtype = 'uint32'
        elif 'int16' in self.dtypes:
            dtype = 'int16'
        elif 'uint16' in self.dtypes:
            dtype = 'uint16'
        elif 'uint16' in self.dtypes:
            dtype = 'uint16'
        elif 'bool' in self.dtypes:
            dtype = 'bool'

        return dtype

    @staticmethod
    def _check_alignment(file_paths):
        """Check that a list of rasters are aligned with the same pixel dimensions
        and geotransforms

        Args
        ----
        file_paths: list-like
            List of file paths to the GDAL-supported rasdters

        Returns
        -------
        src_meta: dict
            Dict containing raster metadata"""

        src_meta = []
        for raster in file_paths:
            with rasterio.open(raster) as src:
                src_meta.append(src.meta.copy())

        if not all(i['crs'] == src_meta[0]['crs'] for i in src_meta):
            Warning('crs of all rasters does not match, '
                    'possible unintended consequences')

        if not all([i['height'] == src_meta[0]['height'] or
                    i['width'] == src_meta[0]['width'] or
                    i['transform'] == src_meta[0]['transform'] for i in src_meta]):
            return False
        else:
            return src_meta[0]

    def _make_name(self, name):
        """Converts a filename to a valid class attribute name

        Args
        ----
        name : str
            File name for convert to a valid class attribute name"""

        valid_name = os.path.basename(name)
        valid_name = valid_name.split(os.path.extsep)[0]
        valid_name = valid_name.replace(' ', '_')

        # check to see if same name already exists
        if valid_name in self.attr_names:
            valid_name = '_'.join([valid_name, '1'])

        return valid_name

    def _extract_by_indices(self, rows, cols):
        """spatial query of RasterStack (by-band)"""

        X = np.ma.zeros((len(rows), self.count))

        for i, src in enumerate(self.iloc):
            if src.count == 1:
                raster_arr = src.read(masked=True)
                X[:, i] = raster_arr[:, rows, cols]
            else:
                for j in range(src.count):
                    raster_arr = src.read(j+1, masked=True)
                    X[:, i+j] = raster_arr[rows, cols]

        return X

    def _clip_xy(self, xy, y=None):
        """Clip array of xy coordinates to extent of RasterStack"""

        extent = self.bounds
        valid_idx = np.where((xy[:, 0] > extent.left) &
                             (xy[:, 0] < extent.right) &
                             (xy[:, 1] > extent.bottom) &
                             (xy[:, 1] < extent.top))[0]
        xy = xy[valid_idx, :]

        if y is not None:
            y = y[valid_idx]

        return xy, y

    def extract_vector(self, response, field, return_array=False, na_rm=True, low_memory=False):
        """Sample the RasterStack by a geopandas GeoDataframe containing points,
        lines or polygon features

        Args
        ----
        response: Geopandas DataFrame
            Containing either point, line or polygon geometries. Overlapping
            geometries will cause the same pixels to be sampled.

        field : str, optional
            Field name of attribute to be used the label the extracted data
            Used only if the response feature represents a GeoDataframe

        return_array : bool, default = False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays

        na_rm : bool, default = True
            Optionally remove rows that contain nodata values

        low_memory : bool, default = False
            Optionally extract pixel values in using a slower but memory-safe
            method

        Returns
        -------
        gpd : geopandas GeoDataframe
            Containing extracted data as point geometries

        X : array-like
            Numpy masked array of extracted raster values, typically 2d

        y: 1d array like
            Numpy masked array of labelled sampled

        xy: 2d array-like
            Numpy masked array of row and column indexes of training pixels"""

        if not field:
            y = None

        # polygon and line geometries
        if all(response.geom_type == 'Polygon') or all(response.geom_type == 'LineString'):

            if all(response.geom_type == 'LineString'):
                all_touched = True
            else:
                all_touched = False

            rows_all, cols_all, y_all = [], [], []

            for i, shape in response.iterrows():

                if not field:
                    shapes = (shape.geometry, 1)
                else:
                    shapes = (shape.geometry, shape[field])

                arr = np.zeros((self.height, self.width))
                arr[:] = -99999
                arr = rasterio.features.rasterize(
                    shapes=(shapes for i in range(1)), fill=-99999, out=arr,
                    transform=self.transform, default_value=1, all_touched=all_touched)

                rows, cols = np.nonzero(arr != -99999)

                if field:
                    y_all.append(arr[rows, cols])

                rows_all.append(rows)
                cols_all.append(cols)

            rows = list(chain.from_iterable(rows_all))
            cols = list(chain.from_iterable(cols_all))
            y = list(chain.from_iterable(y_all))

            xy = np.transpose(
                rasterio.transform.xy(transform=self.transform,
                                      rows=rows, cols=cols))

        # point geometries
        elif all(response.geom_type == 'Point'):
            xy = response.bounds.iloc[:, 2:].values
            if field:
                y = response[field].values

            # clip points to extent of raster
            xy, y = self._clip_xy(xy, y)
            rows, cols = rasterio.transform.rowcol(
                transform=self.transform, xs=xy[:, 0], ys=xy[:, 1])

        # spatial query of RasterStack (by-band)
        if low_memory is False:
            X = self._extract_by_indices(rows, cols)
        else:
            X = self.extract_xy(xy)

        # mask nodata values
        ## flatten masks for X and broadcast for two bands (x & y)
        mask_2d = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))

        ## apply mask to y values and spatial coords only if na_rm is True
        ## otherwise we want to get the y values and coords back even if some of the
        ## X values include nans
        if field and na_rm is True:
            y = np.ma.masked_array(y, mask=X.mask.any(axis=1))
            xy = np.ma.masked_array(xy, mask=mask_2d)

        # optionally remove rows containing nodata
        if na_rm is True:
            mask = X.mask.any(axis=1)
            X = X[~mask].data
            y = y[~mask].data
            xy = xy[~mask].data

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            column_names = [field] + self.names
            gdf = pd.DataFrame(np.ma.column_stack((y, X)), columns=column_names)
            gdf['geometry'] = list(zip(xy[:, 0], xy[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)
            return gdf
        else:
            return X, y, xy

    def extract_raster(self, response, value_name='value', return_array=False, na_rm=True):
        """Sample the RasterStack by an aligned raster of labelled pixels

        Args
        ----
        response: rasterio.io.DatasetReader
            Single band raster containing labelled pixels

        return_array : bool, default = False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays

        na_rm : bool, default = True
            Optionally remove rows that contain nodata values

        Returns
        -------
        gpd : geopandas GeoDataFrame
            Geodataframe containing extracted data as point features

        X : array-like
            Numpy masked array of extracted raster values, typically 2d

        y: 1d array like
            Numpy masked array of labelled sampled

        xy: 2d array-like
            Numpy masked array of row and column indexes of training pixels"""

        # open response raster and get labelled pixel indices and values
        arr = response.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)
        xy = np.transpose(rasterio.transform.xy(response.transform, rows, cols))
        y = arr.data[rows, cols]

        # extract RasterStack values at row, col indices
        X = self._extract_by_indices(rows, cols)

        # summarize data and mask nodatavals in X, y, and xy
        mask_2d = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
        y = np.ma.masked_array(y, mask=X.mask.any(axis=1))
        xy = np.ma.masked_array(xy, mask=mask_2d)

        if na_rm is True:
            mask = X.mask.any(axis=1)
            X = X[~mask].data
            y = y[~mask].data
            xy = xy[~mask].data

        if return_array is False:
            column_names = [value_name] + self.names
            gdf = pd.DataFrame(np.ma.column_stack((y, X)), columns=column_names)
            gdf['geometry'] = list(zip(xy[:, 0], xy[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)
            return gdf
        else:
            return X, y, xy

    def extract_xy(self, xy):
        """Samples pixel values of the RasterStack using an array of xy locations

        Parameters
        ----------
        xy : 2d array-like
            x and y coordinates from which to sample the raster (n_samples, xy)

        Returns
        -------
        values : 2d array-like
            Masked array containing sampled raster values (sample, bands)
            at x,y locations"""

        # clip coordinates to extent of raster
        extent = self.bounds
        valid_idx = np.where((xy[:, 0] > extent.left) &
                             (xy[:, 0] < extent.right) &
                             (xy[:, 1] > extent.bottom) &
                             (xy[:, 1] < extent.top))[0]
        xy = xy[valid_idx, :]

        dtype = self._maximum_dtype()
        values = np.ma.zeros((xy.shape[0], self.count), dtype=dtype)
        rows, cols = rasterio.transform.rowcol(
            transform=self.transform, xs=xy[:, 0], ys=xy[:, 1])

        for i, (row, col) in enumerate(zip(rows, cols)):
            window = Window(col_off=col,
                            row_off=row,
                            width=1,
                            height=1)
            values[i, :] = self.read(masked=True, window=window).reshape((1, self.count))

        return values

    def predict(self, estimator, file_path=None, predict_type='raw',
                indexes=None, driver='GTiff', dtype='float32', nodata=-99999,
                progress=True):
        """Apply prediction of a scikit learn model to a GDAL-supported
        raster dataset

        Args
        ----
        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        file_path : str, optional
            Path to a GeoTiff raster for the classification results
            If not supplied then output is written to a temporary file

        predict_type : str, optional (default='raw')
            'raw' for classification/regression
            'prob' for probabilities

        indexes : List, int, optional
            List of class indices to export

        driver : str, optional. Default is 'GTiff'
            Named of GDAL-supported driver for file export

        dtype : str, optional. Default is 'float32'
            Numpy data type for file export

        nodata : any number, optional. Default is -99999
            Nodata value for file export

        progress : bool, optional. Default is True
            Show tqdm progress bar for prediction

        Returns
        -------
        rasterio.io.DatasetReader with predicted raster"""

        # chose prediction function
        if predict_type == 'raw':
            predfun = self._predfun
        elif predict_type == 'prob':
            predfun = self._probfun

        # determine output count
        if predict_type == 'prob' and isinstance(indexes, int):
            indexes = range(indexes, indexes + 1)

        elif predict_type == 'prob' and indexes is None:
            window = Window(0, 0, self.width, 1)
            img = self.read(masked=True, window=window)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape(
                (n_samples, n_features))
            result = estimator.predict_proba(flat_pixels)
            indexes = np.arange(0, result.shape[1])

        elif predict_type == 'raw':
            indexes = range(1)

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)

        # optionally output to a temporary file
        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name

        with rasterio.open(file_path, 'w', **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            data_gen = (self.read(window=window, masked=True) for window in windows)

            if progress is True:
                for window, arr, pbar in zip(windows, data_gen, tqdm(windows)):
                    result = predfun(arr, estimator)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(dtype), window=window)
            else:
                for window, arr  in zip(windows, data_gen):
                    result = predfun(arr, estimator)
                    result = np.ma.filled(result, fill_value=nodata)
                    dst.write(result[indexes, :, :].astype(dtype), window=window)

        return rasterio.open(file_path)

    @staticmethod
    def _predfun(img, estimator):
        """Prediction function for classification or regression response

        Args
        ----
        img : 3d numpy array of raster data

        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        Returns
        -------
        result_cla : 2d numpy array
            Single band raster as a 2d numpy array containing the
            classification or regression result"""

        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape(
            (n_samples, n_features))

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
        """Class probabilities function

        Args
        ----
        img : 3d numpy array of raster data

        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        Returns
        -------
        result_proba : 3d numpy array
            Multi band raster as a 3d numpy array containing the
            probabilities associated with each class.
            Array is in (class, row, col) order"""

        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape(
            (n_samples, n_features))

        # create mask for NaN values and replace with number
        flat_pixels_mask = flat_pixels.mask.copy()

        # predict probabilities
        result_proba = estimator.predict_proba(flat_pixels)

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result_proba = result_proba.reshape(
            (rows, cols, result_proba.shape[1]))
        flat_pixels_mask = flat_pixels_mask.reshape((rows, cols, n_features))

        # flatten mask into 2d
        mask2d = flat_pixels_mask.any(axis=2)
        mask2d = np.where(mask2d != mask2d.min(), True, False)
        mask2d = np.repeat(mask2d[:, :, np.newaxis],
                           result_proba.shape[2], axis=2)

        # convert proba to masked array using mask2d
        result_proba = np.ma.masked_array(
            result_proba,
            mask=mask2d,
            fill_value=np.nan)

        # reshape band into rasterio format [band, row, col]
        result_proba = result_proba.transpose(2, 0, 1)

        return result_proba

    def calc(self, function, file_path=None, driver='GTiff', dtype='float32',
             nodata=-99999, progress=True):
        """Apply prediction of a scikit learn model to a GDAL-supported
        raster dataset

        Args
        ----
        function : function that takes an numpy array as a single argument

        file_path : str, optional
            Path to a GeoTiff raster for the classification results
            If not supplied then output is written to a temporary file

        driver : str, optional. Default is 'GTiff'
            Named of GDAL-supported driver for file export

        dtype : str, optional. Default is 'float32'
            Numpy data type for file export

        nodata : any number, optional. Default is -99999
            Nodata value for file export

        progress : bool, optional. Default is True
            Show tqdm progress bar for prediction

        Returns
        -------
        rasterio.io.DatasetReader containing result of function output"""

        # determine output dimensions
        window = Window(0, 0, 1, self.width)
        img = self.read(masked=True, window=window)
        arr = function(img)
        if len(arr.shape) > 2:
            indexes = range(arr.shape[0])
        else:
            indexes = 1

        # optionally output to a temporary file
        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)

        with rasterio.open(file_path, 'w', **meta) as dst:

            # define windows
            windows = [window for ij, window in dst.block_windows()]

            # generator gets raster arrays for each window
            data_gen = (self.read(window=window, masked=True) for window in windows)

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

        return rasterio.open(file_path)

    def crop(self, bounds, file_path=None, driver='GTiff', nodata=-99999):
        """Crops a RasterStack object by the supplied bounds

        Args
        ----
        bounds : tuple
            A tuple containing the bounding box to clip by in the
            form of (xmin, xmax, ymin, ymax)

        file_path : str, optional. Default=None
            File path to save to cropped raster.
            If not supplied then the cropped raster is saved to a
            temporary file

        driver : str, optional. Default is 'GTiff'
            Named of GDAL-supported driver for file export

        nodata : int, float
            Nodata value for cropped dataset

        Returns
        -------
        RasterStack object cropped to new extent"""

        xmin, ymin, xmax, ymax = bounds

        rows, cols = rasterio.transform.rowcol(
            self.transform, xs=(xmin, xmax), ys=(ymin, ymax))

        window = Window(col_off=min(cols),
                        row_off=min(rows),
                        width=max(cols)-min(cols),
                        height=max(rows)-min(rows))

        cropped_arr = self.read(masked=True, window=window)
        meta = self.meta
        aff = self.transform
        meta['width'] = max(cols) - min(cols)
        meta['height'] = max(rows) - min(rows)
        meta['transform'] = Affine(aff.a, aff.b, xmin, aff.d, aff.e, ymin)
        meta['driver'] = driver
        meta['nodata'] = nodata

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(cropped_arr)

        # create new RasterStack object
        new_stack = RasterStack(file_path)
        new_stack.files = new_stack.files
        new_stack.names = self.names

        return new_stack

    def sample(self, size, strata=None, return_array=False, random_state=None):
        """Generates a random sample of according to size, and samples the pixel
        values from a GDAL-supported raster

        Args
        ----
        size : int
            Number of random samples or number of samples per strata
            if strategy='stratified'

        strata : rasterio.io.DatasetReader, optional (default=None)
            To use stratified instead of random sampling, strata can be supplied
            using an open rasterio DatasetReader object

        return_array : bool, default = False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays

        na_rm : bool, default = True
            Optionally remove rows that contain nodata values

        random_state : int
            integer to use within random.seed

        Returns
        -------
        samples: array-like
            Numpy array of extracted raster values, typically 2d

        valid_coordinates: 2d array-like
            2D numpy array of xy coordinates of extracted values"""

        # set the seed
        np.random.seed(seed=random_state)

        if not strata:

            # create np array to store randomly sampled data
            # we are starting with zero initial rows because data will be appended,
            # and number of columns are equal to n_features
            valid_samples = np.zeros((0, self.count))
            valid_coordinates = np.zeros((0, 2))

            # loop until target number of samples is satified
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
                xy = np.transpose(rasterio.transform.xy(self.transform, rows, cols))

                # sample at random point locations
                samples = self.extract_xy(xy)

                # append only non-masked data to each row of X_random
                samples = samples.astype('float32').filled(np.nan)
                invalid_ind = np.isnan(samples).any(axis=1)
                samples = samples[~invalid_ind, :]
                valid_samples = np.append(valid_samples, samples, axis=0)

                xy = xy[~invalid_ind, :]
                valid_coordinates = np.append(
                    valid_coordinates, xy, axis=0)

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
                    msg = 'Sample size is greater than number of pixels in strata {0}'.format(str(ind))
                    msg = os.linesep.join([msg, 'Sampling using replacement'])
                    Warning(msg)

                # random sample
                sample = np.random.uniform(
                    low=0, high=ind.shape[0], size=size).astype('int')
                xy = ind[sample, :]

                selected = np.append(selected, xy, axis=0)

            # convert row, col indices to coordinates
            x, y = rasterio.transform.xy(
                self.transform, selected[:, 0], selected[:, 1])
            valid_coordinates = np.column_stack((x, y))

            # extract data
            valid_samples = self.extract_xy(valid_coordinates)

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            gdf = pd.DataFrame(valid_samples, columns=self.names)
            gdf['geometry'] = list(zip(valid_coordinates[:, 0], valid_coordinates[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)
            return gdf
        else:
            return valid_samples, valid_coordinates


def _predfun(img, estimator):
    """Prediction function for classification or regression response

    Parameters
    ----------
    img : 3d numpy array of raster data

    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    Returns
    -------
    result_cla : 2d numpy array
        Single band raster as a 2d numpy array containing the
        classification or regression result"""

    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape each image block matrix into a 2D matrix
    # first reorder into rows,cols,bands(transpose)
    # then resample into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape(
        (n_samples, n_features))

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


def _probfun(img, estimator):
    """Class probabilities function

    Parameters
    ----------
    img : 3d numpy array of raster data

    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    Returns
    -------
    result_proba : 3d numpy array
        Multi band raster as a 3d numpy array containing the
        probabilities associated with each class.
        Array is in (class, row, col) order"""

    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape each image block matrix into a 2D matrix
    # first reorder into rows,cols,bands(transpose)
    # then resample into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape(
        (n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()

    # predict probabilities
    result_proba = estimator.predict_proba(flat_pixels)

    # reshape class probabilities back to 3D image [iclass, rows, cols]
    result_proba = result_proba.reshape(
        (rows, cols, result_proba.shape[1]))
    flat_pixels_mask = flat_pixels_mask.reshape((rows, cols, n_features))

    # flatten mask into 2d
    mask2d = flat_pixels_mask.any(axis=2)
    mask2d = np.where(mask2d != mask2d.min(), True, False)
    mask2d = np.repeat(mask2d[:, :, np.newaxis],
                       result_proba.shape[2], axis=2)

    # convert proba to masked array using mask2d
    result_proba = np.ma.masked_array(
        result_proba,
        mask=mask2d,
        fill_value=np.nan)

    # reshape band into rasterio format [band, row, col]
    result_proba = result_proba.transpose(2, 0, 1)

    return result_proba


def _maximum_dtype(src):
    """Returns a single dtype that is large enough to store data
    within all raster bands

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Rasterio datasetreader in the opened mode

    Returns
    -------
    dtype : str
        Dtype that is sufficiently large to store all raster
        bands in a single numpy array"""

    if 'complex128' in src.dtypes:
        dtype = 'complex128'
    elif 'complex64' in src.dtypes:
        dtype = 'complex64'
    elif 'complex' in src.dtypes:
        dtype = 'complex'
    elif 'float64' in src.dtypes:
        dtype = 'float64'
    elif 'float32' in src.dtypes:
        dtype = 'float32'
    elif 'int32' in src.dtypes:
        dtype = 'int32'
    elif 'uint32' in src.dtypes:
        dtype = 'uint32'
    elif 'int16' in src.dtypes:
        dtype = 'int16'
    elif 'uint16' in src.dtypes:
        dtype = 'uint16'
    elif 'uint16' in src.dtypes:
        dtype = 'uint16'
    elif 'bool' in src.dtypes:
        dtype = 'bool'

    return dtype


def predict(estimator, dataset, file_path=None, predict_type='raw',
            indexes=None, driver='GTiff', dtype='float32', nodata=-99999):
    """Apply prediction of a scikit learn model to a GDAL-supported
    raster dataset

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    dataset : rasterio.io.DatasetReader
        An opened Rasterio DatasetReader

    file_path : str, optional
        Path to a GeoTiff raster for the classification results
        If not supplied then output is written to a temporary file

    predict_type : str, optional (default='raw')
        'raw' for classification/regression
        'prob' for probabilities

    indexes : List, int, optional
        List of class indices to export

    driver : str, optional. Default is 'GTiff'
        Named of GDAL-supported driver for file export

    dtype : str, optional. Default is 'float32'
        Numpy data type for file export

    nodata : any number, optional. Default is -99999
        Nodata value for file export

    Returns
    -------
    rasterio.io.DatasetReader with predicted raster"""

    src = dataset

    # chose prediction function
    if predict_type == 'raw':
        predfun = _predfun
    elif predict_type == 'prob':
        predfun = _probfun

    # determine output count
    if predict_type == 'prob' and isinstance(indexes, int):
        indexes = range(indexes, indexes+1)

    elif predict_type == 'prob' and indexes is None:
        img = src.read(masked=True, window=(0, 0, 1, src.width))
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape(
            (n_samples, n_features))
        result = estimator.predict_proba(flat_pixels)
        indexes = range(result.shape[0])

    elif predict_type == 'raw':
        indexes = range(1)

    # open output file with updated metadata
    meta = src.meta
    meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)

    # optionally output to a temporary file
    if file_path is None:
        file_path = tempfile.NamedTemporaryFile().name

    with rasterio.open(file_path, 'w', **meta) as dst:

        # define windows
        windows = [window for ij, window in dst.block_windows()]

        # generator gets raster arrays for each window
        # read all bands if single dtype
        if src.dtypes.count(src.dtypes[0]) == len(src.dtypes):
            data_gen = (src.read(window=window, masked=True)
                        for window in windows)

        # else read each band separately
        else:
            def read(src, window):
                dtype = _maximum_dtype(src)
                arr = np.ma.zeros((src.count, window.height, window.width),
                                  dtype=dtype)

                for band in range(src.count):
                    arr[band, :, :] = src.read(
                        band+1, window=window, masked=True)

                return arr

            data_gen = (read(src=src, window=window) for window in windows)

        with tqdm(total=len(windows)) as pbar:
            for window, arr in zip(windows, data_gen):
                result = predfun(arr, estimator)
                result = np.ma.filled(result, fill_value=nodata)
                dst.write(result[indexes, :, :].astype(dtype), window=window)
                pbar.update(1)

    return rasterio.open(file_path)


def calc(dataset, function, file_path=None, driver='GTiff', dtype='float32',
         nodata=-99999):
    """Apply prediction of a scikit learn model to a GDAL-supported
    raster dataset

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        An opened Rasterio DatasetReader

    function : function that takes an numpy array as a single argument

    file_path : str, optional
        Path to a GeoTiff raster for the classification results
        If not supplied then output is written to a temporary file

    driver : str, optional. Default is 'GTiff'
        Named of GDAL-supported driver for file export

    dtype : str, optional. Default is 'float32'
        Numpy data type for file export

    nodata : any number, optional. Default is -99999
        Nodata value for file export

    Returns
    -------
    rasterio.io.DatasetReader containing result of function output"""

    src = dataset

    # determine output dimensions
    img = src.read(masked=True, window=(0, 0, 1, src.width))
    arr = function(img)
    if len(arr.shape) > 2:
        indexes = range(arr.shape[0])
    else:
        indexes = 1

    # optionally output to a temporary file
    if file_path is None:
        file_path = tempfile.NamedTemporaryFile().name

    # open output file with updated metadata
    meta = src.meta
    meta.update(driver=driver, count=len(indexes), dtype=dtype, nodata=nodata)

    with rasterio.open(file_path, 'w', **meta) as dst:

        # define windows
        windows = [window for ij, window in dst.block_windows()]

        # generator gets raster arrays for each window
        # read all bands if single dtype
        if src.dtypes.count(src.dtypes[0]) == len(src.dtypes):
            data_gen = (src.read(window=window, masked=True)
                        for window in windows)

        # else read each band separately
        else:
            def read(src, window):
                dtype = _maximum_dtype(src)
                arr = np.ma.zeros((src.count, window.height, window.width),
                                  dtype=dtype)

                for band in range(src.count):
                    arr[band, :, :] = src.read(
                        band+1, window=window, masked=True)

                return arr

            data_gen = (read(src=src, window=window) for window in windows)

        with tqdm(total=len(windows)) as pbar:

            for window, arr in zip(windows, data_gen):
                result = function(arr)
                result = np.ma.filled(result, fill_value=nodata)
                dst.write(result.astype(dtype), window=window)
                pbar.update(1)

    return rasterio.open(file_path)


def crop(dataset, bounds, file_path=None, driver='GTiff'):
    """Crops a rasterio dataset by the supplied bounds

    dataset : rasterio.io.DatasetReader
        An opened Rasterio DatasetReader

    bounds : tuple
        A tuple containing the bounding box to clip by in the
        form of (xmin, xmax, ymin, ymax)

    file_path : str, optional. Default=None
        File path to save to cropped raster.
        If not supplied then the cropped raster is saved to a
        temporary file

    driver : str, optional. Default is 'GTiff'
        Named of GDAL-supported driver for file export

    Returns
    -------
    rasterio.io.DatasetReader with the cropped raster"""

    src = dataset

    xmin, xmax, ymin, ymax = bounds

    rows, cols = rasterio.transform.rowcol(
        src.transform, xs=(xmin, xmax), ys=(ymin, ymax))

    cropped_arr = src.read(window=Window(col_off=min(cols),
                                         row_off=min(rows),
                                         width=max(cols) - min(cols),
                                         height=max(rows) - min(rows)))

    meta = src.meta
    aff = src.transform
    meta['width'] = max(cols) - min(cols)
    meta['height'] = max(rows) - min(rows)
    meta['transform'] = Affine(aff.a, aff.b, xmin, aff.d, aff.e, ymin)
    meta['driver'] = driver

    if file_path is None:
        file_path = tempfile.NamedTemporaryFile().name

    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(cropped_arr)

    return rasterio.open(file_path)
