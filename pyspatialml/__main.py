import os
import tempfile
from collections import namedtuple
from copy import deepcopy
from functools import partial
from itertools import chain

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Point
from tqdm import tqdm


def from_singleband(fp, bidx=None):

    if bidx and isinstance(bidx, int) is False:
        raise ValueError('Single integer bidx required')

    src = rasterio.open(fp)
    band = rasterio.band(src, bidx)
    layer = RasterLayer(band)
    return layer


def from_multiband(fp):

    if isinstance(fp, str):
        fp = [fp]

    # get band objects from datasets
    bands = []

    for f in fp:
        src = rasterio.open(f)
        for i in range(src.count):
            band = rasterio.band(src, i+1)
            bands.append(RasterLayer(band))

    raster_stack = RasterStack(bands)
    return raster_stack


class BaseRasterMixin:
    def __init__(self, band):
        self.shape = band.shape
        self.crs = band.ds.crs
        self.transform = band.ds.transform
        self.width = band.ds.width
        self.height = band.ds.height

    def reproject(self):
        pass

    def mask(self):
        pass

    def resample(self):
        pass

    def aggregate(self):
        pass

    def calc(self, function, file_path=None, driver='GTiff', dtype='float32',
             nodata=-99999, progress=True):
        """Apply user-supplied function to a RasterLayer or RasterStack object

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
        pyspatialml.RasterLayer object"""

        # determine output dimensions
        window = Window(0, 0, 1, self.width)
        img = self.read(masked=True, window=window)
        arr = function(img)

        if len(arr.shape) > 2:
            indexes = range(arr.shape[0])
        else:
            indexes = 1

        count = len(indexes)

        # optionally output to a temporary file
        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=count, dtype=dtype, nodata=nodata)

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

        if count == 1:
            result = from_singleband(file_path)
        else:
            result = from_multiband(file_path)

        return result

    def crop(self, bounds, file_path=None, driver='GTiff', nodata=-99999):
        """Crops a RasterLayer or RasterStack object by the supplied bounds

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
        RasterLayer or RasterStack object cropped to new extent"""

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
        if isinstance(self, RasterLayer):
            result = from_singleband(file_path)
        else:
            result = from_multiband(file_path)
            result.names = self.names

        return result

    def plot(self):
        pass

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

    def head(self):
        """Show the head (first rows, first columns) or tail (last rows, last columns)
         of the cells of a RasterLayer or RasterStack object"""

        window = Window(col_off=0,
                        row_off=0,
                        width=20,
                        height=10)

        arr = self.read(window=window)

        return arr

    def tail(self):
        """Show the head (first rows, first columns) or tail (last rows, last columns)
         of the cells of a RasterLayer or RasterStack object"""

        window = Window(col_off=self.width-20,
                        row_off=self.height-10,
                        width=20,
                        height=10)

        arr = self.read(window=window)

        return arr

    def to_pandas(self, max_pixels=50000, resampling='nearest'):
        """RasterStack to pandas DataFrame

        Args
        ----
        max_pixels: int, default=50000
            Maximum number of pixels to sample

        resampling : str, default = 'nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'

        Returns
        -------
        df : pandas DataFrame"""

        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels

        # read dataset using decimated reads
        out_shape = (round(self.shape[0] * scaling), round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape, resampling=resampling)

        if isinstance(self, RasterLayer):
            arr = arr[np.newaxis, :, :]

        # x and y grid coordinate arrays
        x_range = np.linspace(start=self.bounds.left, stop=self.bounds.right, num=arr.shape[2])
        y_range = np.linspace(start=self.bounds.top, stop=self.bounds.bottom, num=arr.shape[1])
        xs, ys = np.meshgrid(x_range, y_range)

        arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        arr = arr.transpose()
        df = pd.DataFrame(np.column_stack((xs.flatten(), ys.flatten(), arr)),
                          columns=['x', 'y'] + self.names)

        # set nodata values to nan
        for i, col_name in enumerate(self.names):
            df.loc[df[col_name] == self.nodatavals[i], col_name] = np.nan

        return df


class RasterLayer(BaseRasterMixin):
    def __init__(self, band):
        """Defines a read-only single band object with selected methods"""

        # access inherited methods/attributes overriden by __init__
        super().__init__(band)

        # rasterlayer specific attributes
        self.bidx = band.bidx
        self.dtype = band.dtype
        self.nodata = band.ds.nodata
        self.file = band.ds.files[0]
        self.read = partial(band.ds.read, indexes=band.bidx)
        self.driver = band.ds.meta['driver']
        self.ds = band.ds

    def fill(self):
        pass

    def sieve(self):
        pass

    def clump(self):
        pass

    def focal(self):
        pass


class RasterStack(BaseRasterMixin):
    def __init__(self, layers):

        self.loc = {}          # name-based indexing
        self.iloc = []         # index-based indexing
        self.names = []        # syntactically-valid names of datasets with appended band number
        self.files = []        # files that are linked to as RasterLayer objects
        self.dtypes = []       # dtypes of stacked raster datasets and bands
        self.nodatavals = []   # no data values of stacked raster datasets and bands
        self.count = 0         # number of bands in stacked raster datasets
        self.res = None        # (x, y) resolution of aligned raster datasets
        self.bounds = None     # BoundingBox class (namedtuple) ('left', 'bottom', 'right', 'top')
        self.meta = None       # dict containing 'crs', 'transform', 'width', 'height', 'count', 'dtype'
        self._layers = None     # set proxy for self._files
        self.layers = layers    # call property

    def __getitem__(self, x):
        return getattr(self, x)

    def __del__(self):
        """Deconstructor for RasterLayer class to close files"""
        self.close()

    @property
    def layers(self):
        """Getter method for file names within the RasterStack object"""
        return self._layers

    @layers.setter
    def layers(self, layers):
        """Setter method for the files attribute in the RasterStack object"""

        # if tuple of (bands, labels) for custom names of band attributes
        try:
            layers, labels = layers

        except ValueError:
            labels = None

        # some checks
        if isinstance(layers, RasterLayer):
            layers = [layers]

        if all(isinstance(x, type(layers[0])) for x in layers) is False:
            raise ValueError('Cannot create a RasterStack from a mixture of input types')

        meta = self._check_alignment(layers)
        if meta is False:
            raise ValueError(
                'Raster datasets do not all have the same dimensions or transform')

        # reset existing attributes
        for name in self.names:
            delattr(self, name)
        self.iloc = []
        self.names = []
        self.files = []
        self.dtypes = []
        self.nodatavals = []

        # update global RasterStack attributes with new values
        self.count = len(layers)
        self.width = meta['width']
        self.height = meta['height']
        self.shape = (self.height, self.width)
        self.transform = meta['transform']
        self.res = (abs(meta['transform'].a), abs(meta['transform'].e))
        self.crs = meta['crs']
        bounds = rasterio.transform.array_bounds(self.height, self.width, self.transform)
        BoundingBox = namedtuple('BoundingBox', ['left', 'bottom', 'right', 'top'])
        self.bounds = BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])
        self._layers = layers

        # update attributes per dataset
        for i, layer in enumerate(layers):
            if labels is None:
                valid_name = self._make_name(layer.ds.files[0])
            else:
                valid_name = labels[i]

            self.dtypes.append(layer.dtype)
            self.nodatavals.append(layer.nodata)
            self.files.append(layer.file)

            if layer.ds.count > 1:
                valid_name = '_'.join([valid_name, str(layer.bidx)])

            self.names.append(valid_name)
            self.loc.update({valid_name: layer})
            self.iloc.append(layer)
            setattr(self, valid_name, layer)

        self.meta = dict(crs=self.crs,
                         transform=self.transform,
                         width=self.width,
                         height=self.height,
                         count=self.count,
                         dtype=self._maximum_dtype())

    @staticmethod
    def _check_alignment(layers):
        """Check that a list of rasters are aligned with the same pixel dimensions
        and geotransforms"""

        src_meta = []
        for layer in layers:
            src_meta.append(layer.ds.meta.copy())

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
        if valid_name in self.names:
            valid_name = '_'.join([valid_name, '1'])

        return valid_name

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

    def read(self, masked=False, window=None, out_shape=None, resampling='nearest'):
        """Reads data from the RasterStack object into a numpy array

        Overrides read BaseRasterMixin read method and replaces it with a method that
        reads from multiple RasterLayer objects

        Args
        ----
        masked : bool, optional, default = False
            Read data into a masked array

        window : rasterio.window.Window object, optional
            Tuple of col_off, row_off, width, height of a window of data
            to read

        out_shape : tuple, optional
            Shape of shape of array (rows, cols) to read data into using
            decimated reads

        resampling : str, default = 'nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'

        Returns
        -------
        arr : ndarray
            Raster values in 3d numpy array in [band, row, col] order"""

        dtype = self.meta['dtype']

        resampling_methods = [i.name for i in rasterio.enums.Resampling]
        if resampling not in resampling_methods:
            raise ValueError(
                'Invalid resampling method.' +
                'Resampling method must be one of {0}:'.format(
                    resampling_methods))

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

        # read masked or non-masked data
        if masked is True:
            arr = np.ma.zeros((self.count, height, width), dtype=dtype)
        else:
            arr = np.zeros((self.count, height, width), dtype=dtype)

        # read bands separately into numpy array
        for i, layer in enumerate(self.iloc):
            arr[i, :, :] = layer.read(
                masked=masked,
                window=window,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling[resampling])

        return arr

    def open(self):
        """Open raster datasets contained within the RasterStack object"""
        for layer in self.layers:
            layer.open()
            layer.ds.open()

    def close(self):
        """Deconstructor for RasterLayer class to close files"""
        for layer in self.layers:
            layer.ds.close()

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
        RasterLayer or RasterStack object"""

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

        if len(indexes) == 1:
            output = from_singleband(file_path, 1)
        else:
            output = from_multiband(file_path)
            output.names = ['_'.join(['prob', str(i+1)]) for i in output.count]

        return output

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

    def append(self, other, inplace=False):
        """Setter method to add new RasterLayer objects to the RasterStack

        Args
        ----
        other : RasterStack, RasterLayer, or list of RasterLayer objects

        inplace : bool, default = False
            Modify the RasterStack in place"""

        if isinstance(other, RasterLayer):
            other = [other]

        elif isinstance(other, RasterStack):
            other = other.layers

        if inplace is True:
            self.layers = self.layers + other
        else:
            new_stack = RasterStack(self.layers + other)
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

        # convert single label to list
        if isinstance(labels, (str, int)):
            labels = [labels]

        existing_layers = deepcopy(self.layers)

        # index-based method
        if len([i for i in labels if isinstance(i, int)]) == len(labels):
            existing_layers = [existing_layers[i] for i in range(len(existing_layers)) if i not in labels]

        # label-based method
        elif len([i for i in labels if isinstance(i, str)]) == len(labels):
            existing_layers = [existing_layers[i] for i in range(len(existing_layers)) if self.names[i] not in labels]

        else:
            raise ValueError('Cannot drop layers based on mixture of indexes and labels')

        if inplace is True:
            self.layers = existing_layers
        else:
            new_stack = RasterStack(existing_layers)
            return new_stack

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

    def _extract_by_indices(self, rows, cols):
        """spatial query of RasterStack (by-band)"""

        X = np.ma.zeros((len(rows), self.count))

        for i, layer in enumerate(self.iloc):
            arr = layer.read(masked=True)
            X[:, i] = arr[rows, cols]

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