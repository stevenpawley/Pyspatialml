import os
import tempfile
import re
from collections import namedtuple, OrderedDict
from itertools import chain
from functools import partial
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio import features
from shapely.geometry import Point
from tqdm import tqdm
from collections.abc import Mapping

def stack_from_files(file_path, mode='r'):
    """
    Create a Raster object from a GDAL-supported raster file, or list of files

    Args
    ----
    file_path : str, list
        File path, or list of file paths to GDAL-supported rasters

    mode : str
        File open mode. Mode must be one of 'r', 'r+', or 'w'

    Returns
    -------
    raster : pyspatialml.Raster object
    """

    if isinstance(file_path, str):
        file_path = [file_path]

    if mode not in ['r', 'r+', 'w']:
        raise ValueError("mode must be one of 'r', 'r+', or 'w'")

    # get band objects from datasets
    bands = []

    for f in file_path:
        src = rasterio.open(f, mode=mode)
        for i in range(src.count):
            band = rasterio.band(src, i+1)
            bands.append(RasterLayer(band))

    raster = Raster(bands)
    return raster


def layer_from_file(file_path, bidx=1, mode='r'):
    """
    Simple wrapper to create a RasterLayer object which refers to a single band
    within rasterio.band object
    """

    if mode not in ['r', 'r+', 'w']:
        raise ValueError("mode must be one of 'r', 'r+', or 'w'")

    return RasterLayer(rasterio.band(rasterio.open(file_path, mode), bidx))


class BaseRaster(object):
    """
    Raster base class that contains methods that apply both to RasterLayer and
    Raster objects. Wraps a rasterio.band object, which is a named tuple
    consisting of the file path, the band index, the dtype and shape a
    individual band within a raster dataset
    """

    def __init__(self, band):
        self.shape = band.shape
        self.crs = band.ds.crs
        self.transform = band.ds.transform
        self.width = band.ds.width
        self.height = band.ds.height
        self.bounds = band.ds.bounds  # BoundingBox class (namedtuple) ('left', 'bottom', 'right', 'top')
        self.read = partial(band.ds.read, indexes=band.bidx)

        try:
            self.write = partial(band.ds.write, indexes=band.bidx)
        except AttributeError:
            pass

    def reproject(self):
        raise NotImplementedError

    def mask(self):
        raise NotImplementedError

    def resample(self):
        raise NotImplementedError

    def aggregate(self):
        raise NotImplementedError

    def calc(self, function, file_path=None, driver='GTiff', dtype='float32',
             nodata=-99999, progress=True):
        """
        Apply user-supplied function to a Raster object

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
        pyspatialml.Raster object
        """

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

        return self._newraster(file_path)

    def crop(self, bounds, file_path=None, driver='GTiff', nodata=-99999):
        """
        Crops a Raster object by the supplied bounds

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
        pyspatialml.Raster object cropped to new extent
        """

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

        return self._newraster(file_path, self.names)

    def _newraster(self, file_path, names=None):
        """
        Return a new Raster object

        Args
        ----
        file_path : str
            Path to files to create the new Raster object from
        names : list, optional
            List to name the RasterLayer objects in the stack. If not supplied
            then the names will be generated from the filename

        Returns
        -------
        raster : pyspatialml.Raster object
        """

        raster = stack_from_files(file_path)

        if names is not None:
            rename = {old : new for old, new in zip(raster.names, self.names)}
            raster.rename(rename)

        return raster

    def plot(self):
        raise NotImplementedError

    def sample(self, size, strata=None, return_array=False, random_state=None):
        """
        Generates a random sample of according to size, and samples the pixel
        values from a GDAL-supported raster

        Args
        ----
        size : int
            Number of random samples or number of samples per strata
            if strategy='stratified'

        strata : rasterio.io.DatasetReader, optional (default=None)
            To use stratified instead of random sampling, strata can be
            supplied using an open rasterio DatasetReader object

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
            2D numpy array of xy coordinates of extracted values
        """

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
        """
        Show the head (first rows, first columns) or tail
        (last rows, last columns) of the cells of a Raster object
        """

        window = Window(col_off=0, row_off=0, width=20, height=10)
        arr = self.read(window=window)

        return arr

    def tail(self):
        """
        Show the head (first rows, first columns) or tail
        (last rows, last columns) of the cells of a Raster object
        """

        window = Window(col_off=self.width-20,
                        row_off=self.height-10,
                        width=20,
                        height=10)
        arr = self.read(window=window)

        return arr

    def to_pandas(self, max_pixels=50000, resampling='nearest'):
        """
        Raster to pandas DataFrame

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
        df : pandas DataFrame
        """

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


class RasterLayer(BaseRaster):
    """
    A single-band raster object that wraps selected attributes and methods from
    a rasterio.band object into a simpler class. Inherits attributes and
    methods from RasterBase. Contains methods that are only relevant to a
    single-band raster. A RasterLayer is initiated from an underlying
    rasterio.band object
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
        self.ds = band.ds

    def fill(self):
        raise NotImplementedError

    def sieve(self):
        raise NotImplementedError

    def clump(self):
        raise NotImplementedError

    def focal(self):
        raise NotImplementedError  


class ExtendedDict(Mapping):
    """
    Dict that can return based on multiple keys
    
    Args
    ---
    parent : Raster object to store RasterLayer indexing
        Requires to parent Raster object in order to setattr when
        changes in the dict, reflecting changes in the RasterLayers occur
    """

    def __init__(self, parent, *args, **kw):
        self.parent = parent
        self._dict = OrderedDict(*args, **kw)
    
    def __getitem__(self, keys):
        if isinstance(keys, str):
            return self._dict[keys]
        return [self._dict[i] for i in keys]
    
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


class LinkedList:
    """
    Provides integer-based indexing of a ExtendedDict
    
    Args
    ---
    parent : Raster object to store RasterLayer indexing
        Requires to parent Raster object in order to setattr when
        changes in the dict, reflecting changes in the RasterLayers occur
    """
    def __init__(self, parent, d):
        self._index = d
        self.parent = parent
    
    def __setitem__(self, index, value):
        
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            self._index[key] = value
            setattr(self.parent, key, value)
        
        if isinstance(index, slice):
            index = list(range(index.start, index.stop))
        
        if isinstance(index, (list, tuple)):
            for i, idx in enumerate(index):
                key = list(self._index.keys())[idx]
                self._index[key] = value[i]
                setattr(self.parent, key, value[i])
    
    def __getitem__(self, index):
        key = list(self._index.keys())[index]
        return self._index[key]


class Raster(BaseRaster):
    """
    Flexible class that represents a collection of file-based GDAL-supported
    raster datasets which share a common coordinate reference system and
    geometry. Raster objects encapsulate RasterLayer objects, which represent
    single band rasters that can physically be represented by separate
    single-band raster files, multi-band raster files, or any combination of
    individual bands from multi-band rasters and single-band rasters.
    RasterLayer objects only exist within Raster objects.

    A Raster object should be created using the pyspatialml.stack_from_files()
    function, where a single file, or a list of files is passed as the file_path
    argument.

    Additional RasterLayer objects can be added to an existing Raster object
    using the append() method. Either the path to file(s) or an existing
    RasterLayer from another Raster object can be passed to this method and
    those layers, if they are spatially aligned, will be appended to the Raster
    object. Any RasterLayer can also be removed from a Raster object using the
    drop() method.
    """

    def __init__(self, layers):

        self.loc = ExtendedDict(self) # label-based indexing
        self.iloc = LinkedList(self, self.loc) # integer-based indexing
        self.files = []               # files that are linked to as RasterLayer objects
        self.dtypes = []              # dtypes of stacked raster datasets and bands
        self.nodatavals = []          # no data values of stacked raster datasets and bands
        self.count = 0                # number of bands in stacked raster datasets
        self.res = None               # (x, y) resolution of aligned raster datasets
        self.meta = None              # dict containing 'crs', 'transform', 'width', 'height', 'count', 'dtype'
        
        self.layers = layers          # call property
              
    def __getitem__(self, label):
        """
        Subset the Raster object using a label or list of labels
        
        Args
        ----
        label : str, list
            
        Returns
        -------
        A new Raster object only containing the subset of layers specified
        in the label argument
        """
        
        if isinstance(label, str):
            label = [label]
        
        subset_layers = []
        
        for i in label:
            
            if i in self.names is False:
                raise KeyError('layername not present in Raster object')
            else:
                subset_layers.append(self.loc[i])
            
        subset_raster = Raster(subset_layers)
        subset_raster.rename(
            {old : new for old, new in zip(subset_raster.names, label)})
        
        return subset_raster

    def __setitem__(self, key, value):
        """
        Replace a RasterLayer within the Raster object with a new RasterLayer
        
        Note that this modifies the Raster object in place
        
        Args
        ----
        key : str
            key-based index of layer to be replaced
        
        value : RasterLayer object
            RasterLayer to use for replacement
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
        return(iter(self.loc.items()))

    @staticmethod
    def _check_alignment(layers):
        """
        Check that a list of rasters are aligned with the same pixel dimensions
        and geotransforms
        """

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

    @staticmethod
    def _make_name(name, existing_names):
        """
        Converts a filename to a valid class attribute name

        Args
        ----
        name : str
            File name for convert to a valid class attribute name
        
        existing_names : list
            List of existing names to check that the new name will not
            result in duplicated layer names

        Returns
        -------
        valid_name : str
            Syntatically-correct name of layer so that it can form a class
            instance attribute
        """

        # replace spaces with underscore
        valid_name = os.path.basename(name)
        valid_name = valid_name.split(os.path.extsep)[0]
        valid_name = valid_name.replace(' ', '_')
        
        # ensure that does not start with number
        if valid_name[0].isdigit():
            valid_name = "x" + valid_name
        
        # remove parentheses and brackets
        valid_name = re.sub(r'[\[\]\(\)\{\}\;]','', valid_name)

        # check to see if same name already exists
        if valid_name in existing_names:
            valid_name = '_'.join([valid_name, '1'])

        return valid_name

    @staticmethod
    def _check_names(existing_names, new_names):
        
        # check that other raster does not result in duplicated names
        combined_names = existing_names + new_names
        
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

    def _maximum_dtype(self):
        """
        Returns a single dtype that is large enough to store data
        within all raster bands
        """

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
    
    @property
    def names(self):
        """
        Return the names of the RasterLayers in the Raster object
        """
        return list(self.loc.keys())
    
    @property
    def layers(self):
        return self.loc

    @layers.setter
    def layers(self, layers):
        """
        Setter method for the files attribute in the Raster object
        """

        # some checks
        if isinstance(layers, tuple):
            layers, names = layers
        else:
            names = None
                    
        if isinstance(layers, RasterLayer):
            layers = [layers]

        if all(isinstance(x, type(layers[0])) for x in layers) is False:
            raise ValueError('Cannot create a Raster object from a mixture of input types')

        if names is not None:        
            if len(names) != len(layers):
                raise ValueError ('Number of layer names has to match the number of layers')

        meta = self._check_alignment(layers)
        if meta is False:
            raise ValueError(
                'Raster datasets do not all have the same dimensions or transform')

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
        bounds = rasterio.transform.array_bounds(self.height, self.width, self.transform)
        BoundingBox = namedtuple('BoundingBox', ['left', 'bottom', 'right', 'top'])
        self.bounds = BoundingBox(bounds[0], bounds[1], bounds[2], bounds[3])

        # update attributes per dataset
        for i, layer in enumerate(layers):
            self.dtypes.append(layer.dtype)
            self.nodatavals.append(layer.nodata)
            self.files.append(layer.file)

            if names is not None:
                valid_name = names[i]
            else:
                valid_name = self._make_name(layer.file, self.names)
                if layer.ds.count > 1:
                    valid_name = '_'.join([valid_name, "band" + str(layer.bidx)])
            
            self.loc[valid_name] = layer
            setattr(self, valid_name, self.loc[valid_name])

        self.meta = dict(crs=self.crs,
                         transform=self.transform,
                         width=self.width,
                         height=self.height,
                         count=self.count,
                         dtype=self._maximum_dtype())

    def read(self, masked=False, window=None, out_shape=None,
             resampling='nearest', **kwargs):
        """
        Reads data from the Raster object into a numpy array

        Overrides read BaseRaster class read method and replaces it with a
        method that reads from multiple RasterLayer objects

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

        **kwargs : dict
            Other arguments to pass to rasterio.DatasetReader.read method

        Returns
        -------
        arr : ndarray
            Raster values in 3d numpy array [band, row, col]
        """

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
                resampling=rasterio.enums.Resampling[resampling],
                **kwargs)

        return arr

    def write(self, file_path, driver="GTiff", dtype=None, nodata=None):
        """
        Write the Raster object to a file

        Overrides the write RasterBase class method, which is a partial
        function of the rasterio.DatasetReader.write method

        Args
        ----
        file_path : str
            File path to save the Raster object as a multiband file-based
            raster dataset

        driver : str, default = GTiff
            GDAL compatible driver

        dtype : str, optional
            Optionally specify a data type when saving to file. Otherwise
            a datatype is selected based on the RasterLayers in the stack

        nodata : int, float, optional
            Optionally assign a new nodata value when saving to file. Otherwise
            a nodata value that is appropriate for the dtype is used
        """

        if dtype is None:
            dtype = self.meta['dtype']

        if nodata is None:
            nodata = np.iinfo(dtype).min

        with rasterio.open(file_path, mode='w', driver=driver, nodata=nodata,
                           **self.meta) as dst:

            for i, layer in enumerate(self.iloc):
                arr = layer.read()
                arr[arr == layer.nodata] = nodata

                dst.write(arr.astype(dtype), i+1)

        return self._newraster(file_path, self.names)

    def predict(self, estimator, file_path=None, predict_type='raw',
                indexes=None, driver='GTiff', dtype='float32', nodata=-99999,
                progress=True):
        """
        Apply prediction of a scikit learn model to a pyspatialml.Raster object

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
        Raster object
        """

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

        raster = stack_from_files(file_path)
        if len(indexes) > 1:
            raster.names = ['_'.join(['prob', str(i+1)]) for i in range(raster.count)]

        return raster

    def append(self, other):
        """
        Setter method to add new RasterLayers to a Raster object
        
        Note that this modifies the Raster object in-place

        Args
        ----
        other : Raster object or list of Raster objects
        """
        
        if isinstance(other, Raster):
            other = [other]

        for new_raster in other:
        
            # check that other raster does not result in duplicated names
            combined_names = self.names + new_raster.names
            
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

            # update layers and names
            self.layers = (list(self.loc.values()) + 
                           list(new_raster.loc.values()),
                           combined_names)

    def drop(self, labels):
        """
        Drop individual RasterLayers from a Raster object
        
        Note that this modifies the Raster object in-place

        Args
        ----
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or label,
            or a list of integers or labels
        """

        # convert single label to list
        if isinstance(labels, (str, int)):
            labels = [labels]

        # numerical index based subsetting
        if len([i for i in labels if isinstance(i, int)]) == len(labels):
            
            subset_layers = [v for (i, v) in enumerate(list(self.loc.values())) if i not in labels]
            subset_names = [v for (i, v) in enumerate(self.names) if i not in labels]
            
            
        # str label based subsetting
        elif len([i for i in labels if isinstance(i, str)]) == len(labels):
            
            subset_layers = [v for (i, v) in enumerate(list(self.loc.values())) if self.names[i] not in labels]
            subset_names = [v for (i, v) in enumerate(self.names) if self.names[i] not in labels]

        else:
            raise ValueError('Cannot drop layers based on mixture of indexes and labels')
        
        self.layers = (subset_layers, subset_names)

    def rename(self, names):
        """
        Rename a RasterLayer within the Raster object
        
        Note that this modifies the Raster object in-place

        Args
        ----
        names : dict
            dict of old_name : new_name
        """
        
        for old_name, new_name in names.items():
            self.loc[new_name] = self.loc.pop(old_name)
            
    def extract_xy(self, xy):
        """
        Samples pixel values of a Raster using an array of xy locations

        Args
        ----
        xy : 2d array-like
            x and y coordinates from which to sample the raster (n_samples, xy)

        Returns
        -------
        values : 2d array-like
            Masked array containing sampled raster values (sample, bands)
            at x,y locations
        """

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
        """
        Spatial query of Raster object (by-band)
        """

        X = np.ma.zeros((len(rows), self.count))

        for i, layer in enumerate(self.iloc):
            arr = layer.read(masked=True)
            X[:, i] = arr[rows, cols]

        return X

    def _clip_xy(self, xy, y=None):
        """
        Clip array of xy coordinates to extent of Raster object
        """

        extent = self.bounds
        valid_idx = np.where((xy[:, 0] > extent.left) &
                             (xy[:, 0] < extent.right) &
                             (xy[:, 1] > extent.bottom) &
                             (xy[:, 1] < extent.top))[0]
        xy = xy[valid_idx, :]

        if y is not None:
            y = y[valid_idx]

        return xy, y

    def extract_vector(self, response, field=None, return_array=False,
                       duplicates='keep', na_rm=True, low_memory=False):
        """
        Sample a Raster object by a geopandas GeoDataframe containing points,
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
        
        duplicates : str, default = 'keep'
            Method to deal with duplicates points that fall inside the same
            pixel. Available options are ['keep', 'mean', min', 'max']

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
            Returned only if return_array is True

        y: 1d array like
            Numpy masked array of labelled sampled
            Returned only if return_array is True

        xy: 2d array-like
            Numpy masked array of row and column indexes of training pixels
            Returned only if return_array is True
        """

        if not field:
            y = None
        
        duplicate_methods = ['keep', 'mean', 'min', 'max']
        if duplicates not in duplicate_methods:
            raise ValueError('duplicates must be one of ' + str(duplicate_methods))

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
                arr = features.rasterize(
                    shapes=(shapes for i in range(1)), fill=-99999, out=arr,
                    transform=self.transform, default_value=1,
                    all_touched=all_touched)

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
            
            # deal with duplicate points that fall inside same pixel
            if duplicates != "keep":
                rowcol_df = pd.DataFrame(
                    np.column_stack((rows, cols, y)),
                    columns=['row', 'col'] + field)
                rowcol_df['Duplicated'] = rowcol_df.loc[:, ['row', 'col']].duplicated()
        
                if duplicates == 'mean':
                    rowcol_df = rowcol_df.groupby(by=['Duplicated', 'row', 'col'], sort=False).mean().reset_index()
                elif duplicates == 'min':
                    rowcol_df = rowcol_df.groupby(by=['Duplicated', 'row', 'col'], sort=False).min().reset_index()
                elif duplicates == 'max':
                    rowcol_df = rowcol_df.groupby(by=['Duplicated', 'row', 'col'], sort=False).max().reset_index()
        
                rows, cols = rowcol_df['row'].values, rowcol_df['col'].values
                y = rowcol_df[field].values

        # spatial query of Raster object (by-band)
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
            X = np.ma.getdata(X)
            if field:
                y = np.ma.masked_array(data=y, mask=mask)
                y = np.ma.getdata(y)
            xy = np.ma.getdata(xy)

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            if field is not None:
                data = np.ma.column_stack((y, X))
                column_names = [field] + self.names
            else:
                data = X
                column_names = self.names

            gdf = pd.DataFrame(data, columns=column_names)
            gdf['geometry'] = list(zip(xy[:, 0], xy[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)
            return gdf
        else:
            return X, y, xy

    def extract_raster(self, response, value_name='value', return_array=False,
                       na_rm=True):
        """
        Sample a Raster object by an aligned raster of labelled pixels

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
            Numpy masked array of row and column indexes of training pixels
        """

        # open response raster and get labelled pixel indices and values
        arr = response.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)
        xy = np.transpose(rasterio.transform.xy(response.transform, rows, cols))
        y = arr.data[rows, cols]

        # extract Raster object values at row, col indices
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
