from __future__ import print_function

import math
import os
import re
import tempfile
from collections import Counter
from collections import namedtuple
from functools import partial
from itertools import chain
from copy import deepcopy

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio import features
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject
from scipy import ndimage
from shapely.geometry import Point
from tqdm import tqdm

from .indexing import ExtendedDict, LinkedList


class BaseRaster(object):
    """
    Raster base class that contains methods that apply both to RasterLayer and
    Raster objects

    The methods that are encapsulated within BaseRaster comprise those that
    return some non-raster type of object, i.e. a geopandas.GeoDataFrame
    representing a random sample of cells.

    Wraps a rasterio.Band object consisting of a named tuple of the file path,
    the band index, the dtype and shape an individual band within a raster
    file-based dataset
    """

    def __init__(self, band):
        self.shape = band.shape
        self.crs = band.ds.crs
        self.transform = band.ds.transform
        self.width = band.ds.width
        self.height = band.ds.height
        self.bounds = band.ds.bounds  # namedtuple('left', 'bottom', 'right', 'top')
        self.read = partial(band.ds.read, indexes=band.bidx)

    @staticmethod
    def _make_name(name, existing_names=None):
        """
        Converts a filename to a valid class attribute name

        Parameters
        ----------
        name : str
            File name for convert to a valid class attribute name

        existing_names : list
            List of existing names to check that the new name will not
            result in duplicated layer names

        Returns
        -------
        str
            Syntatically-correct name of layer so that it can form a class
            instance attribute
        """

        # replace spaces and hyphens with underscore
        valid_name = os.path.basename(name)
        valid_name = valid_name.split(os.path.extsep)[0]
        valid_name = valid_name.replace(' ', '_')
        valid_name = valid_name.replace('-', '_')

        # ensure that does not start with number
        if valid_name[0].isdigit():
            valid_name = "x" + valid_name

        # remove parentheses and brackets
        valid_name = re.sub(r'[\[\]\(\)\{\}\;]', '', valid_name)

        # remove occurrences of multiple underscores
        valid_name = re.sub(r'_+', '_', valid_name)

        # check to see if same name already exists
        if existing_names is not None:
            if valid_name in existing_names:
                valid_name = '_'.join([valid_name, '1'])

        return valid_name

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

    def sample(self, size, strata=None, return_array=False, random_state=None):
        """
        Generates a random sample of according to size, and samples the pixel
        values

        Parameters
        ----------
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
        numpy.ndarray
            Numpy array of extracted raster values, typically 2d

        numpy.ndarray
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
                xy = np.transpose(rasterio.transform.xy(
                    self.transform, rows, cols))

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
                    msg = 'Sample size is greater than number of pixels in strata {0}'.format(
                        str(ind))
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
            gdf['geometry'] = list(
                zip(valid_coordinates[:, 0], valid_coordinates[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)
            return gdf
        else:
            return valid_samples, valid_coordinates

    def extract_xy(self, xy):
        """
        Samples pixel values of a Raster using an array of xy locations

        Parameters
        ----------
        xy : 2d array-like
            x and y coordinates from which to sample the raster (n_samples, xy)

        Returns
        -------
        numpy.ndarray
            2d masked array containing sampled raster values (sample, bands)
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
            values[i, :] = self.read(
                masked=True, window=window).reshape((1, self.count))

        return values

    def _extract_by_indices(self, rows, cols):
        """
        Spatial query of Raster object (by-band)

        Parameters
        ----------
        rows : ndarray
            1d numpy array of row indices

        cols : ndarray
            1d numpy array of column induces

        Returns
        -------
        numpy.ndarray
            2d numpy array of extracted training data in [sample, feature] shape
        """

        X = np.ma.zeros((len(rows), self.count))

        if isinstance(self, Raster):
            for i, layer in enumerate(self.iloc):
                arr = layer.read(masked=True)
                X[:, i] = arr[rows, cols]
        else:
            arr = self.read(masked=True)
            X[:, 0] = arr[rows, cols]

        return X

    def extract_vector(self, response, field=None, return_array=False,
                       duplicates='keep', na_rm=True, low_memory=False):
        """
        Sample a Raster object by a geopandas GeoDataframe containing points,
        lines or polygon features

        TODO
        ----
        Allow multiple 'fields' to be extracted simultaneously

        Parameters
        ----------
        response: geopandas.GeoDataFrame
            Containing either point, line or polygon geometries. Overlapping
            geometries will cause the same pixels to be sampled.

        field : str, optional
            Field name of attribute to be used the label the extracted data
            Used only if the response feature represents a GeoDataframe

        return_array : bool, default=False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays
        
        duplicates : str, default='keep'
            Method to deal with duplicates points that fall inside the same
            pixel. Available options are ['keep', 'mean', min', 'max']

        na_rm : bool, default=True
            Optionally remove rows that contain nodata values if extracted
            values are returned as a geopandas.GeoDataFrame

        low_memory : bool, default=False
            Optionally extract pixel values in using a slower but memory-safe
            method

        Returns
        -------
        Either:

        geopandas.GeoDataframe
            Containing extracted data as point geometries

        Or if return_array=True:

        numpy.ndarray
            Numpy masked array of extracted raster values, typically 2d
            Returned only if return_array is True

        numpy.ndarray
            1d numpy masked array of labelled sampled
            Returned only if return_array is True

        numpy.ndarray
            2d numpy masked array of row and column indexes of training pixels
            Returned only if return_array is True
        """

        if not field:
            y = None

        duplicate_methods = ['keep', 'mean', 'min', 'max']
        if duplicates not in duplicate_methods:
            raise ValueError('duplicates must be one of ' +
                             str(duplicate_methods))

        # polygon and line geometries
        if all(response.geom_type == 'Polygon') or all(response.geom_type == 'LineString'):

            # rasterize
            rows_all, cols_all, y_all = [], [], []

            for _, shape in response.iterrows():
                if not field:
                    shapes = (shape.geometry, 1)
                else:
                    shapes = (shape.geometry, shape[field])

                arr = np.zeros((self.height, self.width))
                arr[:] = -99999
                arr = features.rasterize(
                    shapes=(shapes for i in range(1)), fill=-99999, out=arr,
                    transform=self.transform, default_value=1,
                    all_touched=True)

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
            extent = self.bounds
            valid_idx = np.where((xy[:, 0] > extent.left) &
                                (xy[:, 0] < extent.right) &
                                (xy[:, 1] > extent.bottom) &
                                (xy[:, 1] < extent.top))[0]
            xy = xy[valid_idx, :]

            if y is not None:
                y = y[valid_idx]
            
            # convert to row, col indices
            rows, cols = rasterio.transform.rowcol(
                transform=self.transform, xs=xy[:, 0], ys=xy[:, 1])

            # deal with duplicate points that fall inside same pixel
            if duplicates != "keep":
                rowcol_df = pd.DataFrame(
                    np.column_stack((rows, cols, y)),
                    columns=['row', 'col'] + [field])
                rowcol_df['Duplicated'] = rowcol_df.loc[:, ['row', 'col']].duplicated()

                if duplicates == 'mean':
                    rowcol_df = rowcol_df.groupby(
                        by=['Duplicated', 'row', 'col'], sort=False).mean().reset_index()
                elif duplicates == 'min':
                    rowcol_df = rowcol_df.groupby(
                        by=['Duplicated', 'row', 'col'], sort=False).min().reset_index()
                elif duplicates == 'max':
                    rowcol_df = rowcol_df.groupby(
                        by=['Duplicated', 'row', 'col'], sort=False).max().reset_index()

                rows, cols = rowcol_df['row'].values.astype(
                    'int'), rowcol_df['col'].values.astype('int')
                xy = np.stack(
                    rasterio.transform.xy(
                        transform=self.transform, rows=rows, cols=cols), axis=1)
                y = rowcol_df[field].values

        # spatial query of Raster object (loads each band into memory)
        if low_memory is False:
            X = self._extract_by_indices(rows, cols)

        # samples each point separately (much slower)
        else:
            X = self.extract_xy(xy)

        # apply mask
        mask_2d = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
        xy = np.ma.masked_array(xy, mask=mask_2d)

        if field:
            y = np.ma.masked_array(y, mask=X.mask.any(axis=1))

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            if na_rm is True:
                X = np.ma.compress_rows(X)
                xy = np.ma.compress_rows(xy)
                if field is not None:
                    y = np.ma.compressed(y)

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
        
        return X, y, xy

    def extract_raster(self, response, value_name='value', return_array=False,
                       na_rm=True):
        """
        Sample a Raster object by an aligned raster of labelled pixels

        Parameters
        ----------
        response: rasterio.io.DatasetReader
            Single band raster containing labelled pixels

        return_array : bool, default = False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays

        na_rm : bool, default = True
            Optionally remove rows that contain nodata values

        Returns
        -------
        geopandas GeoDataFrame
            Geodataframe containing extracted data as point features

        numpy.ndarray
            Numpy masked array of extracted raster values, typically 2d

        numpy.ndarray
            1d numpy masked array of labelled sampled

        numpy.ndarray
            2d numpy masked array of row and column indexes of training pixels
        """

        # open response raster and get labelled pixel indices and values
        arr = response.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)
        xy = np.transpose(rasterio.transform.xy(
            response.transform, rows, cols))
        y = arr.data[rows, cols]

        # extract Raster object values at row, col indices
        X = self._extract_by_indices(rows, cols)

        # summarize data and mask nodatavals in X, y, and xy
        mask_2d = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
        y = np.ma.masked_array(y, mask=X.mask.any(axis=1))
        xy = np.ma.masked_array(xy, mask=mask_2d)

        if return_array is False:
            if na_rm is True:
                X = np.ma.compress_rows(X)
                y = np.ma.compressed(y)
                xy = np.ma.compress_rows(xy)

            column_names = [value_name] + self.names
            gdf = pd.DataFrame(np.ma.column_stack(
                (y, X)), columns=column_names)
            gdf['geometry'] = list(zip(xy[:, 0], xy[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)
            return gdf
        else:
            return X, y, xy

class RasterLayer(BaseRaster):
    """
    Represents a single rasterband derived from a single or multiband raster
    dataset

    Simple wrapper around a rasterio.Band object with additional methods. 
    Used because the Rasterio.Band.ds.read method reads all bands from a
    multiband dataset, whereas the RasterLayer.read method only reads 
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
        Calculate euclidean grid distances to non-NaN cells in a RasterLayer

        Parameters
        ----------
        file_path : str, path to save distance raster, optional
            If not specified output is saved to a temporary file

        driver : str, default='GTiff'
            GDAL-supported driver format

        nodata : int, float, default=-99999
            Value to use as the nodata value of the output raster

        Returns
        -------
        pyspatialml.RasterLayer object
            Grid distance raster as a pyspatialml.RasterLayer object
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
        return RasterLayer(rasterio.band(src, 1))

    def plot(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        arr = self.read(masked=True)
        ax.imshow(arr, extent=rasterio.plot.plotting_extent(
            self.ds), cmap=self.cmap)
        
        return fig, ax


class Raster(BaseRaster):
    """
    Flexible class that represents a collection of file-based GDAL-supported
    raster datasets which share a common coordinate reference system and
    geometry

    Raster objects encapsulate RasterLayer objects, which represent
    single band rasters that can physically be represented by separate
    single-band raster files, multi-band raster files, or any combination of
    individual bands from multi-band rasters and single-band rasters

    Methods defined in the Raster class are those that usually would be applied
    to multiple rasters, and always return a new Raster object
    """

    def __init__(self, src=None, arr=None, crs=None, transform=None,
                 nodata=-99999, mode='r', file_path=None):
        """
        Initiate a new Raster object

        Parameters
        ----------
        src : file path, RasterLayer or list of either
            Inititate a Raster object from file paths to GDAL-supported
            raster datasets, pyspatialml.RasterLayer objects,
            rasterio.io.datasetreader, or rasterio.band objects

        arr : ndarray, optional
            Initiated a Raster object from a numpy.ndarray. Additional arguments
            crs and transform should also be provided to supply spatial coordinate
            information

        crs : rasterio.crs.CRS object, optional
            CRS object containing projection information for the associated
            arr parameter

        transform : affine.Affine object
            Affine object containing transform information for the associated
            arr parameter

        file_path : str
            Path to save new Raster object if created from arr

        Returns
        -------
        pyspatialml.Raster
        """

        self.loc = ExtendedDict(self)
        self.iloc = LinkedList(self, self.loc)
        self.files = []
        self.dtypes = []
        self.nodatavals = []
        self.count = 0
        self.res = None
        self.meta = None
        self.tempfiles = []

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
                file_path, 'w', driver='GTiff', height=arr.shape[1],
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
                'Cannot initiated a Raster from a list of different type objects')

        # call property with a list of rasterio.band objects
        self._layers = src_layers

    def __getitem__(self, label):
        """
        Subset the Raster object using a label or list of labels
        
        Parameters
        ----------
        label : str, list
            
        Returns
        -------
        pyspatialml.Raster
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

        return subset_raster

    def __setitem__(self, key, value):
        """
        Replace a RasterLayer within the Raster object with a new RasterLayer
        
        Note that this modifies the Raster object in place
        
        Parameters
        ----------
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
    
    def __del__(self):
        self.close()
    
    def close(self):
        for layer in self.iloc:
            layer.ds.close()
        
        for f in self.tempfiles:
            os.remove(f)

    @staticmethod
    def _check_alignment(layers):
        """
        Check that a list of rasters are aligned with the same pixel dimensions
        and geotransforms

        Parameters
        ----------
        layers : list
            List of RasterLayer objects

        Returns
        -------
        dict, False
            dict of metadata if all layers are spatially aligned,
            otherwise returns False
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

    @staticmethod
    def _fix_names(combined_names):

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
    def names(self):
        """
        Return the names of the RasterLayers in the Raster object
        """
        return list(self.loc.keys())

    @property
    def _layers(self):
        """
        Getter method

        Returns
        -------
        pyspatialml.indexing.ExtendedDict
        """
        return self.loc

    @_layers.setter
    def _layers(self, layers):
        """
        Setter method for the files attribute in the Raster object

        Parameters
        ----------
        layers : RasterLayer, list of RasterLayer objects, or tuple with
            (RasterLayers, names_of_layers)
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
                         dtype=self._maximum_dtype())

    def read(self, masked=False, window=None, out_shape=None,
             resampling='nearest', **kwargs):
        """
        Reads data from the Raster object into a numpy array

        Overrides read BaseRaster class read method and replaces it with a
        method that reads from multiple RasterLayer objects

        Parameters
        ----------
        masked : bool, optional, default=False
            Read data into a masked array

        window : rasterio.window.Window object, optional
            Tuple of col_off, row_off, width, height of a window of data
            to read

        out_shape : tuple, optional
            Shape of shape of array (rows, cols) to read data into using
            decimated reads

        resampling : str, default='nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'

        **kwargs : dict
            Other arguments to pass to rasterio.DatasetReader.read method

        Returns
        -------
        ndarray
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

        Parameters
        ----------
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

        Returns
        -------
        pyspatialml.Raster
            New Raster object from saved file
        """

        if dtype is None:
            dtype = self.meta['dtype']

        if nodata is None:
            try:
                nodata = np.iinfo(dtype).min
            except ValueError:
                nodata = np.finfo(dtype).min

        meta = self.meta
        meta['driver'] = driver
        meta['nodata'] = nodata

        with rasterio.open(file_path, mode='w', **self.meta) as dst:

            for i, layer in enumerate(self.iloc):
                arr = layer.read()
                arr[arr == layer.nodata] = nodata

                dst.write(arr.astype(dtype), i+1)

        return self._newraster(file_path, self.names)

    def to_pandas(self, max_pixels=50000, resampling='nearest'):
        """
        Raster to pandas DataFrame

        Parameters
        ----------
        max_pixels: int, default=50000
            Maximum number of pixels to sample

        resampling : str, default = 'nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'

        Returns
        -------
        pandas.DataFrame
        """

        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels

        # read dataset using decimated reads
        out_shape = (round(self.shape[0] * scaling),
                     round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape,
                        resampling=resampling)
        
        # x and y grid coordinate arrays
        x_range = np.linspace(start=self.bounds.left,
                              stop=self.bounds.right, num=arr.shape[2])
        y_range = np.linspace(start=self.bounds.top,
                              stop=self.bounds.bottom, num=arr.shape[1])
        xs, ys = np.meshgrid(x_range, y_range)

        arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        arr = arr.transpose()
        df = pd.DataFrame(np.column_stack((xs.flatten(), ys.flatten(), arr)),
                          columns=['x', 'y'] + self.names)

        # set nodata values to nan
        for i, col_name in enumerate(self.names):
            df.loc[df[col_name] == self.nodatavals[i], col_name] = np.nan

        return df

    def predict_proba(self, estimator, file_path=None, indexes=None,
                      driver='GTiff', dtype='float32', nodata=-99999,
                      progress=True):
        """
        Apply prediction of a scikit learn model to a pyspatialml.Raster object

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        file_path : str, optional
            Path to a GeoTiff raster for the classification results
            If not supplied then output is written to a temporary file

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
        pyspatialml.Raster object
        """

        tempfiles = None

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

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=len(
            indexes), dtype=dtype, nodata=nodata)

        # optionally output to a temporary file
        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name
            tempfiles = file_path

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
        
        new_raster = self._newraster(file_path, names)
        new_raster.tempfiles.append(tempfiles)

        return new_raster

    def predict(self, estimator, file_path=None, driver='GTiff',
                dtype='float32', nodata=-99999, progress=True,
                block_shape=None):
        """
        Apply prediction of a scikit learn model to a pyspatialml.Raster object

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data

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

        block_shape : tuple, optional
            Optionally supply a block shape to read and write the
            raster data. Specified as a tuple (height, width)

        Returns
        -------
        pyspatialml.Raster object
        """

        tempfiles = None

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

        # open output file with updated metadata
        meta = self.meta
        meta.update(driver=driver, count=len(
            indexes), dtype=dtype, nodata=nodata)

        # optionally output to a temporary file
        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name
            tempfiles = file_path

        with rasterio.open(file_path, 'w', **meta) as dst:

            # define windows
            if block_shape is None:
                windows = [window for ij, window in dst.block_windows()]
            else:
                windows = [window for window in self.block_shapes(*block_shape)]

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
        
        new_raster = self._newraster(file_path, names)
        new_raster.tempfiles.append(tempfiles)
        
        return new_raster

    def _predfun(self, img, estimator):
        """
        Prediction function for classification or regression response

        Parameters
        ----
        img : 3d numpy array of raster data

        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        Returns
        -------
        numpy.ndarray
            2d numpy array representing a single band raster containing the
            classification or regression result
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
        Class probabilities function

        Parameters
        ----------
        img : ndarray
            3d numpy array of raster data [band, row, col]

        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        Returns
        -------
        numpy.ndarray
            Multi band raster as a 3d numpy array containing the
            probabilities associated with each class.
            Array is in (class, row, col) order
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
        Multi output prediction

        Parameters
        ----------
        img : ndarray
            3d numpy array of raster data [band, row, col]

        estimator : estimator object implementing 'fit'
            The object to use to fit the data

        Returns
        -------
        narray
            Multi band raster as a 3d numpy array containing the
            probabilities associated with each class.
            Array is in (class, row, col) order
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
        Setter method to add new RasterLayers to a Raster object
        
        Note that this modifies the Raster object in-place by default

        Parameters
        ----------
        other : Raster object, or list of Raster objects
        
        in_place : bool, default=True
            Change the Raster object in-place or 
            leave original and return a new Raster object
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
        Drop individual RasterLayers from a Raster object
        
        Note that this modifies the Raster object in-place by default
        
        Parameters
        ---------
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or label,
            or a list of integers or labels
        
        in_place : bool, default=True
            Change the Raster object in-place or 
            leave original and return a new Raster object
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
        Rename a RasterLayer within the Raster object
        
        Note that by default this modifies the Raster object in-place

        Parameters
        ----------
        names : dict
            dict of old_name : new_name
        
        in_place : bool, default=True
            Change names of the Raster object in-place or 
            leave original and return a new Raster object
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
                
            return(new_raster)


    def plot(self, out_shape=(100, 100), label_fontsize=8, title_fontsize=8,
             names=None, **kwargs):
        """
        Plotting of a Raster object

        Parameters
        ----------
        out_shape : tuple
            Number of rows, cols to read from the raster datasets for plotting

        label_fontsize : int, float, default = 8
            Size in pts of labels

        title_fontsize : int, float, default = 8
            Size in pts of titles

        names : list, optional
            Optionally supply a list of names for each RasterLayer to override the
            default layer names for the titles

        **kwargs : dict
            Additional arguments to pass when creating the figure object

        Returns
        -------
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

    def _newraster(self, file_path, names=None, tempfiles=None):
        """
        Return a new Raster object

        Parameters
        ----------
        file_path : str
            Path to files to create the new Raster object from

        names : list, optional
            List to name the RasterLayer objects in the stack. If not supplied
            then the names will be generated from the filename
        
        tempfiles : list, optional
            List of file paths for RasterLayer objects that are stored as tempfiles
            These will be deleted on destruction of the Raster object

        Returns
        -------
        pyspatialml.Raster object
        """

        if isinstance(file_path, str):
            file_path = [file_path]

        raster = Raster(file_path)

        if names is not None:
            rename = {old: new for old, new in zip(raster.names, names)}
            raster.rename(rename)
        
        for old_layer, new_layer in zip(self.iloc, raster.iloc):
            raster.new_layer.cmap = old_layer.cmap
        
        raster.tempfiles = self.tempfiles + tempfiles

        return raster

    def mask(self, shapes=None, invert=False, crop=False, filled=False,
             pad=False, file_path=None, driver='GTiff', nodata=-99999):
        """
        Mask a Raster object based on the outline of shapes in a
        geopandas.GeoDataFrame
        """

        tempfiles = []
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

        masked_ndarrays = np.ma.filled(masked_ndarrays, fill_value=nodata)

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name
            tempfiles.append(file_path)

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(masked_ndarrays)

        new_raster = self._newraster(file_path, self.names, tempfiles)
        
        return new_raster

    def intersect(self, file_path=None, driver='GTiff', nodata=-99999):
        """
        Perform a intersect operation on the Raster object

        Computes the geometric intersection of the RasterLayers with the
        Raster object. This will cause nodata values in any of the rasters
        to be propagated through all of the output rasters

        Returns
        -------
        pyspatial.Raster object
        """
        tempfiles = []

        arr = self.read(masked=True)
        mask_2d = arr.mask.any(axis=0)

        # repeat mask for n_bands
        mask_3d = np.repeat(a=mask_2d[np.newaxis, :, :],
                           repeats=self.count, axis=0)

        intersected_arr = np.ma.masked_array(arr, mask=mask_3d, fill_value=nodata)
        intersected_arr = np.ma.filled(intersected_arr, fill_value=nodata)

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name
            tempfiles.append(file_path)

        meta = self.meta
        meta['driver'] = driver
        meta['nodata'] = nodata

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(intersected_arr)

        new_raster = self._newraster(file_path, self.names, tempfiles)
        
        return new_raster

    def crop(self, bounds, file_path=None, driver='GTiff', nodata=-99999):
        """
        Crops a Raster object by the supplied bounds

        Parameters
        ----------
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
        pyspatialml.Raster object
            Cropped to new extent
        """

        tempfiles = []

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
            tempfiles.append(file_path)

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(cropped_arr)

        new_raster = self._newraster(file_path, self.names, tempfiles)
        
        return new_raster

    def to_crs(self, crs, resampling='nearest', file_path=None,
               driver='GTiff', nodata=-99999, n_jobs=1,
               warp_mem_lim=0, progress=False):
        """
        Reprojects a Raster object to a different crs

        Parameters
        ----------
        crs : rasterio.transform.CRS object, or dict
            Example: CRS({'init': 'EPSG:4326'})

        resampling : str
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

        file_path : str, optional
            Optional path to save reprojected Raster object. If not
            specified then a tempfile is used

        driver : str, default='GTiff'
            GDAL driver

        nodata : int, float
            No data to use for reprojected layers in reprojected Raster

        n_jobs : int, default=1
            The number of warp worker threads

        warp_mem_lim : int, default=0
            The warp operation memory limit in MB. Larger values allow the
            warp operation to be carried out in fewer chunks. The amount of
            memory required to warp a 3-band uint8 2000 row x 2000 col
            raster to a destination of the same size is approximately
            56 MB. The default (0) means 64 MB with GDAL 2.2.

        progress : bool, default=False
            Optionally show progress of transform operations
        """

        tempfiles = []

        resampling_methods = [i.name for i in rasterio.enums.Resampling]
        if resampling not in resampling_methods:
            raise ValueError(
                'Invalid resampling method.' +
                'Resampling method must be one of {0}:'.format(
                    resampling_methods))

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name
            tempfiles.append(file_path)

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

        new_raster = self._newraster(file_path, self.names, tempfiles)
        
        return new_raster

    def aggregate(self, out_shape, resampling='nearest', file_path=None, 
                  driver='GTiff', nodata=-99999):
        """
        Aggregates a raster to (usually) a coarser grid cell size

        Parameters
        ----------
        out_shape : tuple
            New shape in (rows, cols)

        resampling : str, default='nearest'
            Resampling method to use when applying decimated reads when
            out_shape is specified. Supported methods are: 'average',
            'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos',
            'max', 'med', 'min', 'mode', 'q1', 'q3'

        file_path : str, optional. Default=None
            File path to save to cropped raster.
            If not supplied then the aggregated raster is saved to a
            temporary file

        driver : str, optional. Default is 'GTiff'
            Named of GDAL-supported driver for file export

        nodata : int, float
            Nodata value for new dataset

        Returns
        -------
        pyspatialml.Raster object
        """

        tempfiles = []

        rows, cols = out_shape

        arr = self.read(masked=True, out_shape=out_shape,
                        resampling=resampling)

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile().name
            tempfiles.append(file_path)

        meta = self.meta
        meta['driver'] = driver
        meta['nodata'] = nodata
        meta['height'] = rows
        meta['width'] = cols

        bnd = self.bounds
        meta['transform'] = rasterio.transform.from_bounds(
            west=bnd.left, south=bnd.bottom, east=bnd.right, north=bnd.top,
            width=cols, height=rows)

        with rasterio.open(file_path, 'w', **meta) as dst:
            dst.write(arr)
            
        new_raster = self._newraster(file_path, self.names, tempfiles)

        return new_raster

    def calc(self, function, file_path=None, driver='GTiff', dtype='float32',
             nodata=-99999, progress=False):
        """
        Apply user-supplied function to a Raster object

        Parameters
        ----------
        function : function
            Function that takes an numpy array as a single argument

        file_path : str, optional
            Path to a GeoTiff raster for the classification results
            If not supplied then output is written to a temporary file

        driver : str, optional. Default is 'GTiff'
            Named of GDAL-supported driver for file export

        dtype : str, optional. Default is 'float32'
            Numpy data type for file export

        nodata : any number, optional. Default is -99999
            Nodata value for file export

        progress : bool, optional. Default=False
            Show tqdm progress bar for prediction

        Returns
        -------
        pyspatialml.Raster object
        """
        
        tempfiles = []

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
            tempfiles.append(file_path)

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
        
        return new_raster

    def block_shapes(self, rows, cols):
        """
        Generator for windows for optimal reading and writing based on the raster format
        Windows are returns as a tuple with xoff, yoff, width, height

        Parameters
        ----------
        rows : int
            Height of window in rows

        cols : int
            Width of window in columns
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
