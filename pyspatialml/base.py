import os
import re
import tempfile
from itertools import chain

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import Point


class BaseRaster(object):
    """
    Raster base class that contains methods that apply both to RasterLayer and
    Raster objects.

    Internally comprises a rasterio.Band object consisting of a named tuple of
    the file path, the band index, the dtype and shape an individual band within
    a raster file-based dataset.
    """

    def __init__(self, band):
        self.shape = band.shape
        self.crs = band.ds.crs
        self.transform = band.ds.transform
        self.width = band.ds.width
        self.height = band.ds.height
        self.bounds = band.ds.bounds  # ('left', 'bottom', 'right', 'top')

    @staticmethod
    def _make_name(name, existing_names=None):
        """
        Converts a filename to a valid class attribute name.

        Parameters
        ----------
        name : str
            File name for convert to a valid class attribute name.

        existing_names : list (opt)
            List of existing names to check that the new name will not
            result in duplicated layer names.

        Returns
        -------
        str
            Syntatically correct name of layer so that it can form a class
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
        Show the head (first rows, first columns) or tail (last rows, last
        columns) of pixels.
        """

        window = Window(col_off=0, row_off=0, width=20, height=10)

        return self.read(window=window)

    def tail(self):
        """
        Show the head (first rows, first columns) or tail (last rows, last
        columns) of pixels.
        """

        window = Window(col_off=self.width-20,
                        row_off=self.height-10,
                        width=20,
                        height=10)

        return self.read(window=window)
    
    def _stats(self, max_pixels):
        """
        Take a sample of pixels from which to derive per-band statistics.
        """
        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels

        # read dataset using decimated reads
        out_shape = (round(self.shape[0] * scaling),
                     round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape)

        # reshape for summary stats
        if arr.ndim > 2:
            arr = arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2]))
        else:
            arr = arr.flatten()

        return arr
    
    def min(self, max_pixels=10000):
        """
        Minimum value
        
        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.
        
        Returns
        -------
        numpy.float32
        """
        arr = self._stats(max_pixels)
        
        if arr.ndim > 1:
            stats = arr.min(axis=1).data
        else:
            stats = arr.min()
        
        return stats
    
    def max(self, max_pixels=10000):
        """
        Maximum value.
        
        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.
        
        Returns
        -------
        numpy.float32
        """
        arr = self._stats(max_pixels)
        
        if arr.ndim > 1:
            stats = arr.max(axis=1).data
        else:
            stats = arr.max()
        
        return stats
    
    def mean(self, max_pixels=10000):
        """
        Mean value.
        
        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.
        
        Returns
        -------
        numpy.float32
        """
        arr = self._stats(max_pixels)
        
        if arr.ndim > 1:
            stats = arr.mean(axis=1).data
        else:
            stats = arr.mean()
        
        return stats

    def median(self, max_pixels=10000):
        """
        Median value.
        
        Parameters
        ----------
        max_pixels : int
            Number of pixels used to inform statistical estimate.
        
        Returns
        -------
        numpy.float32
        """
        arr = self._stats(max_pixels)
        
        if arr.ndim > 1:
            stats = np.median(arr, axis=1).data
        else:
            stats = np.median(arr)
        
        return stats

    def sample(self, size, strata=None, return_array=False, random_state=None):
        """
        Generates a random sample of according to size, and samples the pixel
        values.

        Parameters
        ----------
        size : int
            Number of random samples or number of samples per strata if
            strategy='stratified'.

        strata : rasterio DatasetReader (opt)
            Whether to use stratified instead of random sampling. Strata can be
            supplied using an open rasterio DatasetReader object.

        return_array : bool (opt). Default is False
            Optionally return extracted data as separate X, y and xy masked
            numpy arrays.

        na_rm : bool (opt). Default is True
            Optionally remove rows that contain nodata values.

        random_state : int (opt)
            integer to use within random.seed.

        Returns
        -------
        tuple
            Two elements:

            numpy.ndarray
                Numpy array of extracted raster values, typically 2d.

            numpy.ndarray
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
                    msg = (('Sample size is greater than number of pixels in '
                            'strata {0}').format(str(ind)))
                    msg = os.linesep.join([msg, 'Sampling using replacement'])
                    Warning(msg)

                # random sample
                sample = np.random.uniform(0, ind.shape[0], size).astype('int')
                xy = ind[sample, :]

                selected = np.append(selected, xy, axis=0)

            # convert row, col indices to coordinates
            x, y = rasterio.transform.xy(
                transform=self.transform,
                rows=selected[:, 0],
                cols=selected[:, 1])
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
        Samples pixel values using an array of xy locations.

        Parameters
        ----------
        xy : 2d array-like
            x and y coordinates from which to sample the raster (n_samples, xy).

        Returns
        -------
        numpy.ndarray
            2d masked array containing sampled raster values (sample, bands)
            at x,y locations.
        """

        # clip coordinates to extent of raster
        extent = self.bounds
        valid_idx = np.where((xy[:, 0] > extent.left) &
                             (xy[:, 0] < extent.right) &
                             (xy[:, 1] > extent.bottom) &
                             (xy[:, 1] < extent.top))[0]
        xy = xy[valid_idx, :]

        dtype = np.find_common_type([np.float32], self.dtypes)
        values = np.ma.zeros((xy.shape[0], self.count), dtype=dtype)
        rows, cols = rasterio.transform.rowcol(
            transform=self.transform, xs=xy[:, 0], ys=xy[:, 1])

        for i, (row, col) in enumerate(zip(rows, cols)):
            window = Window(col_off=col,
                            row_off=row,
                            width=1,
                            height=1)

            values[i, :] = (self.read(masked=True, window=window).
                            reshape((1, self.count)))

        return values

    def extract_vector(self, response, columns=None, return_array=False,
                       duplicates='keep', na_rm=True, low_memory=False):
        """
        Sample a Raster/RasterLayer using a geopandas GeoDataframe containing
        points, lines or polygon features.

        Parameters
        ----------
        response: geopandas.GeoDataFrame
            Containing either point, line or polygon geometries. Overlapping
            geometries will cause the same pixels to be sampled.

        columns : str (opt)
            Column names of attribute to be used the label the extracted data
            Used only if the response feature represents a GeoDataframe.

        return_array : bool (opt). Default is False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays.
        
        duplicates : str (opt). Default is 'keep'
            Method to deal with duplicates points that fall inside the same
            pixel. Available options are ['keep', 'mean', min', 'max'].

        na_rm : bool (opt). Default is True
            Optionally remove rows that contain nodata values if extracted
            values are returned as a GeoDataFrame.

        low_memory : bool (opt). Default is False
            Optionally extract pixel values in using a slower but memory-safe
            method.

        Returns
        -------
        If return_array=False:

            geopandas.GeoDataframe
                Containing extracted data as point geometries

        If return_array=True:

            tuple with three items:

            numpy.ndarray
                Numpy masked array of extracted raster values, typically 2d.

            numpy.ndarray
                1d numpy masked array of labelled sampled.

            numpy.ndarray
                2d numpy masked array of row and column indexes of training
                pixels.
        """
        
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]

        if not columns:
            y = None

        duplicate_methods = ['keep', 'mean', 'min', 'max']
        if duplicates not in duplicate_methods:
            raise ValueError('duplicates must be one of ' +
                             str(duplicate_methods))

        # polygon and line geometries
        if (all(response.geom_type == 'Polygon') or
            all(response.geom_type == 'LineString')):
            
            if len(columns) > 1:
                raise NotImplementedError(
                    'Support for extracting values from multiple columns is '
                    'only supported for point geometries')

            # rasterize
            rows_all, cols_all, y_all = [], [], []

            for _, shape in response.iterrows():
                if not columns:
                    shapes = (shape.geometry, 1)
                else:
                    shapes = (shape.geometry, shape[columns])

                arr = np.zeros((self.height, self.width))
                arr[:] = -99999
                arr = features.rasterize(
                    shapes=(shapes for i in range(1)), fill=-99999, out=arr,
                    transform=self.transform, default_value=1,
                    all_touched=True)

                rows, cols = np.nonzero(arr != -99999)

                if columns:
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
            if columns:
                y = response[columns].values

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
                transform=self.transform,
                xs=xy[:, 0],
                ys=xy[:, 1])

            # deal with duplicate points that fall inside same pixel
            if duplicates != "keep":
                rowcol_df = pd.DataFrame(
                    data=np.column_stack((rows, cols, y)),
                    columns=['row', 'col'] + columns)

                rowcol_df['Duplicated'] = (rowcol_df.loc[:, ['row', 'col']].
                                           duplicated())

                if duplicates == 'mean':
                    rowcol_df = (
                        rowcol_df.
                            groupby(['Duplicated', 'row', 'col'], sort=False).
                            mean().
                            reset_index()
                    )

                elif duplicates == 'min':
                    rowcol_df = (
                        rowcol_df.
                            groupby(['Duplicated', 'row', 'col'], sort=False).
                            min().
                            reset_index()
                    )

                elif duplicates == 'max':
                    rowcol_df = (
                        rowcol_df.
                            groupby(['Duplicated', 'row', 'col'], sort=False).
                            max().
                            reset_index()
                    )

                rows = rowcol_df['row'].values.astype('int')
                cols = rowcol_df['col'].values.astype('int')

                xy = np.stack(
                    rasterio.transform.xy(
                        transform=self.transform,
                        rows=rows,
                        cols=cols),
                    axis=1)

                y = rowcol_df[columns].values

        # spatial query of Raster object (loads each band into memory)
        if low_memory is False:
            X = self._extract_by_indices(rows, cols)

        # samples each point separately (much slower)
        else:
            X = self.extract_xy(xy)

        # apply mask
        mask_2d = X.mask.any(axis=1).repeat(2).reshape((X.shape[0], 2))
        xy = np.ma.masked_array(xy, mask=mask_2d)

        if columns is not None:
            X_mask_columns = X.mask.any(axis=1)
            X_mask_columns = X_mask_columns.repeat(len(columns))
            y = np.ma.masked_array(y, mask=X_mask_columns)

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            if na_rm is True:
                X = np.ma.compress_rows(X)
                xy = np.ma.compress_rows(xy)
                
                if columns is not None:
                    if y.ndim == 1:
                        y = y[:, np.newaxis]
                    y = np.ma.compress_rows(y)
            
            if columns is not None:
                data = np.ma.column_stack((y, X))
                column_names = columns + self.names
                
            else:
                data = X
                column_names = self.names

            gdf = pd.DataFrame(data, columns=column_names)
            df_dtypes = {k: v for k, v in zip(self.names, self.dtypes)}
            gdf = gdf.astype(df_dtypes)
            gdf['geometry'] = list(zip(xy[:, 0], xy[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)

            return gdf
        
        if y is not None:
            y = y.squeeze()
        
        return X, y, xy

    def extract_raster(self, response, value_name='value', return_array=False,
                       na_rm=True):
        """
        Sample a Raster object by an aligned raster of labelled pixels.

        Parameters
        ----------
        response: rasterio DatasetReader
            Single band raster containing labelled pixels as an open
            rasterio DatasetReader object

        return_array : bool (opt). Default is False
            Optionally return extracted data as separate X, y and xy
            masked numpy arrays.

        na_rm : bool (opt). Default is True
            Optionally remove rows that contain nodata values.

        Returns
        -------
        If return_array='False':

            geopandas.GeoDataFrame
                Geodataframe containing extracted data as point features

        If return_array='True:

            tuple with three items:

            numpy.ndarray
                Numpy masked array of extracted raster values, typically 2d.

            numpy.ndarray
                1d numpy masked array of labelled sampled.

            numpy.ndarray
                2d numpy masked array of row and column indexes of training
                pixels.
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
            gdf = pd.DataFrame(
                data=np.ma.column_stack((y, X)),
                columns=column_names)
            gdf['geometry'] = list(zip(xy[:, 0], xy[:, 1]))
            gdf['geometry'] = gdf['geometry'].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=self.crs)

            return gdf

        else:
            return X, y, xy

class TempRasterLayer():
    """Create a NamedTemporaryFile like object on Windows that has a close
    method. Workaround used on Windows which cannot open the file a second time
    """
    def __init__(self):
        self.tfile = tempfile.NamedTemporaryFile().name
        self.name = self.tfile

    def close(self):
        os.unlink(self.tfile)


def _file_path_tempfile(file_path):
    """
    Returns a TemporaryFileWrapper and file path
    if a file_path parameter is None.
    """
    if file_path is None:
        if os.name != 'nt':
            tfile = tempfile.NamedTemporaryFile()
            file_path = tfile.name
        else:
            tfile = TempRasterLayer()
            file_path = tfile.name
    else:
        tfile = None

    return file_path, tfile


def _get_nodata(dtype):
    """
    Get a nodata value based on the minimum value permissible by dtype.
    """
    try:
        nodata = np.iinfo(dtype).min
    except ValueError:
        nodata = np.finfo(dtype).min
        # nodata = -99999

    return nodata
