import multiprocessing
import os
import re
from abc import ABC, abstractmethod
from itertools import chain

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window
from rasterio.sample import sample_gen

from shapely.geometry import Point
from tqdm import tqdm


class BaseRaster(ABC):
    """Base class for Raster and RasterLayer objects
    """

    def __init__(self, band):
        self.shape = band.shape
        self.crs = band.ds.crs
        self.transform = band.ds.transform
        self.width = band.ds.width
        self.height = band.ds.height
        self.bounds = band.ds.bounds

    @abstractmethod
    def read(self, **kwargs):
        """Read method to be defined by subclass"""
        raise NotImplementedError()

    @abstractmethod
    def write(self):
        """Write method to be defined by subclass"""
        raise NotImplementedError()

    @abstractmethod
    def plot(self):
        """Plot method to be defined by subclass"""
        raise NotImplementedError()

    @staticmethod
    def _make_name(name, existing_names=None):
        """Converts a filename to a valid class attribute name.

        Parameters
        ----------
        name : str
            File name for convert to a valid class attribute name.

        existing_names : list (opt)
            List of existing names to check that the new name will not result in 
            duplicated layer names.

        Returns
        -------
        str
            Syntatically correct name of layer so that it can form a class instance
            attribute.
        """

        # replace spaces and hyphens with underscore
        valid_name = os.path.basename(name)
        valid_name = valid_name.split(os.path.extsep)[0]
        valid_name = valid_name.replace(" ", "_")
        valid_name = valid_name.replace("-", "_")

        # ensure that does not start with number
        if valid_name[0].isdigit():
            valid_name = "x" + valid_name

        # remove parentheses and brackets
        valid_name = re.sub(r"[\[\]\(\)\{\}\;]", "", valid_name)

        # remove occurrences of multiple underscores
        valid_name = re.sub(r"_+", "_", valid_name)

        # check to see if same name already exists
        if existing_names is not None:
            if valid_name in existing_names:
                valid_name = "_".join([valid_name, "1"])

        return valid_name

    def _stats(self, max_pixels):
        """Take a sample of pixels from which to derive per-band statistics.
        """
        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels

        # read dataset using decimated reads
        out_shape = (round(self.shape[0] * scaling), round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape)

        # reshape for summary stats
        if arr.ndim > 2:
            arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        else:
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

        if arr.ndim > 1:
            stats = arr.min(axis=1).data
        else:
            stats = arr.min()

        return stats

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

        if arr.ndim > 1:
            stats = arr.max(axis=1).data
        else:
            stats = arr.max()

        return stats

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

        if arr.ndim > 1:
            stats = arr.mean(axis=1).data
        else:
            stats = arr.mean()

        return stats

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

        if arr.ndim > 1:
            stats = np.median(arr, axis=1).data
        else:
            stats = np.median(arr)

        return stats

    def sample(self, size, strata=None, return_array=False, random_state=None):
        """Generates a random sample of according to size, and samples the pixel
        values.

        Parameters
        ----------
        size : int
            Number of random samples or number of samples per strata if
            strategy='stratified'.

        strata : rasterio DatasetReader (opt)
            Whether to use stratified instead of random sampling. Strata can be
            supplied using an open rasterio DatasetReader object.

        return_array : bool (opt), default=False
            Optionally return extracted data as separate X, y and xy masked numpy
            arrays.

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
                xy = np.transpose(rasterio.transform.xy(self.transform, rows, cols))

                # sample at random point locations
                samples = self.extract_xy(xy)

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
                    msg = (
                        "Sample size is greater than number of pixels in " "strata {0}"
                    ).format(str(ind))
                    msg = os.linesep.join([msg, "Sampling using replacement"])
                    Warning(msg)

                # random sample
                sample = np.random.uniform(0, ind.shape[0], size).astype("int")
                xy = ind[sample, :]

                selected = np.append(selected, xy, axis=0)

            # convert row, col indices to coordinates
            x, y = rasterio.transform.xy(
                transform=self.transform, rows=selected[:, 0], cols=selected[:, 1]
            )
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
            x and y coordinates from which to sample the raster (n_samples, xys).
        
        return_array : bool (opt), default=False
            By default the extracted pixel values are returned as a 
            geopandas.GeoDataFrame. If `return_array=True` then the extracted pixel
            values are returned as a tuple of numpy.ndarrays. 

        progress : bool (opt), default=False
            Show a progress bar for extraction.

        Returns
        -------
        geopandas.GeoDataframe
            Containing extracted data as point geometries if `return_array=False`.

        numpy.ndarray
            2d masked array containing sampled raster values (sample, bands) at the 
            x,y locations.
        """

        # extract pixel values
        dtype = np.find_common_type([np.float32], self.dtypes)
        X = np.ma.zeros((xys.shape[0], self.count), dtype=dtype)
        
        if progress is True:
            disable_tqdm = False
        else:
            disable_tqdm = True

        for i, (layer, pbar) in enumerate(zip(self.iloc, tqdm(self.iloc, total=self.count, disable=disable_tqdm))):
            sampler = sample_gen(dataset=layer.ds, xy=xys, indexes=layer.bidx, masked=True)
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
        """Sample a Raster/RasterLayer using a geopandas GeoDataframe containing
        points, lines or polygon features.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Containing either point, line or polygon geometries. Overlapping geometries
            will cause the same pixels to be sampled.

        return_array : bool (opt), default=False
            By default the extracted pixel values are returned as a 
            geopandas.GeoDataFrame. If `return_array=True` then the extracted pixel
            values are returned as a tuple of numpy.ndarrays. 
        
        progress : bool (opt), default=False
            Show a progress bar for extraction.

        Returns
        -------
        geopandas.GeoDataframe
            Containing extracted data as point geometries (one point per pixel) if 
            `return_array=False`. The resulting GeoDataFrame is indexed using a 
            named pandas.MultiIndex, with `pixel_idx` index referring to the index of each
            pixel that was sampled, and the `geometry_idx` index referring to the index
            of the each geometry in the supplied `gdf`. This makes it possible to keep track
            of how sampled pixel relates to the original geometries, i.e. multiple pixels 
            being extracted within the area of a single polygon that can be referred to using
            the `geometry_idx`. 
            
            The extracted data can subsequently be joined with the attribute table of
            the supplied `gdf` using:

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
            A tuple (geodataframe index, extracted values, coordinates) of the extracted
            raster values as a masked array and the  coordinates of the extracted pixels
            if `as_gdf=False`.
        """

        # rasterize polygon and line geometries
        if all(gdf.geom_type == "Polygon") or all(gdf.geom_type == "LineString"):

            shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf.index)]
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
            xys = rasterio.transform.xy(transform=self.transform, rows=rows, cols=cols)
            xys = np.transpose(xys)

        elif all(gdf.geom_type == "Point"):
            ids = gdf.index.values
            xys = gdf.bounds.iloc[:, 2:].values

        # extract raster pixels
        dtype = np.find_common_type([np.float32], self.dtypes)
        X = np.ma.zeros((xys.shape[0], self.count), dtype=dtype)
        
        if progress is True:
            disable_tqdm = False
        else:
            disable_tqdm = True

        for i, (layer, pbar) in enumerate(zip(self.iloc, tqdm(self.iloc, total=self.count, disable=disable_tqdm))):
            sampler = sample_gen(dataset=layer.ds, xy=xys, indexes=layer.bidx, masked=True)
            v = np.ma.asarray([i for i in sampler])
            X[:, i] = v.flatten()

        # return as geopandas array as default (or numpy arrays)
        if return_array is False:
            X = pd.DataFrame(
                data=X,
                columns=self.names,
                index=[pd.RangeIndex(0, X.shape[0]), ids]
            )
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
            geopandas.GeoDataFrame. If `return_array=True` then the extracted pixel
            values are returned as a tuple of numpy.ndarrays. 
        
        progress : bool (opt), default=False
            Show a progress bar for extraction.

        Returns
        -------
        geopandas.GeoDataFrame
            Geodataframe containing extracted data as point features if `return_array=False`

        tuple with three items if `return_array is True
            - numpy.ndarray
                Numpy masked array of extracted raster values, typically 2d.
            - numpy.ndarray
                1d numpy masked array of labelled sampled.
            - numpy.ndarray
                2d numpy masked array of row and column indexes of training pixels.
        """

        # open response raster and get labelled pixel indices and values
        arr = src.read(1, masked=True)
        rows, cols = np.nonzero(~arr.mask)
        xys = np.transpose(rasterio.transform.xy(src.transform, rows, cols))
        ys = arr.data[rows, cols]

        # extract Raster object values at row, col indices
        dtype = np.find_common_type([np.float32], self.dtypes)
        X = np.ma.zeros((xys.shape[0], self.count), dtype=dtype)

        if progress is True:
            disable_tqdm = False
        else:
            disable_tqdm = True

        for i, (layer, pbar) in enumerate(zip(self.iloc, tqdm(self.iloc, total=self.count, disable=disable_tqdm))):
            sampler = sample_gen(dataset=layer.ds, xy=xys, indexes=layer.bidx, masked=True)
            v = np.ma.asarray([i for i in sampler])
            X[:, i] = v.flatten()

        # summarize data
        if return_array is False:
            column_names = ["value"] + self.names
            gdf = pd.DataFrame(data=np.ma.column_stack((ys, X)), columns=column_names)
            gdf["geometry"] = list(zip(xys[:, 0], xys[:, 1]))
            gdf["geometry"] = gdf["geometry"].apply(Point)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=self.crs)
            return gdf

        return X, ys, xys

    def head(self):
        """Show the head (first rows, first columns) or tail (last rows, last columns)
        of pixels.
        """

        window = Window(col_off=0, row_off=0, width=20, height=10)
        return self.read(window=window)

    def tail(self):
        """Show the head (first rows, first columns) or tail (last rows, last columns)
        of pixels.
        """

        window = Window(
            col_off=self.width - 20, row_off=self.height - 10, width=20, height=10
        )
        return self.read(window=window)

    def to_pandas(self, max_pixels=50000, resampling="nearest"):
        """Raster to pandas DataFrame.

        Parameters
        ----------
        max_pixels: int (default 50000)
            Maximum number of pixels to sample.

        resampling : str (default 'nearest')
            Resampling method to use when applying decimated reads when out_shape is
            specified. Supported methods are: 'average', 'bilinear', 'cubic', 
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'q1', 'q3'.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing values of names of RasterLayers in the Raster as
            columns, and pixel values as rows.
        """

        # read dataset using decimated reads
        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels
        out_shape = (round(self.shape[0] * scaling), round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape, resampling=resampling)

        # get shape for Raster or RasterLayer
        try:
            bands, rows, cols = arr.shape
            nodatavals = self.nodatavals
        except ValueError:
            rows, cols = arr.shape
            bands = 1
            nodatavals = [self.nodata]

        # x and y grid coordinate arrays
        x_range = np.linspace(start=self.bounds.left, stop=self.bounds.right, num=cols)
        y_range = np.linspace(start=self.bounds.top, stop=self.bounds.bottom, num=rows)
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
