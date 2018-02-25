from warnings import warn
import numpy as np
import rasterio
import os
import shapely


def print_progressbar(iteration, total, prefix='', suffix='', decimals=1,
                      length=100, fill='█'):
    """Call in a loop to create terminal progress bar
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113

    Parameters
    ----------
    iteration: int
        Current iteration
    total: int
        Total iterations
    prefix: str, optional
        prefix string (Str)
    suffix: str, optional
        suffix
    decimals: int, optional
        Positive number of decimals in percent complete
    length: int, optional
        Character length of bar
    fill: str, optional
        bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s|%s|%s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def applier(rs, func, file_path=None, rowchunk=25, n_jobs=-1, **kwargs):
    """Applies a function to a RasterStack object in image
    strips"""

    # processing region dimensions
    rows, cols = rs.height, rs.width

    # generator for row increments, tuple (startrow, endrow)
    windows = ((row, row+rowchunk) if row+rowchunk <= rows else
               (row, rows) for row in range(0, rows, rowchunk))
    result = []

    # Loop through rasters strip-by-strip
    for start, end, in windows:
        print_progressbar(start, rows, length=50)

        img = rs.read(window=((start, end), (0, cols)))

        if file_path:
            result = func(img, **kwargs)
            if start == 0:
                func_output = rasterio.open(
                    file_path, mode='w', driver='GTiff', height=rows,
                    width=cols, count=result.shape[0], dtype='float32',
                    crs=rs.meta['crs'],
                    transform=rs.meta['transform'],
                    nodata=-99999)
            func_output.write(
                result.astype('float32'),
                window=((start, end), (0, cols)))
        else:
            result.append(func(img, **kwargs))

    # summarize results
    if file_path:
        func_output.close()
        return func_output
    else:
        return result


class RasterStack(object):
    """Access a group of aligned GDAL-supported raster images as a single
    dataset

    Attibutes
    ---------
    names : list (str)
        List of names of rasters in the RasterStack
        Defined by default by filename (sans file extension)
    count : int
        Number of raster layers
    shape : tuple (int)
        Shape of raster data in (n_rows, n_cols)
    height : int
        Height (number of rows) in the raster data
    width : int
        Width (number of columns) in the raster data
    meta : Dict
        The basic metadata of the dataset
    affine : Affine class
        Affine transformation matrix
    bounds : rasterio.coords.BoundingBox
        Bounding box named tuple, defining extent in cartesian coordinates
        BoundingBox(left, bottom, right, top)
    """

    def __init__(self, rasters):
        """Create a RasterStack object

        Parameters
        ----------
        rasters : list (str)
            List of file paths to GDAL-supported rasters to create a
            RasterStack object. The rasters can contain single or multiple
            bands, but they need to be aligned in terms of their width, height,
            and transform.
        """
        self.__check_alignment(rasters)

        # open bands
        self.names = []
        self.count = 0

        for r in rasters:
            bandname = os.path.basename(r)
            bandname = os.path.splitext(bandname)[0]
            self.names.append(bandname)
            src = rasterio.open(r)
            self.count += src.count
            setattr(self, bandname, src)
            src.close()

        self.shape = (self.count, src.shape[0], src.shape[1])
        self.height = src.shape[0]
        self.width = src.shape[1]
        self.meta = src.meta.copy()
        self.meta['count'] = self.count
        self.affine = self.meta['affine']
        self.bounds = src.bounds
        self.files = rasters

    def __check_alignment(self, rasters):
        """Check that a list of rasters are aligned with the same pixel
        dimensions and geotransforms

        Parameters
        ----------
        rasters: list (str)
            List of file paths to the rasters
        """

        src_meta = []
        for r in rasters:
            src = rasterio.open(r)
            src_meta.append(src.meta.copy())
            src.close()

        if not all(i['crs'] == src_meta[0]['crs'] for i in src_meta):
            warn('crs of all rasters does not match possible unintended consequences')

        if not all([i['height'] == src_meta[0]['height'] or
                    i['width'] == src_meta[0]['width'] or
                    i['transform'] == src_meta[0]['transform']
                    for i in src_meta]):
            print('Predictor rasters do not all have the same dimensions or',
                  'transform')
            print('Use the .utils_align_rasters function')

    def read(self, window=None):
        """Read data from RasterStack as a 3D numpy array

        Parameters
        ----------
        window : tuple
            A pair of tuples ((row_start, row_stop), (col_start, col_stop))
            to read a window of raster data as a 3D numpy array

        Returns
        -------
        data : 3D array-like
            3D masked numpy array containing data from RasterStack rasters
        """
        # create numpy array for stack
        if window:
            row_start = window[0][0]
            row_stop = window[0][1]
            col_start = window[1][0]
            col_stop = window[1][1]
            width = abs(col_stop-col_start)
            height = abs(row_stop-row_start)
            shape = (self.count, height, width)
        else:
            shape = self.shape

        data = np.ma.zeros(shape)

        # loop and read data into numpy
        i = 0
        for raster in self.names:
            src = getattr(self, raster)
            src = rasterio.open(src.name)
            data[i:i+src.count, :, :] = src.read(masked=True, window=window)
            i += src.count
            src.close()

        return data

    @staticmethod
    def predfun(img, **kwargs):
        estimator = kwargs['estimator']
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

        # replace mask and fill masked values with nodata value
        result_cla = np.ma.masked_array(
            result_cla, mask=flat_pixels_mask.any(axis=1))
        result_cla = np.ma.filled(result_cla, fill_value=-99999)

        # reshape the prediction from a 1D matrix/list
        # back into the original format
        result_cla = result_cla.reshape((1, rows, cols))

        return result_cla

    @staticmethod
    def probfun(img, **kwargs):
        estimator = kwargs['estimator']
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
        result_proba = np.ma.filled(
            result_proba, fill_value=-99999)

        # reshape band into rasterio format [band, row, col]
        result_proba = result_proba.transpose(2, 0, 1)

        return result_proba

    def predict(self, estimator, file_path=None, rowchunk=25):

        result = applier(rs=self, func=self.predfun, file_path=file_path,
                         rowchunk=rowchunk, **{'estimator': estimator})

        if not file_path:
            # concatenate rows in [band,row,col]
            result = np.concatenate(result, axis=1)
            return result
        else:
            return None

    def predict_proba(self, estimator, file_path=None, index=None,
                      rowchunk=25):

        # convert index to list
        if isinstance(index, int):
            index = [index]

        # prediction
        result = applier(rs=self, func=self.probfun, file_path=file_path,
                         rowchunk=rowchunk, **{'estimator': estimator})

        if not file_path:
            # concatenate rows in [band,row,col]
            result = np.concatenate(result, axis=1)
            return result
        else:
            return None

    @staticmethod
    def __value_extractor(img, **kwargs):
        # split numpy array bands(axis=0) into labelled pixels and
        # raster data
        response_arr = img[-1, :, :]
        raster_arr = img[0:-1, :, :]

        # returns indices of labelled values
        is_train = np.nonzero(~response_arr.mask)

        # get the labelled values
        labels = response_arr.data[is_train]

        # extract data at labelled pixel locations
        data = raster_arr[:, is_train[0], is_train[1]]

        # Remove nan rows from training data
        data = data.filled(np.nan)

        # combine training data, locations and labels
        data = np.vstack((data, labels))

        return data

    def extract_pixels(self, response_path, na_rm=True):

        # create new RasterStack object with labelled pixels as
        # last band in the stack
        temp_stack = RasterStack(self.files + response_path)

        # extract training data
        data = applier(temp_stack, self.__value_extractor)
        data = np.concatenate(data, axis=1)
        raster_vals = data[0:-1, :]
        labelled_vals = data[-1, :]

        if na_rm is True:
            X = raster_vals[~np.isnan(raster_vals).any(axis=1)]
            y = labelled_vals[np.where(~np.isnan(raster_vals).any(axis=1))]
        else:
            X = raster_vals
            y = labelled_vals

        return (X, y)

    def extract_features(self, y):
        """Samples a list of GDAL rasters using a point data set

        Parameters
        ----------
        y : Geopandas GeoDataFrame

        Returns
        -------
        gdf : Geopandas GeoDataFrame
            GeoDataFrame containing extract raster values at the point
            locations
        """

        from rasterio.sample import sample_gen

        # point feature data
        if isinstance(y.geometry.all(), shapely.geometry.point.Point):
            # get coordinates and label for each point in points_gdf
            coordinates = np.array(y.bounds.iloc[:, :2])

            # loop through each raster
            for r in self.names:
                raster = getattr(self, r)
                raster = rasterio.open(raster.name)
                nodata = raster.nodata
                data = np.array([i for i in sample_gen(raster, coordinates)],
                                dtype='float32')
                data[data == nodata] = np.nan
                if data.shape[1] == 1:
                    y[r] = data
                else:
                    for i in range(1, data.shape[1]+1):
                        y[r+'_'+str(i)] = data[:, i-1]
                raster.close()

        # polygon features
        if isinstance(y.geometry.all(), shapely.geometry.Polygon):
            print('Not implemented yet')
    #        template = rasterio.open(rasters[0], mode='r')
    #        response_np = np.zeros((template.height, template.width))
    #        response_np[:] = -99999
    #
    #        # this is where we create a generator of geom to use in rasterizing
    #        if field is None:
    #            shapes = (geom for geom in gdf.geometry)
    #
    #        if field is not None:
    #            shapes = ((geom, value) for geom, value in zip(
    #                      gdf.geometry, gdf[field]))
    #
    #        features.rasterize(
    #            shapes=shapes, fill=-99999, out=response_np,
    #            transform=template.transform, default_value=1)
    #
    #        response_np = np.ma.masked_where(response_np == -99999, response_np)
    #
    #        X, y, y_indexes = __extract_pixels(
    #                response_np, rasters, field=None, na_rm=True, lowmem=False)

        return y
