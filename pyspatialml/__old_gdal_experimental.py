import numpy as np
import os
from tqdm import tqdm
from osgeo import gdal
import tempfile
import inspect

class RasterLayer(object):
    """Base class for opening and reading raster data"""

    def __init__(self, Band):
        """Initiates a RasterLayer object

        Parameters
        ----------
        Band : osgeo.gdal.Band object"""

        self.Band = Band
        self.nodata = Band.GetNoDataValue()

    def read(self, xoff=0, yoff=0, win_xsize=None, win_ysize=None, masked=False, **kwargs):
        """Reads the raster or a window of the raster into a numpy array

        Parameters
        ----------
        xoff : int
            Raster column index to start reading data from

        yoff : int
            Raster row index to start reading data from

        win_xsize : int
            Size of rectangular subset of raster to read in columns

        win_ysize : int
            Size of rectangular subset of raster to read in rows

        masked : bool, default=False
            Optionally mask the raster's nodata values and return as a
            numpy masked array

        Returns
        -------
        arr : array-like
            2D or 3D array if reading a osgeo.gdal.Band object, or reading a osgeo.gdal.Dataset"""

        # read GDAL Dataset
        if isinstance(self, PyRaster):
            arr = self.Dataset.ReadAsArray(xoff, yoff, win_xsize, win_ysize, **kwargs)

        # read GDAL GetRasterBand
        else:
            arr = self.Band.ReadAsArray(xoff, yoff, win_xsize, win_ysize, **kwargs)

        # optionally mask nodata values
        if masked is True:
            arr = np.ma.masked_array(arr, np.isin(arr, self.nodata))

        return arr

    def extract(self, response, field=None):
        """Sample a GDAL-supported raster dataset point of polygon
        features in a Geopandas Geodataframe or a labelled singleband
        raster dataset

        Parameters
        ----------
        response: osgeo.gdal.Band object or Geopandas DataFrame
            Single band raster containing labelled pixels, or
            a Geopandas GeoDataframe containing either point or polygon features

        field : str, optional
            Field name of attribute to be used the label the extracted data
            Used only if the response feature represents a GeoDataframe

        Returns
        -------
        X : array-like
            Numpy masked array of extracted raster values, typically 2d

        y: 1d array like
            Numpy masked array of labelled sampled

        xy: 2d array-like
            Numpy masked array of row and column indexes of training pixels"""

        if isinstance(self, PyRaster):
            src = self.Dataset
        else:
            src = self.Band

    def distance(self):
        pass

    def density(self):
        pass


class PyRaster(RasterLayer):
    def __init__(self, files, output=None, resolution='highest', outputBounds=None, xRes=None,
                 yRes=None, targetAlignedPixels=None, separate=True, bandList=None,
                 addAlpha=False, resampleAlg='nearest', outputSRS=None,
                 allowProjectionDifference=False, srcNodata=None, VRTNodata=None):

        """Simple function to stack a set of raster bands as a GDAL VRT file
        in order to perform spatial operations

        Parameters
        ----------
        files : list, str
            List of file paths of individual rasters to be stacked

        output : str
            File path of output VRT

        resolution : str, optional (default='highest')
            Resolution of output
            Options include 'highest', 'lowest', 'average', 'user'

        outputBounds : tuple, optional
            Bounding box of output in (xmin, ymin, xmax, ymax)

        xRes : float, optional
            x-resolution of output if 'resolution=user'

        yRes : float, optional
            y-resolution of output if 'resolution=user'

        targetAlignedPixels : bool, optional (default=False)
            whether to force output bounds to be multiple of output resolution

        separate : bool, optional (default=True)
            whether each source file goes into a separate stacked band in the VRT band

        bandList : list
            array of band numbers (index start at 1)

        addAlpha : bool, optional (default=False)
            whether to add an alpha mask band to the VRT when the source raster have none

        resampleAlg : str, optional (default = 'nearest')
            resampling mode

        outputSRS : str, optional
            assigned output SRS

        allowProjectionDifference : bool, optional (default=False)
            whether to accept input datasets have not the same projection.
            Note: they will *not* be reprojected

        srcNodata : list
            source nodata value(s)

        VRTNodata : list
            nodata values at the VRT band level"""

        # get dict of arguments from function call
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        self.meta = values

        # buildvrt
        if output is None:
            temp_filename = next(tempfile._get_candidate_names())
            temp_dirname = tempfile._get_default_tempdir()
            output = os.path.join(temp_dirname, temp_filename)

        outds = gdal.BuildVRT(
            destName=output, srcDSOrSrcDSTab=files, separate=separate,
            resolution=resolution, resampleAlg=resampleAlg,
            outputBounds=outputBounds, xRes=xRes, yRes=yRes,
            targetAlignedPixels=targetAlignedPixels, bandList=bandList,
            addAlpha=addAlpha, srcNodata=srcNodata, VRTNodata=VRTNodata,
            allowProjectionDifference=allowProjectionDifference,
            outputSRS=outputSRS)
        outds.FlushCache()

        # open and assign gdal RasterBands to file names
        self.file = output
        self.Dataset = gdal.Open(self.file)
        self.nodata = np.array([])

        for i, name in enumerate(files):
            # attach bands
            validname = self._make_names(name)
            band = self.Dataset.GetRasterBand(i+1)
            setattr(self, validname, RasterLayer(band))

            # set nodata values
            self.nodata = np.append(self.nodata, band.GetNoDataValue())

        # dataset resolution
        self.xres = self.Dataset.GetGeoTransform()[1]
        self.yres = -self.Dataset.GetGeoTransform()[5]

        # dataset extent
        x_min, xres, xskew, y_max, yskew, yres = self.Dataset.GetGeoTransform()
        x_max = x_min + (self.Dataset.RasterXSize * self.xres)
        y_min = y_max + (self.Dataset.RasterYSize * self.yres)
        self.extent = (x_min, y_min, x_max, y_max)
        self.RasterXSize = int((x_max - x_min) / self.xres)
        self.RasterYSize = abs(int((y_max - y_min) / self.yres))

    @staticmethod
    def _make_names(name):
        """Converts a filename to a valid method name"""
        validname = os.path.basename(name)
        validname = validname.split(os.path.extsep)[0]
        validname = validname.replace(' ', '_')
        return validname

    def addBand(self, file):
        pass

    def predict(self, estimator, file_path, predict_type='raw', indexes=None,
                driver='GTiff', dtype='float32', nodata=-99999):

        """Prediction on list of GDAL rasters using a fitted scikit learn model

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        file_path : str
            Path to a GeoTiff raster for the classification results

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
            Nodata value for file export"""

        pass

        # chose prediction function
        if predict_type == 'raw':
            predfun = self._predfun

        elif predict_type == 'prob':
            predfun = self._probfun

        # determine output count
        if predict_type == 'prob' and isinstance(indexes, int):
            indexes = range(indexes, indexes + 1)

        elif predict_type == 'prob' and indexes is None:
            img = self.read(masked=True, xoff=0, yoff=0, win_xsize=1, win_ysize=1)
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape(
                (n_samples, n_features))
            result = estimator.predict_proba(flat_pixels)
            indexes = range(result.shape[0])

        elif predict_type == 'raw':
            indexes = range(1)

        # define windows
        windows = self.block_shapes(self.Dataset.GetRasterBand(1))

        # generator gets raster arrays for each window
        data_gen = (self.read(*window, masked=True) for window in windows)

        # open output file with updated metadata
        n_bands = len(indexes)
        dst = gdal.GetDriverByName(driver).Create(file_path,
                                                  self.Dataset.RasterXSize,
                                                  self.Dataset.RasterYSize,
                                                  n_bands,
                                                  gdal.GDT_Int32)

        transform = self.Dataset.GetGeoTransform()
        dst.SetGeoTransform(transform)

        try:
            with tqdm(total=len(windows)) as pbar:
                for window, arr in zip(windows, data_gen):
                    result = predfun(arr, estimator)

                    for index in indexes:
                        band = dst.GetRasterBand(index+1)
                        band.WriteArray(result[index, :, :].astype(dtype),
                                        xoff=window[0],
                                        yoff=window[1])
                    pbar.update(1)

            for index in indexes:
                band = dst.GetRasterBand(index+1)
                band.SetNoDataValue(nodata)
                band.FlushCache()
        except:
            pass

        finally:
            dst.FlushCache()

        return PyRaster([file_path])

    def block_shapes(self, Band):
        """Generator for windows for optimal reading and writing based on the raster format
        Windows are returns as a tuple with xoff, yoff, xsize, ysize

        Parameters
        ----------
        Band : osgeo.gdal.Band object"""

        block_shape = Band.GetBlockSize() # xsize, ysize

        for i in range(0, self.RasterXSize, block_shape[0]):
            if i + block_shape[0] < self.RasterXSize:
                numCols = block_shape[0]
            else:
                numCols = self.RasterXSize - i

            for j in range(0, self.RasterYSize, block_shape[1]):
                if j + block_shape[1] < self.RasterYSize:
                    numRows = block_shape[1]
                else:
                    numRows = self.RasterYSize - j

                yield (i, j, numCols, numRows)

    @staticmethod
    def _predfun(img, estimator):
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

        # reshape the prediction from a 1D into 3D array [band, row, col]
        result_cla = result_cla.reshape((1, rows, cols))

        return result_cla

    @staticmethod
    def _probfun(img, estimator):
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