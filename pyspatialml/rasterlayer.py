import tempfile

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

import pyspatialml.base


class RasterLayer(pyspatialml.base.BaseRaster):
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
    
    def __add__(self, other):
        raise NotImplementedError
    
    def __sub__(self, other):
        raise NotImplementedError
    
    def __mul__(self, other):
        raise NotImplementedError

    def __div__(self, other):
        raise NotImplementedError
    
    def __pow__(self, other):
        raise NotImplementedError
    
    def __and__(self, other):
        """
        Intersects self with other. Equivalent to a intersection operation
        """
        raise NotImplementedError
    
    def __or__(self, other):
        """
        Fills gaps in self with pixels from other. Equivalent to a union 
        operation
        """
        raise NotImplementedError
    
    def __xor__(self, other):
        """
        Exclusive OR. Equivalent to a symmetrical difference where the result
        comprises pixels that occur in self or other, but not both
        """
        raise NotImplementedError

    def read(self, **kwargs):
        
        if 'resampling' in kwargs.keys():
            resampling_methods = [i.name for i in rasterio.enums.Resampling]

            if kwargs['resampling'] not in resampling_methods:
                raise ValueError(
                    'Invalid resampling method.' +
                    'Resampling method must be one of {0}:'.format(
                        resampling_methods))

            kwargs['resampling'] = rasterio.enums.Resampling[
                kwargs['resampling']]
            
        return self.ds.read(indexes=self.bidx, **kwargs)

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
        return pyspatialml.rasterlayer.RasterLayer(rasterio.band(src, 1))

    def plot(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        arr = self.read(masked=True)
        im = ax.imshow(arr,
                       extent=rasterio.plot.plotting_extent(self.ds),
                       cmap=self.cmap)
        plt.colorbar(im)
        
        return fig, ax
