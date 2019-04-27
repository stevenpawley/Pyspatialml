from unittest import TestCase
from pyspatialml import Raster, RasterLayer
import rasterio

class TestInit(TestCase):

    band1 = 'lsat7_2000_10.tif'
    band2 = 'lsat7_2000_20.tif'
    band3 = 'lsat7_2000_30.tif'
    band4 = 'lsat7_2000_40.tif'
    band5 = 'lsat7_2000_50.tif'
    band7 = 'lsat7_2000_70.tif'
    predictors = [band1, band2, band3, band4, band5, band7]

    def test_initiation(self):

        # file paths ----------------------
        # test init from list of file paths
        stack = Raster(self.predictors)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 6)
        stack = None

        # test init from single file path
        stack = Raster(self.band1)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 1)
        stack = None

        # rasterio.io.datasetreader --------
        # test init from single rasterio.io.datasetreader
        with rasterio.open(self.band1) as src:
            stack = Raster(src)
            self.assertIsInstance(stack, Raster)
            self.assertEqual(stack.count, 1)
            stack = None

        # test init from list of rasterio.io.datasetreader objects
        srcs = []
        for f in self.predictors:
            srcs.append(rasterio.open(f))
        stack = Raster(srcs)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 6)
        stack = None

        # rasterio.band ---------------------
        # test init from single rasterio.band object
        with rasterio.open(self.band1) as src:
            band = rasterio.band(src, 1)
            stack = Raster(band)
            self.assertIsInstance(stack, Raster)
            self.assertEqual(stack.count, 1)
            stack = None

        # test init from list of rasterio.band objects
        bands = []
        for f in self.predictors:
            src = rasterio.open(f)
            bands.append(rasterio.band(src, 1))
        stack = Raster(bands)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 6)
        stack = None

        # RasterLayer objects ---------------
        # test init from a single RasterLayer object
        with rasterio.open(self.band1) as src:
            band = rasterio.band(src, 1)
            layer = RasterLayer(band)
            stack = Raster(layer)
            self.assertIsInstance(stack, Raster)
            self.assertEqual(stack.count, 1)
            stack = None

        # test init from a list of RasterLayer objects
        layers = []
        for f in self.predictors:
            src = rasterio.open(f)
            band = rasterio.band(src, 1)
            layers.append(RasterLayer(band))
        stack = Raster(layers)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 6)
        stack = None

        # Mixed types ------------------------
        with rasterio.open(self.band1) as src:
            self.assertRaises(Raster([src, self.band2]), ValueError)
