from unittest import TestCase
from pyspatialml import Raster, RasterLayer
from pyspatialml.datasets import nc
import rasterio


class TestInit(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_initiation(self):

        # file paths ----------------------
        # test init from list of file paths
        stack = Raster(self.predictors)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 6)
        stack = None

        # test init from single file path
        stack = Raster(nc.band1)
        self.assertIsInstance(stack, Raster)
        self.assertEqual(stack.count, 1)
        stack = None

        # rasterio.io.datasetreader --------
        # test init from single rasterio.io.datasetreader
        with rasterio.open(nc.band1) as src:
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
        with rasterio.open(nc.band1) as src:
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
        with rasterio.open(nc.band1) as src:
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
