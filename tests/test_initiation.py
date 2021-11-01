from unittest import TestCase

import os
import rasterio
import numpy as np

from pyspatialml import Raster, RasterLayer
from pyspatialml.datasets import nc


class TestInit(TestCase):
    def setUp(self) -> None:
        # inputs
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]

        # test results
        self.stack = None

    def tearDown(self) -> None:
        self.stack.close()

    def test_initiation_files(self):
        # test init from list of file paths
        self.stack = Raster(self.predictors)
        self.assertIsInstance(self.stack, Raster)
        self.assertEqual(self.stack.count, 6)

    def test_initiation_file(self):
        # test init from single file path
        self.stack = Raster(nc.band1)
        self.assertIsInstance(self.stack, Raster)
        self.assertEqual(self.stack.count, 1)

    def test_initiation_datasetreader(self):
        # test init from single rasterio.io.datasetreader
        with rasterio.open(nc.band1) as src:
            self.stack = Raster(src)
            self.assertIsInstance(self.stack, Raster)
            self.assertEqual(self.stack.count, 1)

    def test_initiation_list_datasetreader(self):
        # test init from list of rasterio.io.datasetreader objects
        srcs = []
        for f in self.predictors:
            srcs.append(rasterio.open(f))
        self.stack = Raster(srcs)
        self.assertIsInstance(self.stack, Raster)
        self.assertEqual(self.stack.count, 6)

    def test_initiation_band(self):
        # test init from single rasterio.band object
        with rasterio.open(nc.band1) as src:
            band = rasterio.band(src, 1)
            self.stack = Raster(band)
            self.assertIsInstance(self.stack, Raster)
            self.assertEqual(self.stack.count, 1)

    def test_initiation_list_bands(self):
        # test init from list of rasterio.band objects
        bands = []
        for f in self.predictors:
            src = rasterio.open(f)
            bands.append(rasterio.band(src, 1))
        self.stack = Raster(bands)
        self.assertIsInstance(self.stack, Raster)
        self.assertEqual(self.stack.count, 6)

    def test_initiation_rasterlayer(self):
        # test init from a single RasterLayer object
        with rasterio.open(nc.band1) as src:
            band = rasterio.band(src, 1)
            layer = RasterLayer(band)
            self.stack = Raster(layer)
            self.assertIsInstance(self.stack, Raster)
            self.assertEqual(self.stack.count, 1)

    def test_initiation_list_rasterlayer(self):
        # test init from a list of RasterLayer objects
        layers = []
        for f in self.predictors:
            src = rasterio.open(f)
            band = rasterio.band(src, 1)
            layers.append(RasterLayer(band))
        self.stack = Raster(layers)
        self.assertIsInstance(self.stack, Raster)
        self.assertEqual(self.stack.count, 6)

    def test_initiation_array(self):
        # check initiation of single-band raster from file
        arr = np.zeros((100, 100))
        self.stack = Raster(arr)
        
        # check output is written to tempfile
        self.assertTrue(os.path.exists(self.stack.iloc[0].file))

        # check some operations on the created raster
        layer_name = list(self.stack.names)[0]
        self.stack = self.stack.rename({layer_name: 'new_layer'})
        self.assertEqual(list(self.stack.names)[0], 'new_layer')

        # check initiation from array in memory
        arr = np.zeros((100, 100))
        self.stack = Raster(arr, in_memory=True)
        layer_name = list(self.stack.names)[0]

        self.stack = self.stack.rename({layer_name: 'new_layer'})
        self.assertEqual(list(self.stack.names)[0], 'new_layer')
