from unittest import TestCase

import geopandas as gpd
import numpy as np

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestMask(TestCase):
    def setUp(self) -> None:
        # test inputs
        training_py = gpd.read_file(nc.polygons)
        self.mask_py = training_py.iloc[0:1, :]

        predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                      nc.band7]
        self.stack = Raster(predictors)

        # test results
        self.masked_object = None

    def tearDown(self) -> None:
        self.stack.close()
        self.masked_object.close()

    def test_mask_defaults(self):
        self.masked_object = self.stack.mask(self.mask_py)

        # check raster object
        self.assertIsInstance(self.masked_object, Raster)
        self.assertEqual(self.masked_object.count, self.stack.count)
        self.assertEqual(self.masked_object.read(masked=True).count(), 738)

        # test nodata value is recognized
        self.assertEqual(self.masked_object.read(masked=True).min(), 38.0)
        self.assertEqual(self.masked_object.read(masked=True).max(), 168.0)

    def test_mask_inverted(self):
        self.masked_object = self.stack.mask(self.mask_py, invert=True)

        # check raster object
        self.assertIsInstance(self.masked_object, Raster)
        self.assertEqual(self.masked_object.count, self.stack.count)
        self.assertEqual(self.masked_object.read(masked=True).count(), 1051444)

        # test nodata value is recognized
        self.assertEqual(self.masked_object.read(masked=True).min(), 1.0)
        self.assertEqual(self.masked_object.read(masked=True).max(), 255.0)

    def test_mask_custom_dtype(self):
        self.masked_object = self.stack.mask(self.mask_py, dtype=np.int16)

        # check raster object
        self.assertIsInstance(self.masked_object, Raster)
        self.assertEqual(self.masked_object.count, self.stack.count)
        self.assertEqual(self.masked_object.read(masked=True).count(), 738)

        # test nodata value is recognized
        self.assertEqual(self.masked_object.read(masked=True).min(), 38)
        self.assertEqual(self.masked_object.read(masked=True).max(), 168)

    def test_mask_custom_nodata(self):
        self.masked_object = self.stack.mask(self.mask_py, nodata=-99999)

        # check raster object
        self.assertIsInstance(self.masked_object, Raster)
        self.assertEqual(self.masked_object.count, self.stack.count)
        self.assertEqual(self.masked_object.read(masked=True).count(), 738)

        # test nodata value is recognized
        self.assertEqual(self.masked_object.read(masked=True).min(), 38.0)
        self.assertEqual(self.masked_object.read(masked=True).max(), 168.0)

    def test_mask_in_memory(self):
        self.masked_object = self.stack.mask(self.mask_py, in_memory=True)

        # check raster object
        self.assertIsInstance(self.masked_object, Raster)
