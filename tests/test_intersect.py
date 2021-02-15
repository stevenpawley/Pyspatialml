from unittest import TestCase

import numpy as np

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestIntersect(TestCase):
    def setUp(self) -> None:
        # inputs
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]
        self.stack = Raster(self.predictors)

        # test results
        self.result = None

    def tearDown(self) -> None:
        self.stack.close()
        self.result.close()

    def test_intersect_defaults(self):
        self.result = self.stack.intersect()

        # check raster object
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, self.stack.count)
        self.assertEqual(self.result.read(masked=True).count(), 810552)

        # test nodata value is recognized
        self.assertEqual(self.result.read(masked=True).min(), 1.0)
        self.assertEqual(self.result.read(masked=True).max(), 255.0)

    def test_intersect_custom_dtype(self):
        self.result = self.stack.intersect(dtype=np.int16)

        # check raster object
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, self.stack.count)
        self.assertEqual(self.result.read(masked=True).count(), 810552)

        # test nodata value is recognized
        self.assertEqual(self.result.read(masked=True).min(), 1)
        self.assertEqual(self.result.read(masked=True).max(), 255)

    def test_intersect_custom_nodata(self):
        self.result = self.stack.intersect(dtype=np.int16, nodata=-999)

        # check raster object
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, self.stack.count)
        self.assertEqual(self.result.read(masked=True).count(), 810552)

        # test nodata value is recognized
        self.assertEqual(self.result.read(masked=True).min(), 1)
        self.assertEqual(self.result.read(masked=True).max(), 255)

    def test_intersect_in_memory(self):
        self.result = self.stack.intersect(in_memory=True)

        # check raster object
        self.assertIsInstance(self.result, Raster)
