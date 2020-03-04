from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc
import numpy as np


class TestIntersect(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    def test_intersect_defaults(self):

        result = self.stack.intersect()

        # check raster object
        self.assertIsInstance(result, Raster)
        self.assertEqual(result.count, self.stack.count)
        self.assertEqual(result.read(masked=True).count(), 810552)

        # test nodata value is recognized
        self.assertEqual(result.read(masked=True).min(), 1.0)
        self.assertEqual(result.read(masked=True).max(), 255.0)

    def test_intersect_custom_dtype(self):

        result = self.stack.intersect(dtype=np.int16)

        # check raster object
        self.assertIsInstance(result, Raster)
        self.assertEqual(result.count, self.stack.count)
        self.assertEqual(result.read(masked=True).count(), 810552)

        # test nodata value is recognized
        self.assertEqual(result.read(masked=True).min(), 1)
        self.assertEqual(result.read(masked=True).max(), 255)

    def test_intersect_custom_nodata(self):

        result = self.stack.intersect(dtype=np.int16, nodata=-999)

        # check raster object
        self.assertIsInstance(result, Raster)
        self.assertEqual(result.count, self.stack.count)
        self.assertEqual(result.read(masked=True).count(), 810552)

        # test nodata value is recognized
        self.assertEqual(result.read(masked=True).min(), 1)
        self.assertEqual(result.read(masked=True).max(), 255)
