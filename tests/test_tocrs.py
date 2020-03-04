from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc


class TestToCrs(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    def test_to_crs_defaults(self):

        stack_prj = self.stack.to_crs({"init": "EPSG:4326"})

        # check raster object
        self.assertIsInstance(stack_prj, Raster)
        self.assertEqual(stack_prj.count, self.stack.count)
        self.assertEqual(stack_prj.read(masked=True).count(), 1012061)

        # test nodata value is recognized
        self.assertEqual(
            stack_prj.read(masked=True).min(), self.stack.read(masked=True).min()
        )
        self.assertEqual(
            stack_prj.read(masked=True).max(), self.stack.read(masked=True).max()
        )

    def test_to_crs_custom_nodata(self):

        stack_prj = self.stack.to_crs({"init": "EPSG:4326"}, nodata=-999)

        # check raster object
        self.assertIsInstance(stack_prj, Raster)
        self.assertEqual(stack_prj.count, self.stack.count)
        self.assertEqual(stack_prj.read(masked=True).count(), 1012061)

        # test nodata value is recognized
        self.assertEqual(
            stack_prj.read(masked=True).min(), self.stack.read(masked=True).min()
        )
        self.assertEqual(
            stack_prj.read(masked=True).max(), self.stack.read(masked=True).max()
        )
