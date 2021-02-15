from unittest import TestCase

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestToCrs(TestCase):
    def setUp(self) -> None:
        # test inputs
        predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                      nc.band7]
        self.stack = Raster(predictors)

        # test results
        self.stack_prj = None

    def tearDown(self) -> None:
        self.stack.close()
        self.stack_prj.close()

    def test_to_crs_defaults(self):
        self.stack_prj = self.stack.to_crs({"init": "EPSG:4326"})

        # check raster object
        self.assertIsInstance(self.stack_prj, Raster)
        self.assertEqual(self.stack_prj.count, self.stack.count)
        self.assertEqual(self.stack_prj.read(masked=True).count(), 1012061)

        # test nodata value is recognized
        self.assertEqual(
            self.stack_prj.read(masked=True).min(),
            self.stack.read(masked=True).min()
        )
        self.assertEqual(
            self.stack_prj.read(masked=True).max(),
            self.stack.read(masked=True).max()
        )

    def test_to_crs_custom_nodata(self):
        self.stack_prj = self.stack.to_crs({"init": "EPSG:4326"}, nodata=-999)

        # check raster object
        self.assertIsInstance(self.stack_prj, Raster)
        self.assertEqual(self.stack_prj.count, self.stack.count)
        self.assertEqual(self.stack_prj.read(masked=True).count(), 1012061)

        # test nodata value is recognized
        self.assertEqual(
            self.stack_prj.read(masked=True).min(),
            self.stack.read(masked=True).min()
        )
        self.assertEqual(
            self.stack_prj.read(masked=True).max(),
            self.stack.read(masked=True).max()
        )

    def test_to_crs_in_memory(self):
        self.stack_prj = self.stack.to_crs({"init": "EPSG:4326"},
                                           in_memory=True)

        # check raster object
        self.assertIsInstance(self.stack_prj, Raster)
