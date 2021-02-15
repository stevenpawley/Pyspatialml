from unittest import TestCase

import geopandas as gpd

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestToCrs(TestCase):
    def setUp(self) -> None:
        # inputs
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]
        self.stack = Raster(self.predictors)
        training_py = gpd.read_file(nc.polygons)
        self.crop_bounds = training_py.loc[0, "geometry"].bounds

        # outputs
        self.cropped = None

    def tearDown(self) -> None:
        self.stack.close()
        self.cropped.close()

    def test_crop_defaults(self):
        self.cropped = self.stack.crop(self.crop_bounds)

        # check raster object
        self.assertIsInstance(self.cropped, Raster)
        self.assertEqual(self.cropped.count, self.stack.count)
        self.assertEqual(self.cropped.read(masked=True).count(), 1440)

        # test nodata value is recognized
        self.assertEqual(self.cropped.read(masked=True).min(), 35.0)
        self.assertEqual(self.cropped.read(masked=True).max(), 168.0)

    def test_crop_in_memory(self):
        self.cropped = self.stack.crop(self.crop_bounds, in_memory=True)
        self.assertIsInstance(self.cropped, Raster)
