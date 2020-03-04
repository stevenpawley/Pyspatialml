from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc
import geopandas as gpd


class TestToCrs(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    training_py = gpd.read_file(nc.polygons)
    crop_bounds = training_py.loc[0, "geometry"].bounds

    def test_crop_defaults(self):

        stack_cropped = self.stack.crop(self.crop_bounds)

        # check raster object
        self.assertIsInstance(stack_cropped, Raster)
        self.assertEqual(stack_cropped.count, self.stack.count)
        self.assertEqual(stack_cropped.read(masked=True).count(), 1440)

        # test nodata value is recognized
        self.assertEqual(stack_cropped.read(masked=True).min(), 35.0)
        self.assertEqual(stack_cropped.read(masked=True).max(), 168.0)
