from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc
import geopandas as gpd
import numpy as np


class TestMask(TestCase):

    training_py = gpd.read_file(nc.polygons)
    mask_py = training_py.iloc[0:1, :]

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    def test_mask_defaults(self):

        masked_object = self.stack.mask(self.mask_py)

        # check raster object
        self.assertIsInstance(masked_object, Raster)
        self.assertEqual(masked_object.count, self.stack.count)
        self.assertEqual(masked_object.read(masked=True).count(), 738)

        # test nodata value is recognized
        self.assertEqual(masked_object.read(masked=True).min(), 38.0)
        self.assertEqual(masked_object.read(masked=True).max(), 168.0)

    def test_mask_inverted(self):

        masked_object = self.stack.mask(self.mask_py, invert=True)

        # check raster object
        self.assertIsInstance(masked_object, Raster)
        self.assertEqual(masked_object.count, self.stack.count)
        self.assertEqual(masked_object.read(masked=True).count(), 1051444)

        # test nodata value is recognized
        self.assertEqual(masked_object.read(masked=True).min(), 1.0)
        self.assertEqual(masked_object.read(masked=True).max(), 255.0)

    def test_mask_custom_dtype(self):

        masked_object = self.stack.mask(self.mask_py, dtype=np.int16)

        # check raster object
        self.assertIsInstance(masked_object, Raster)
        self.assertEqual(masked_object.count, self.stack.count)
        self.assertEqual(masked_object.read(masked=True).count(), 738)

        # test nodata value is recognized
        self.assertEqual(masked_object.read(masked=True).min(), 38)
        self.assertEqual(masked_object.read(masked=True).max(), 168)

    def test_mask_custom_nodata(self):

        masked_object = self.stack.mask(self.mask_py, nodata=-99999)

        # check raster object
        self.assertIsInstance(masked_object, Raster)
        self.assertEqual(masked_object.count, self.stack.count)
        self.assertEqual(masked_object.read(masked=True).count(), 738)

        # test nodata value is recognized
        self.assertEqual(masked_object.read(masked=True).min(), 38.0)
        self.assertEqual(masked_object.read(masked=True).max(), 168.0)
