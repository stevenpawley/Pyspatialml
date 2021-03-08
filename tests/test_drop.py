from unittest import TestCase

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestDrop(TestCase):
    def setUp(self) -> None:
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]

    def test_drop_inplace(self):
        stack = Raster(self.predictors)
        stack.drop(labels="lsat7_2000_50", in_place=True)

        # check that Raster object is returned
        self.assertIsInstance(stack, Raster)

        # check that RasterLayer has been dropped
        self.assertEqual(stack.count, 5)
        self.assertNotIn("lsat7_2000_50", stack.names)
        stack.close()

    def test_drop_with_copy(self):
        stack = Raster(self.predictors)
        names = stack.names
        result = stack.drop(labels="lsat7_2000_50", in_place=False)

        # check that Raster object is returned
        self.assertIsInstance(result, Raster)

        # check that RasterLayer has been dropped
        self.assertEqual(result.count, 5)
        self.assertNotIn("lsat7_2000_50", result.names)

        # check that original raster is unaffected
        self.assertEqual(stack.count, 6)
        self.assertEqual(stack.names, names)
        stack.close()
        result.close()

    def test_drop_in_memory(self):
        stack = Raster(self.predictors)
        names = stack.names

        result = stack.intersect(in_memory=True)
        result = stack.drop(labels="lsat7_2000_50", in_place=False)

        # check that Raster object is returned
        self.assertIsInstance(result, Raster)

        # check that RasterLayer has been dropped
        self.assertEqual(result.count, 5)
        self.assertNotIn("lsat7_2000_50", result.names)

        # check that original raster is unaffected
        self.assertEqual(stack.count, 6)
        self.assertEqual(stack.names, names)
        stack.close()
