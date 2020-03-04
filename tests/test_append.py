from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc


class TestAppend(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_append_inplace(self):

        # append another Raster containing a single layer with identical name
        stack = Raster(self.predictors)
        band7_mean = stack["lsat7_2000_70"].read(masked=True).mean()
        stack.append(Raster(nc.band7))

        self.assertEqual(stack.names[5], "lsat7_2000_70_1")
        self.assertEqual(stack.names[-1], "lsat7_2000_70_2")
        self.assertEqual(
            stack.lsat7_2000_70_1.read(masked=True).mean(),
            stack.lsat7_2000_70_2.read(masked=True).mean(),
            band7_mean,
        )

        # append a multiband raster
        stack = Raster(self.predictors)
        stack.append(Raster(nc.multiband))
        self.assertEqual(stack.names[6], "landsat_multiband_1")

    def test_append_with_copy(self):

        # append another Raster containing a single layer with identical name
        stack = Raster(self.predictors)
        band7_mean = stack["lsat7_2000_70"].read(masked=True).mean()
        result = stack.append(Raster(nc.band7), in_place=False)

        # check that original is untouched
        self.assertEqual(stack.count, 6)

        # check that result contains appended raster
        self.assertEqual(result.names[5], "lsat7_2000_70_1")
        self.assertEqual(result.names[-1], "lsat7_2000_70_2")

        # check that band 7 stats are the same after appending
        self.assertEqual(
            result.lsat7_2000_70_1.read(masked=True).mean(),
            result.lsat7_2000_70_2.read(masked=True).mean(),
            band7_mean,
        )

        # append a multiband raster
        result = stack.append(Raster(nc.multiband), in_place=False)
        self.assertEqual(result.names[6], "landsat_multiband_1")
