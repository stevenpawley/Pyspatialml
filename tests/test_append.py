from unittest import TestCase

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestAppend(TestCase):
    def setUp(self) -> None:
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]

    def test_append_inplace(self):
        """Append another Raster containing a single layer with identical name

        This test should cause the Raster object to automatically rename the
        duplicated names as "lsat7_2000_70_1", "lsat7_2000_70_2", etc.

        Appending a multi-band raster should result in a new layer with the
        multi-band name "landsat_multiband_1", "landsat_multiband_2", etc.

        A list of Rasters can be passed to append() to append multiple rasters
        """
        # append a single band raster with the same name
        stack = Raster(self.predictors)
        band7_mean = stack["lsat7_2000_70"].read(masked=True).mean()
        stack.append(Raster(nc.band7), in_place=True)

        self.assertEqual(list(stack.names)[5], "lsat7_2000_70_1")
        self.assertEqual(list(stack.names)[-1], "lsat7_2000_70_2")
        self.assertEqual(
            stack.lsat7_2000_70_1.read(masked=True).mean(),
            stack.lsat7_2000_70_2.read(masked=True).mean(),
            band7_mean,
        )

        # append a multiband raster
        stack = Raster(self.predictors)
        stack.append(Raster(nc.multiband), in_place=True)
        self.assertEqual(list(stack.names)[6], "landsat_multiband_1")
        stack.close()

        # append multiple rasters
        stack = Raster(self.predictors)
        stack.append([Raster(nc.band5), Raster(nc.band7)], in_place=True)
        self.assertEqual(stack.count, 8)

    def test_append_with_copy(self):
        """Same tests as above but create a new Raster rather than append
        in place
        """
        # append another Raster containing a single layer with identical name
        stack = Raster(self.predictors)
        band7_mean = stack["lsat7_2000_70"].read(masked=True).mean()
        result = stack.append(Raster(nc.band7), in_place=False)

        # check that original is untouched
        self.assertEqual(stack.count, 6)

        # check that result contains appended raster
        self.assertEqual(list(result.names)[5], "lsat7_2000_70_1")
        self.assertEqual(list(result.names)[-1], "lsat7_2000_70_2")

        # check that band 7 stats are the same after appending
        self.assertEqual(
            result.lsat7_2000_70_1.read(masked=True).mean(),
            result.lsat7_2000_70_2.read(masked=True).mean(),
            band7_mean,
        )

        # append a multiband raster
        result = stack.append(Raster(nc.multiband), in_place=False)
        self.assertEqual(list(result.names)[6], "landsat_multiband_1")
        stack.close()

        # append multiple rasters
        stack = Raster(self.predictors)
        new_stack = stack.append([Raster(nc.band5), Raster(nc.band7)], in_place=False)
        self.assertEqual(new_stack.count, 8)
