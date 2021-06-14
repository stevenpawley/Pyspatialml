from unittest import TestCase

from pyspatialml import Raster, RasterLayer
from pyspatialml.datasets import nc


class TestIndexing(TestCase):
    def setUp(self) -> None:
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_naming(self):
        stack = Raster(self.predictors + [nc.multiband])

        # check unique naming when stacking multiband raster
        self.assertEqual(stack.count, 11)
        expected_names = [
            "lsat7_2000_10",
            "lsat7_2000_20",
            "lsat7_2000_30",
            "lsat7_2000_40",
            "lsat7_2000_50",
            "lsat7_2000_70",
            "landsat_multiband_1",
            "landsat_multiband_2",
            "landsat_multiband_3",
            "landsat_multiband_4",
            "landsat_multiband_5",
        ]
        self.assertListEqual(list(stack.names), expected_names)
        stack.close()

    def test_subset_single_layer(self):
        stack = Raster(self.predictors + [nc.multiband])

        # Subset a single layer using an index position - returns a RasterLayer
        self.assertIsInstance(stack.iloc[0], RasterLayer)

        # Subset a single layer using a label - returns a RasterLayer
        self.assertIsInstance(stack["lsat7_2000_10"], RasterLayer)

        # Subset a single layer using an attribute - returns a RasterLayer
        self.assertIsInstance(stack.lsat7_2000_10, RasterLayer)

        # Check that the raster values are the same as the original values
        # after subsetting
        self.assertEqual(
            stack.lsat7_2000_10.read(masked=True).mean(),
            80.56715262406088
        )
        self.assertEqual(
            stack.lsat7_2000_70.read(masked=True).mean(),
            59.17773813401238
        )
        stack.close()

    def test_subset_multiple_layers(self):
        stack = Raster(self.predictors + [nc.multiband])

        # Subset multiple layers using a slice of index positions
        # - returns a Raster object
        self.assertIsInstance(stack.iloc[0:2], Raster)

        # Subset multiple layers using a list of index positions
        # - returns a Raster object
        self.assertIsInstance(stack.iloc[[0, 1, 2]], Raster)

        # Subset multiple layers using a list of labels
        # - returns a Raster object
        subset_raster = stack[["lsat7_2000_10", "lsat7_2000_70"]]
        self.assertIsInstance(subset_raster, Raster)
        self.assertListEqual(
            list(subset_raster.names),
            ["lsat7_2000_10", "lsat7_2000_70"]
        )

        # Check that label and integer subset return the same layers
        self.assertListEqual(
            list(stack.iloc[0:3].names),
            list(stack[["lsat7_2000_10", "lsat7_2000_20", "lsat7_2000_30"]].names),
        )

        stack.close()

    def test_indexing(self):
        stack = Raster(self.predictors + [nc.multiband])

        # replace band 1 with band 7
        band7_mean = stack["lsat7_2000_70"].read(masked=True).mean()

        stack.iloc[0] = Raster(nc.band7).iloc[0]

        self.assertEqual(stack.iloc[0].read(masked=True).mean(), band7_mean)
        self.assertEqual(stack["lsat7_2000_10"].read(masked=True).mean(),
                         band7_mean)
        self.assertEqual(stack["lsat7_2000_10"].read(masked=True).mean(),
                         band7_mean)
        self.assertEqual(stack.lsat7_2000_10.read(masked=True).mean(),
                         band7_mean)

        stack.close()
