from unittest import TestCase

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster
import rasterio
import numpy as np
import os
from tempfile import NamedTemporaryFile


class TestRename(TestCase):
    def setUp(self) -> None:
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]

    def test_rename_inplace(self):
        stack = Raster(self.predictors)
        band3_stats = stack.lsat7_2000_30.mean()

        # rename band 3
        stack.rename(names={"lsat7_2000_30": "new_name"}, in_place=True)

        # check that renaming occurred in Raster
        self.assertEqual(list(stack.names)[2], "new_name")
        self.assertNotIn("lsat7_2000_30", stack.names)

        # check that Raster layer properties also renamed
        self.assertIn("new_name", dir(stack))
        self.assertNotIn("lsat7_2000_30", dir(stack))

        # check that internal name of RasterLayer was also renamed
        self.assertEqual(stack.iloc[2].name, "new_name")

        # check that the RasterLayer attached to the new name is the same
        self.assertEqual(stack["new_name"].mean(), band3_stats)
        self.assertEqual(stack.new_name.mean(), band3_stats)
        self.assertEqual(stack.iloc[2].mean(), band3_stats)

        # check that a new Raster object derived from the renamed data
        # have the right names
        new_raster = Raster(src=stack.iloc[2])
        self.assertIn("new_name", new_raster.names)

    def test_rename_with_copy(self):
        stack = Raster(self.predictors)
        names = list(stack.names)
        band3_stats = stack.lsat7_2000_30.mean()

        # rename band 3
        result = stack.rename(names={"lsat7_2000_30": "new_name"},
                              in_place=False)

        # check that original is untouched
        self.assertEqual(list(stack.names), names)

        # check that renaming occurred in Raster
        self.assertEqual(list(result.names)[2], "new_name")
        self.assertNotIn("lsat7_2000_30", result.names)

        # check that Raster layer properties also renamed
        self.assertIn("new_name", dir(result))
        self.assertNotIn("lsat7_2000_30", dir(result))

        # check that internal name of RasterLayer was also renamed
        self.assertEqual(result.iloc[2].name, "new_name")

        # check that the RasterLayer attached to the new name is the same
        self.assertEqual(result["new_name"].mean(), band3_stats)
        self.assertEqual(result["new_name"].mean(), band3_stats)
        self.assertEqual(result.new_name.mean(), band3_stats)
        self.assertEqual(result.iloc[2].mean(), band3_stats)

        # check that a new Raster object derived from the renamed data
        # have the right names
        new_raster = Raster(src=result.iloc[2])
        self.assertIn("new_name", new_raster.names)

    def rename_multiband(self):
        # Create a fake 3-band image for testing
        arr = np.random.rand(3, 64, 64)
        file = NamedTemporaryFile(prefix="test", suffix=".tif").name
        layer_name = os.path.basename(file).split(".")[0]
        layer_names = ["_".join([layer_name, str(i)]) for i in [1, 2 ,3]]

        with rasterio.open(file, "w", width=64, height=64, count=3, dtype=np.float32) as dst:
            dst.write(arr)

        r = Raster(file)
        self.assertListEqual(list(r.names), layer_names)

        renamed = r.rename(dict(zip(r.names, ["Red", "Green", "Blue"])))
        self.assertListEqual(list(renamed.names), ["Red", "Green", "Blue"])

    def rename_in_memory(self):
        # Create a fake 3-band image for testing
        arr = np.random.rand(3, 64, 64)
        file = NamedTemporaryFile(prefix="test", suffix=".tif").name

        with rasterio.open(file, "w", width=64, height=64, count=3, dtype=np.float32) as dst:
            dst.write(arr)

        r = Raster(file)
        in_memory = r.aggregate((32, 32), in_memory=True)

        renamed = r.rename(dict(zip(in_memory.names, ["Red", "Green", "Blue"])))
        self.assertListEqual(list(renamed.names), ["Red", "Green", "Blue"])
