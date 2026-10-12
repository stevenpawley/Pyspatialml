from unittest import TestCase
from tempfile import NamedTemporaryFile

import numpy as np

from pyspatialml import Raster
from pyspatialml.datasets import nc


class TestWrite(TestCase):
    def setUp(self) -> None:
        # inputs
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                           nc.band7]

        # test results
        self.stack = None

    def tearDown(self) -> None:
        self.stack.close()

    def test_write(self):
        # test writing to file
        self.stack = Raster(self.predictors)
        fp = NamedTemporaryFile(suffix=".tif").name

        result = self.stack.write(fp)

        self.assertIsInstance(result, Raster)
        self.assertEqual(result.count, self.stack.count)

    def test_write_uses_windows(self):
        self.stack = Raster(self.predictors)
        self.stack.block_shape = (64, 64)
        fp = NamedTemporaryFile(suffix=".tif").name

        result = self.stack.write(fp)

        self.assertTrue(
            np.ma.allequal(
                result.read(masked=True), self.stack.read(masked=True)
            )
        )
        result.close()
