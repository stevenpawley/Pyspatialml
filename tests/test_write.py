from unittest import TestCase
from tempfile import NamedTemporaryFile

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
