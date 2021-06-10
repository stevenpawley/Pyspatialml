from unittest import TestCase

import numpy as np

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestCalc(TestCase):
    def setUp(self) -> None:
        predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
        self.stack = Raster(predictors)
        self.result = None

    def tearDown(self) -> None:
        self.stack.close()
        self.result.close()
        self.stack = None
        self.result = None

    def test_calc_with_2d_output(self):
        def compute_outputs_2d_array(arr):
            return arr[0, :, :] + arr[1, :, :]

        self.result = self.stack.apply(compute_outputs_2d_array)
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, 1)
        self.assertEqual(self.result.read(masked=True).count(), 183418)

    def test_calc_with_2d_output_coerce_dtype(self):
        def compute_outputs_2d_array(arr):
            return arr[0, :, :] + arr[1, :, :]

        self.result = self.stack.apply(compute_outputs_2d_array, dtype=np.int16)
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, 1)
        self.assertEqual(self.result.read(masked=True).count(), 183418)

    def test_calc_with_3d_output(self):
        def compute_outputs_3d_array(arr):
            arr[0, :, :] = arr[0, :, :] + arr[1, ::]
            return arr

        self.result = self.stack.apply(compute_outputs_3d_array)
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, 6)
        self.assertEqual(self.result.read(masked=True).count(), 1052182)

    def test_calc_with_multiprocessing(self):
        def compute_outputs_2d_array(arr):
            return arr[0, :, :] + arr[1, :, :]

        self.result = self.stack.apply(compute_outputs_2d_array)
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, 1)
        self.assertEqual(self.result.read(masked=True).count(), 183418)

    def test_calc_in_memory(self):
        def compute_outputs_2d_array(arr):
            return arr[0, :, :] + arr[1, :, :]

        self.result = self.stack.apply(compute_outputs_2d_array, in_memory=True)
        self.assertIsInstance(self.result, Raster)
        self.assertEqual(self.result.count, 1)
        self.assertEqual(self.result.read(masked=True).count(), 183418)
