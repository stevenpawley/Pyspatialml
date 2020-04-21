from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc
import numpy as np


class TestCalc(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    def test_calc_with_2d_output(self):
        def compute_outputs_2d_array(arr):
            new_arr = arr[0, :, :] + arr[1, :, :]
            return new_arr

        calculation = self.stack.apply(compute_outputs_2d_array, n_jobs=1)

        self.assertIsInstance(calculation, Raster)
        self.assertEqual(calculation.count, 1)
        self.assertEqual(calculation.read(masked=True).count(), 183418)

    def test_calc_with_2d_output_coerce_dtype(self):
        def compute_outputs_2d_array(arr):
            new_arr = arr[0, :, :] + arr[1, :, :]
            return new_arr

        calculation = self.stack.apply(
            compute_outputs_2d_array, dtype=np.int16, n_jobs=1
        )

        self.assertIsInstance(calculation, Raster)
        self.assertEqual(calculation.count, 1)
        self.assertEqual(calculation.read(masked=True).count(), 183418)

        # test that nodata value is properly recognised
        self.assertEqual(calculation.read(masked=True).min(), 94)
        self.assertEqual(calculation.read(masked=True).max(), 510)

    def test_calc_with_3d_output(self):
        def compute_outputs_3d_array(arr):
            arr[0, :, :] = arr[0, :, :] + arr[1, ::]
            return arr

        calculation = self.stack.apply(compute_outputs_3d_array, n_jobs=1)

        self.assertIsInstance(calculation, Raster)
        self.assertEqual(calculation.count, 6)
        self.assertEqual(calculation.read(masked=True).count(), 1052182)

    def test_calc_with_multiprocessing(self):
        def compute_outputs_2d_array(arr):
            new_arr = arr[0, :, :] + arr[1, :, :]
            return new_arr

        calculation = self.stack.apply(compute_outputs_2d_array, n_jobs=-1)

        self.assertIsInstance(calculation, Raster)
        self.assertEqual(calculation.count, 1)
        self.assertEqual(calculation.read(masked=True).count(), 183418)
