import numpy as np
import unittest
import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestArith(unittest.TestCase):
    def setUp(self) -> None:
        arr = np.zeros((3, 100, 100))
        arr[1,:,:] = 1
        arr[2,:,:] = 2
        self.obj = Raster(arr)
        self.obj.names = ["band1", "band2", "band3"]
    
    def test_scalar(self):
        addition = self.obj.iloc[0] + 100
        self.assertEqual(addition.min(), 100)

        division = addition / 10
        self.assertEqual(division.min(), 10.0)

        multiplication = self.obj.iloc[1] * 100
        self.assertEqual(multiplication.min(), 100)
    
    def test_rasterlayer(self):
        addition = self.obj.iloc[0] + self.obj.iloc[1]
        self.assertEqual(addition.min(), 1.0)

        multiplication = self.obj.iloc[1] * self.obj.iloc[2]
        self.assertEqual(multiplication.min(), 2)
    
    def test_raster(self):
        pass
