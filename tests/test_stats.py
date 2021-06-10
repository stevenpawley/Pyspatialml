import numpy as np
import unittest
import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestStats(unittest.TestCase):
    def setUp(self) -> None:
        predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
        self.predictors = predictors
        self.stack = Raster(predictors)

    def test_rasterstats(self):
        self.assertEqual(len(self.stack.min()), len(self.predictors))
        self.assertTrue(~np.isnan(self.stack.min()).all())

        self.assertEqual(len(self.stack.max()), len(self.predictors))
        self.assertTrue(~np.isnan(self.stack.max()).all())

        self.assertEqual(len(self.stack.mean()), len(self.predictors))
        self.assertTrue(~np.isnan(self.stack.mean()).all())

        self.assertEqual(len(self.stack.median()), len(self.predictors))
        self.assertTrue(~np.isnan(self.stack.median()).all())

    def test_layerstats(self):
        self.assertEqual(self.stack.iloc[0].min(), 56.0)
        self.assertEqual(self.stack.iloc[0].max(), 255.0)
        self.assertAlmostEqual(self.stack.iloc[0].mean(), 80.6, places=0)
        self.assertEqual(self.stack.iloc[0].median(), 75.0)

if __name__ == '__main__':
    unittest.main()
