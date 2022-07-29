from unittest import TestCase

import numpy as np

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestSample(TestCase):
    def setUp(self) -> None:
        predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
        self.stack = Raster(predictors)
        self.strata = Raster(nc.strata)
    
    def tearDown(self) -> None:
        self.stack.close()
        self.strata.close()

    def test_sample_strata(self):
        # extract using a strata raster and returning two arrays
        size = 100
        categories = self.strata.read(masked=True).flatten()
        categories = categories[~categories.mask]
        n_categories = np.unique(categories).shape[0]
        n_samples = size * n_categories

        X, xy = self.stack.sample(size=size, strata=self.strata, return_array=True)
        self.assertEqual(X.shape, (n_samples, 6))
        self.assertEqual(xy.shape, (n_samples, 2))

        # extract using a strata raster and returning a dataframe
        samples = self.stack.sample(size=size, strata=self.strata, return_array=False)
        self.assertEqual(samples.shape, (n_samples, 7))
    
    def test_sample_no_strata(self):
        size = 100
        X, xy = self.stack.sample(size=size, return_array=True)
        self.assertEqual(X.shape, (size, 6))
        self.assertEqual(xy.shape, (size, 2))

        samples = self.stack.sample(size=size, return_array=False)
        self.assertEqual(samples.shape, (size, 7))