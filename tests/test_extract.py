from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc_dataset as nc
from copy import deepcopy
import geopandas
import rasterio
import numpy as np
import os


class TestExtract(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_extract_points(self):

        stack = Raster(self.predictors)

        # extract training data from points
        training_pt = geopandas.read_file(nc.points)
        X, y, xy = stack.extract_vector(
            response=training_pt, field='id', return_array=True)

        # remove masked values
        mask2d = X.mask.any(axis=1)
        X = X[~mask2d]
        y = y[~mask2d]
        xy = xy[~mask2d]

        # check shapes of extracted pixels
        self.assertTupleEqual(X.shape, (562, 6))
        self.assertTupleEqual(y.shape, (562, ))
        self.assertTupleEqual(xy.shape, (562, 2))

        # check summarized values of extracted y values
        self.assertTrue(
            np.equal(np.bincount(y),
                     np.asarray([0, 161, 3, 76, 36, 275, 8, 3])).all()
        )

        # check extracted X values
        self.assertAlmostEqual(X[:, 0].mean(), 81.588968, places=2)
        self.assertAlmostEqual(X[:, 1].mean(), 67.619217, places=2)
        self.assertAlmostEqual(X[:, 2].mean(), 67.455516, places=2)
        self.assertAlmostEqual(X[:, 3].mean(), 69.153025, places=2)
        self.assertAlmostEqual(X[:, 4].mean(), 90.051601, places=2)
        self.assertAlmostEqual(X[:, 5].mean(), 59.558719, places=2)

    def test_extract_polygons(self):

        stack = Raster(self.predictors)

        # extract training data from polygons
        training_py = geopandas.read_file(nc.polygons)
        X, y, xy = stack.extract_vector(
            response=training_py, field='id', return_array=True)

        # remove masked values
        mask2d = X.mask.any(axis=1)
        X = X[~mask2d]
        y = y[~mask2d]
        xy = xy[~mask2d]

        # check shapes of extracted pixels
        self.assertTupleEqual(X.shape, (2436, 6))
        self.assertTupleEqual(y.shape, (2436, ))
        self.assertTupleEqual(xy.shape, (2436, 2))

    def test_extract_lines(self):

        stack = Raster(self.predictors)

        # extract training data from lines
        training_py = geopandas.read_file(nc.polygons)
        training_lines = deepcopy(training_py)
        training_lines['geometry'] = training_lines.geometry.boundary
        X, y, xy = stack.extract_vector(
            response=training_lines, field='id', return_array=True)

        # remove masked values
        mask2d = X.mask.any(axis=1)
        X = X[~mask2d]
        y = y[~mask2d]
        xy = xy[~mask2d]

        # check shapes of extracted pixels
        self.assertTupleEqual(X.shape, (948, 6))
        self.assertTupleEqual(y.shape, (948, ))
        self.assertTupleEqual(xy.shape, (948, 2))

    def test_extract_raster(self):

        stack = Raster(self.predictors)

        # extract training data from labelled pixels
        training_px = rasterio.open(nc.labelled_pixels)
        X, y, xy = stack.extract_raster(
            response=training_px, value_name='id', return_array=True)

        # remove masked values
        mask2d = X.mask.any(axis=1)
        X = X[~mask2d]
        y = y[~mask2d]
        xy = xy[~mask2d]

        # check shapes of extracted pixels
        self.assertTupleEqual(X.shape, (2436, 6))
        self.assertTupleEqual(y.shape, (2436, ))
        self.assertTupleEqual(xy.shape, (2436, 2))
