from unittest import TestCase
from pyspatialml import Raster
from copy import deepcopy
import os
import geopandas
import rasterio
import numpy as np

THIS_DIR = os.path.abspath(__file__)

class TestExtract(TestCase):

    band1 = os.path.join(THIS_DIR, 'lsat7_2000_10.tif')
    band2 = os.path.join(THIS_DIR, 'lsat7_2000_20.tif')
    band3 = os.path.join(THIS_DIR, 'lsat7_2000_30.tif')
    band4 = os.path.join(THIS_DIR, 'lsat7_2000_40.tif')
    band5 = os.path.join(THIS_DIR, 'lsat7_2000_50.tif')
    band7 = os.path.join(THIS_DIR, 'lsat7_2000_70.tif')
    predictors = [band1, band2, band3, band4, band5, band7]

    def test_extract_points(self):

        stack = Raster(self.predictors)

        # extract training data from points
        training_pt = geopandas.read_file(os.path.join(THIS_DIR, 'landsat96_points.shp'))
        X, y, xy = stack.extract_vector(response=training_pt, field='id', return_array=True)

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
        training_py = geopandas.read_file(os.path.join(THIS_DIR, 'landsat96_polygons.shp'))
        X, y, xy = stack.extract_vector(response=training_py, field='id', return_array=True)

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
        training_py = geopandas.read_file(os.path.join(THIS_DIR, 'landsat96_polygons.shp'))
        training_lines = deepcopy(training_py)
        training_lines['geometry'] = training_lines.geometry.boundary
        X, y, xy = stack.extract_vector(response=training_lines, field='id', return_array=True)

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
        training_px = rasterio.open(os.path.join(THIS_DIR, 'landsat96_labelled_pixels.tif'))
        X, y, xy = stack.extract_raster(response=training_px, value_name='id', return_array=True)

        # remove masked values
        mask2d = X.mask.any(axis=1)
        X = X[~mask2d]
        y = y[~mask2d]
        xy = xy[~mask2d]

        # check shapes of extracted pixels
        self.assertTupleEqual(X.shape, (2436, 6))
        self.assertTupleEqual(y.shape, (2436, ))
        self.assertTupleEqual(xy.shape, (2436, 2))
