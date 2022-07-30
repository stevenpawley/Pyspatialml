from unittest import TestCase

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster
import geopandas as gpd
from sklearn.preprocessing import StandardScaler


class TestAlter(TestCase):
    def setUp(self) -> None:
        predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
        self.stack = Raster(predictors)
        points = gpd.read_file(nc.points)
        data = self.stack.extract_vector(points)
        self.data = data.dropna()

    def tearDown(self) -> None:
        self.stack.close()

    def test_alter(self):
        scaler = StandardScaler()
        scaler.fit(self.data.drop(columns=["geometry"]).values)
        out = self.stack.alter(scaler)

        self.assertIsInstance(out, Raster)
        self.assertEqual(out.shape, self.stack.shape)
