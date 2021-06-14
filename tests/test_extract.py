from copy import deepcopy
from unittest import TestCase

import geopandas
import pandas as pd
import rasterio

from pyspatialml import Raster
from pyspatialml.datasets import nc


class TestExtract(TestCase):
    def setUp(self) -> None:
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
        self.extracted_grass = pd.read_table(nc.extracted_pixels, delimiter=" ")

        self.stack = Raster(self.predictors)

    def tearDown(self) -> None:
        self.stack.close()

    def test_extract_points(self):
        training_pt = geopandas.read_file(nc.points)

        # check that extracted training data as a DataFrame match known values
        df = self.stack.extract_vector(gdf=training_pt)
        df = df.dropna()
        training_pt = training_pt.dropna()

        self.assertTrue(
            (df["lsat7_2000_10"].values == training_pt["b1"].values).all()
        )
        self.assertTrue(
            (df["lsat7_2000_20"].values == training_pt["b2"].values).all()
        )
        self.assertTrue(
            (df["lsat7_2000_30"].values == training_pt["b3"].values).all()
        )
        self.assertTrue(
            (df["lsat7_2000_40"].values == training_pt["b4"].values).all()
        )
        self.assertTrue(
            (df["lsat7_2000_50"].values == training_pt["b5"].values).all()
        )
        self.assertTrue(
            (df["lsat7_2000_70"].values == training_pt["b7"].values).all()
        )

    def test_extract_polygons(self):
        # extract training data from polygons
        training_py = geopandas.read_file(nc.polygons)
        df = self.stack.extract_vector(gdf=training_py)
        df = df.dropna()

        df = df.merge(
            right=training_py.loc[:, ("id", "label")],
            left_on="geometry_idx",
            right_on="index",
            right_index=True,
        )

        # compare to extracted data using GRASS GIS
        self.assertEqual(df.shape[0], self.extracted_grass.shape[0])
        self.assertAlmostEqual(
            df["lsat7_2000_10"].mean(), self.extracted_grass["b1"].mean(),
            places=2
        )
        self.assertAlmostEqual(
            df["lsat7_2000_20"].mean(), self.extracted_grass["b2"].mean(),
            places=2
        )
        self.assertAlmostEqual(
            df["lsat7_2000_30"].mean(), self.extracted_grass["b3"].mean(),
            places=2
        )
        self.assertAlmostEqual(
            df["lsat7_2000_40"].mean(), self.extracted_grass["b4"].mean(),
            places=2
        )
        self.assertAlmostEqual(
            df["lsat7_2000_50"].mean(), self.extracted_grass["b5"].mean(),
            places=2
        )
        self.assertAlmostEqual(
            df["lsat7_2000_70"].mean(), self.extracted_grass["b7"].mean(),
            places=2
        )

    def test_extract_lines(self):
        # extract training data from lines
        training_py = geopandas.read_file(nc.polygons)
        training_lines = deepcopy(training_py)
        training_lines["geometry"] = training_lines.geometry.boundary
        df = self.stack.extract_vector(gdf=training_lines).dropna()

        # check shapes of extracted pixels
        self.assertEqual(df.shape[0], 948)

    def test_extract_raster(self):
        # extract training data from labelled pixels
        with rasterio.open(nc.labelled_pixels) as src:
            df = self.stack.extract_raster(src)

        df = df.dropna()

        self.assertEqual(df.shape[0], self.extracted_grass.shape[0])
        self.assertAlmostEqual(
            df["lsat7_2000_10"].mean(), self.extracted_grass["b1"].mean(),
            places=3
        )
        self.assertAlmostEqual(
            df["lsat7_2000_20"].mean(), self.extracted_grass["b2"].mean(),
            places=3
        )
        self.assertAlmostEqual(
            df["lsat7_2000_30"].mean(), self.extracted_grass["b3"].mean(),
            places=3
        )
        self.assertAlmostEqual(
            df["lsat7_2000_40"].mean(), self.extracted_grass["b4"].mean(),
            places=3
        )
        self.assertAlmostEqual(
            df["lsat7_2000_50"].mean(), self.extracted_grass["b5"].mean(),
            places=3
        )
        self.assertAlmostEqual(
            df["lsat7_2000_70"].mean(), self.extracted_grass["b7"].mean(),
            places=3
        )
