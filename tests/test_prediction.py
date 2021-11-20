from unittest import TestCase

import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import pyspatialml.datasets.meuse as ms
from pyspatialml import Raster
from pyspatialml.datasets import nc


class TestPrediction(TestCase):
    nc_predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack_nc = Raster(nc_predictors)
    stack_meuse = Raster(ms.predictors)

    def test_classification(self):
        training_pt = gpd.read_file(nc.points)
        df_points = self.stack_nc.extract_vector(gdf=training_pt)
        df_points["class_id"] = training_pt["id"].values
        df_points = df_points.dropna()

        clf = RandomForestClassifier(n_estimators=50)
        X = df_points.drop(columns=["class_id", "geometry"]).values
        y = df_points.class_id.values
        clf.fit(X, y)

        # classification
        cla = self.stack_nc.predict(estimator=clf, dtype="int16", nodata=0)
        self.assertIsInstance(cla, Raster)
        self.assertEqual(cla.count, 1)
        self.assertEqual(cla.read(masked=True).count(), 135092)

        # class probabilities
        probs = self.stack_nc.predict_proba(estimator=clf)
        self.assertIsInstance(cla, Raster)
        self.assertEqual(probs.count, 7)

        for layer in probs.values():
            self.assertEqual(layer.read(masked=True).count(), 135092)

    def test_classification_in_memory(self):
        training_pt = gpd.read_file(nc.points)
        df_points = self.stack_nc.extract_vector(gdf=training_pt)
        df_points["class_id"] = training_pt["id"].values
        df_points = df_points.dropna()

        clf = RandomForestClassifier(n_estimators=50)
        X = df_points.drop(columns=["class_id", "geometry"]).values
        y = df_points.class_id.values
        clf.fit(X, y)

        # classification
        cla = self.stack_nc.predict(estimator=clf, dtype="int16", nodata=0,
                                    in_memory=True)
        self.assertIsInstance(cla, Raster)
        self.assertEqual(cla.count, 1)
        self.assertEqual(cla.read(masked=True).count(), 135092)
        cla.close()

        # class probabilities
        probs = self.stack_nc.predict_proba(estimator=clf, in_memory=True)
        self.assertIsInstance(cla, Raster)
        self.assertEqual(probs.count, 7)

        for layer in probs.values():
            self.assertEqual(layer.read(masked=True).count(), 135092)

        probs.close()

    def test_regression(self):
        training_pt = gpd.read_file(ms.meuse)
        training = self.stack_meuse.extract_vector(gdf=training_pt)
        training["zinc"] = training_pt["zinc"].values
        training["cadmium"] = training_pt["cadmium"].values
        training["copper"] = training_pt["copper"].values
        training["lead"] = training_pt["lead"].values
        training = training.dropna()

        # single target regression
        regr = RandomForestRegressor(n_estimators=50)
        X = training.loc[:, self.stack_meuse.names].values
        y = training["zinc"].values
        regr.fit(X, y)

        single_regr = self.stack_meuse.predict(regr)
        self.assertIsInstance(single_regr, Raster)
        self.assertEqual(single_regr.count, 1)
        single_regr.close()

        # multi-target regression
        y = training.loc[:, ["zinc", "cadmium", "copper", "lead"]]
        regr.fit(X, y)
        multi_regr = self.stack_meuse.predict(regr)
        self.assertIsInstance(multi_regr, Raster)
        self.assertEqual(multi_regr.count, 4)
        multi_regr.close()

    def test_regression_in_memory(self):
        training_pt = gpd.read_file(ms.meuse)
        training = self.stack_meuse.extract_vector(gdf=training_pt)
        training["zinc"] = training_pt["zinc"].values
        training["cadmium"] = training_pt["cadmium"].values
        training["copper"] = training_pt["copper"].values
        training["lead"] = training_pt["lead"].values
        training = training.dropna()

        # single target regression
        regr = RandomForestRegressor(n_estimators=50)
        X = training.loc[:, self.stack_meuse.names].values
        y = training["zinc"].values
        regr.fit(X, y)

        single_regr = self.stack_meuse.predict(regr, in_memory=True)
        self.assertIsInstance(single_regr, Raster)
        self.assertEqual(single_regr.count, 1)
        single_regr.close()

        # multi-target regression
        y = training.loc[:, ["zinc", "cadmium", "copper", "lead"]]
        regr.fit(X, y)
        multi_regr = self.stack_meuse.predict(regr, in_memory=True)
        self.assertIsInstance(multi_regr, Raster)
        self.assertEqual(multi_regr.count, 4)
        multi_regr.close()

    def test_classification_with_single_constant(self):
        training_pt = gpd.read_file(nc.points)
        df_points = self.stack_nc.extract_vector(gdf=training_pt)
        df_points["class_id"] = training_pt["id"].values
        df_points = df_points.dropna()

        # classification with a single constant
        df_points["constant"] = 1

        clf = RandomForestClassifier(n_estimators=50)
        X = df_points.drop(columns=["class_id", "geometry"]).values
        y = df_points.class_id.values
        clf.fit(X, y)

        cla = self.stack_nc.predict(
            estimator=clf, 
            dtype="int16", 
            nodata=0,
            constants=[1]
        )
        self.assertIsInstance(cla, Raster)
        self.assertEqual(cla.count, 1)
        self.assertEqual(cla.read(masked=True).count(), 135092)

        probs = self.stack_nc.predict_proba(estimator=clf, constants=[1])
        self.assertIsInstance(cla, Raster)
        self.assertEqual(probs.count, 7)

        for layer in probs.values():
            self.assertEqual(layer.read(masked=True).count(), 135092)

    def test_classification_with_list_constants(self):
        training_pt = gpd.read_file(nc.points)
        df_points = self.stack_nc.extract_vector(gdf=training_pt)
        df_points["class_id"] = training_pt["id"].values
        df_points = df_points.dropna()

        df_points["constant"] = 1
        df_points["constant2"] = 2

        clf = RandomForestClassifier(n_estimators=50)
        X = df_points.drop(columns=["class_id", "geometry"]).values
        y = df_points.class_id.values
        clf.fit(X, y)

        cla = self.stack_nc.predict(
            estimator=clf, 
            dtype="int16", 
            nodata=0,
            constants=[1, 2]
        )
        self.assertIsInstance(cla, Raster)
        self.assertEqual(cla.count, 1)
        self.assertEqual(cla.read(masked=True).count(), 135092)

        probs = self.stack_nc.predict_proba(estimator=clf, constants=[1, 2])
        self.assertIsInstance(cla, Raster)
        self.assertEqual(probs.count, 7)

        for layer in probs.values():
            self.assertEqual(layer.read(masked=True).count(), 135092)

    def test_classification_with_dict_constants(self):
        # classification using constant to replace an existing layer
        training_pt = gpd.read_file(nc.points)
        df_points = self.stack_nc.extract_vector(gdf=training_pt)
        df_points["class_id"] = training_pt["id"].values
        df_points = df_points.dropna()

        clf = RandomForestClassifier(n_estimators=50)
        X = df_points.drop(columns=["class_id", "geometry"]).values
        y = df_points.class_id.values
        clf.fit(X, y)

        cla = self.stack_nc.predict(
            estimator=clf, 
            dtype="int16", 
            nodata=0,
            constants={"lsat7_2000_10": 150}
        )
        self.assertIsInstance(cla, Raster)
        self.assertEqual(cla.count, 1)
        self.assertEqual(cla.read(masked=True).count(), 135092)

        probs = self.stack_nc.predict_proba(estimator=clf, constants={"lsat7_2000_10": 150})
        self.assertIsInstance(cla, Raster)
        self.assertEqual(probs.count, 7)

        for layer in probs.values():
            self.assertEqual(layer.read(masked=True).count(), 135092)
