from unittest import TestCase
from pyspatialml import Raster
from pyspatialml.datasets import nc
import pyspatialml.datasets.meuse as ms
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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

        for _, layer in probs:
            self.assertEqual(layer.read(masked=True).count(), 135092)

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

        # multi-target regression
        y = training.loc[:, ["zinc", "cadmium", "copper", "lead"]]
        regr.fit(X, y)
        multi_regr = self.stack_meuse.predict(regr)
        self.assertIsInstance(multi_regr, Raster)
        self.assertEqual(multi_regr.count, 4)
