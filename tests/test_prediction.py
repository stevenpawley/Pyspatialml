from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc_dataset as nc
import pyspatialml.datasets.meuse_dataset as ms
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TestPrediction(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_classification(self):
        print(self.predictors)
        stack = Raster(self.predictors)
        training_pt = gpd.read_file(nc.points)
        
        df_points = stack.extract_vector(response=training_pt, columns='id')

        clf = RandomForestClassifier(n_estimators=50)
        X = df_points.drop(columns=['id', 'geometry'])
        y = df_points.id
        clf.fit(X, y)

        # classification
        cla = stack.predict(estimator=clf, dtype='int16', nodata=0)
        self.assertIsInstance(cla, Raster)
        self.assertEquals(cla.count, 1)
        self.assertEquals(cla.read(masked=True).count(), 135092)

        # class probabilities
        probs = stack.predict_proba(estimator=clf)
        self.assertIsInstance(cla, Raster)
        self.assertEquals(probs.count, 7)

        for _, layer in probs:
            self.assertEquals(layer.read(masked=True).count(), 135092)

    def test_regression(self):

        stack = Raster(ms.predictors)
        
        training_pt = gpd.read_file(ms.meuse)
        training = stack.extract_vector(
            response=training_pt, columns=['cadmium', 'copper', 'lead', 'zinc'])

        # single target regression
        regr = RandomForestRegressor(n_estimators=50)
        X = training.loc[:, stack.names]
        y = training['zinc']
        regr.fit(X, y)

        single_regr = stack.predict(regr)
        self.assertIsInstance(single_regr, Raster)
        self.assertEqual(single_regr.count, 1)

        # multi-target regression
        y = training.loc[:, ['zinc', 'cadmium', 'copper', 'lead']]
        regr.fit(X, y)
        multi_regr = stack.predict(regr)
        self.assertIsInstance(multi_regr, Raster)
        self.assertEqual(multi_regr.count, 4)
