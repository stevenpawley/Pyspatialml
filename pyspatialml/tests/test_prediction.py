from unittest import TestCase
from pyspatialml import Raster
import os
import sys
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
test_dir = os.path.dirname(__file__)
pkg_dir = os.path.join(test_dir, os.path.pardir)
nc_dir = os.path.join(pkg_dir, 'nc_dataset')
meuse_dir = os.path.join(pkg_dir, 'meuse_dataset')

class TestPrediction(TestCase):

    band1 = os.path.join(nc_dir, 'lsat7_2000_10.tif')
    band2 = os.path.join(nc_dir, 'lsat7_2000_20.tif')
    band3 = os.path.join(nc_dir, 'lsat7_2000_30.tif')
    band4 = os.path.join(nc_dir, 'lsat7_2000_40.tif')
    band5 = os.path.join(nc_dir, 'lsat7_2000_50.tif')
    band7 = os.path.join(nc_dir, 'lsat7_2000_70.tif')
    predictors = [band1, band2, band3, band4, band5, band7]

    def test_classification(self):
        stack = Raster(self.predictors)
        training_pt = gpd.read_file(
            os.path.join(nc_dir, 'landsat96_points.shp'))
        
        df_points = stack.extract_vector(response=training_pt, field='id')

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

        for name, layer in probs:
            self.assertEquals(layer.read(masked=True).count(), 135092)

    def test_regression(self):

        meuse_predictors = os.listdir(meuse_dir)
        meuse_predictors = [os.path.join(meuse_dir, i) for i in meuse_predictors if i.endswith('.tif')]
        stack = Raster(meuse_predictors)
        self.assertEqual(stack.count, 21)
        
        training_pt = gpd.read_file(
            os.path.join(meuse_dir, 'meuse.shp'))
        training = stack.extract_vector(
            response=training_pt, field='cadmium')
        training['copper'] = stack.extract_vector(
            response=training_pt, field='copper')['copper']
        training['lead'] = stack.extract_vector(
            response=training_pt, field='lead')['lead']
        training['zinc'] = stack.extract_vector(
            response=training_pt, field='zinc')['zinc']

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




        
