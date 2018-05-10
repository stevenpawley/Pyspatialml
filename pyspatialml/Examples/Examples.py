from osgeo import gdal
from pyspatialml.sampling import extract
from pyspatialml.spectral import indices
from pyspatialml import predict
import geopandas as gpd
import rasterio
import os
import matplotlib.pyplot as plt

# project base directory
projectdir = '/Users/steven/Github/pyspatialml/tests'

# raster data
band1 = os.path.join(projectdir, 'lsat7_2000_10.tif')
band2 = os.path.join(projectdir, 'lsat7_2000_20.tif')
band3 = os.path.join(projectdir, 'lsat7_2000_30.tif')
band4 = os.path.join(projectdir, 'lsat7_2000_40.tif')
band5 = os.path.join(projectdir, 'lsat7_2000_50.tif')
band7 = os.path.join(projectdir, 'lsat7_2000_70.tif')

# vector data
training_points = os.path.join(projectdir, 'landclass96_roi.shp')

# prepare virtual raster
predictors = [band1, band2, band3, band4, band5, band7]
vrt_file = os.path.join(projectdir, 'landsat.vrt')
outds = gdal.BuildVRT(
    destName=vrt_file, srcDSOrSrcDSTab=predictors, separate=True,
    resolution='highest', resampleAlg='bilinear')
outds.FlushCache()

# read vector data
training_gpd = gpd.read_file(training_points)

# calculate spectral indices
spec_ind = indices(src=vrt_file, blue=0, green=1, red=2, nir=3, swir2=4, swir3=5)
ndvi = spec_ind.ndvi()

# plotting
bounds = rasterio.open(vrt_file).bounds
plt.imshow(ndvi, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
plt.scatter(x=training_gpd.bounds.iloc[:, 0], y=training_gpd.bounds.iloc[:, 1],
            s=2, color='black')
plt.show()

# extract training data
X, y, xy = extract(raster=vrt_file, response_gdf=training_gpd, field='id')

# classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate
clf = ExtraTreesClassifier(n_estimators=1000)
clf.fit(X, y)
scores = cross_validate(clf, X, y, cv=3, scoring=['accuracy'])
scores['test_accuracy'].mean()

predict(estimator=clf, raster=vrt_file, file_path='/Users/steven/Github/pyspatialml/tests/classification.tif')