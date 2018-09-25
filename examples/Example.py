from pyspatialml import RasterStack
from copy import deepcopy
import os
import geopandas
import rasterio.plot
import matplotlib.pyplot as plt

# First, import the extract and predict functions:
basedir = os.getcwd()
os.chdir(os.path.join('.', 'examples'))
band1 = os.path.join(basedir, 'pyspatialml', 'tests', 'lsat7_2000_10.tif')
band2 = os.path.join(basedir, 'pyspatialml', 'tests', 'lsat7_2000_20.tif')
band3 = os.path.join(basedir, 'pyspatialml', 'tests', 'lsat7_2000_30.tif')
band4 = os.path.join(basedir, 'pyspatialml', 'tests', 'lsat7_2000_40.tif')
band5 = os.path.join(basedir, 'pyspatialml', 'tests', 'lsat7_2000_50.tif')
band7 = os.path.join(basedir, 'pyspatialml', 'tests', 'lsat7_2000_70.tif')
predictors = [band1, band2, band3, band4, band5, band7]

# create a RasterStack instance
stack = RasterStack(predictors)

# Load some training data in the form of a shapefile of point feature locations:
training_py = geopandas.read_file(os.path.join(basedir, 'pyspatialml', 'tests', 'landsat96_polygons.shp'))
training_pt = geopandas.read_file(os.path.join(basedir, 'pyspatialml', 'tests', 'landsat96_points.shp'))
training_px = rasterio.open(os.path.join(basedir, 'pyspatialml', 'tests', 'landsat96_labelled_pixels.tif'))
training_lines = deepcopy(training_py)
training_lines['geometry'] = training_lines.geometry.boundary

# Create a training dataset by extracting the raster values at the training point locations:
df = stack.extract_vector(response=training_pt, field='id')
df = stack.extract_vector(response=training_py, field='id')
df = stack.extract_vector(response=training_lines, field='id')
df = stack.extract_raster(response=training_px, value_name='id')

# Next we can train a logistic regression classifier:
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# define the classifier with standardization of the input features in a pipeline
lr = Pipeline(
    [('scaling', StandardScaler()),
     ('classifier', LogisticRegressionCV(n_jobs=-1))])

# fit the classifier
X = df.drop(columns=['id', 'geometry'])
y = df.id
lr.fit(X, y)

# spatial cross-validation
from sklearn.cluster import KMeans

# create 10 spatial clusters based on clustering of the training data point x,y coordinates
clusters = KMeans(n_clusters=10, n_jobs=-1)
clusters.fit(df.geometry.bounds.iloc[:, 0:2])

# cross validate
scores = cross_validate(
  lr, X, y, groups=clusters.labels_,
  scoring='accuracy',
  cv=3,  n_jobs=1)
scores['test_score'].mean()

# prediction
result = stack.predict(estimator=lr, dtype='int16', nodata=0)
plt.imshow(result.read(1, masked=True))
plt.show()

# sampling
# extract training data using a random sample
df_rand = stack.sample(size=1000, random_state=1)
df_rand.plot()

# extract training data using a stratified random sample from a map containing categorical data
# here we are taking 50 samples per category
strata = rasterio.open(os.path.join(basedir, 'pyspatialml', 'tests', 'strata.tif'))
df_strata = stack.sample(size=5, strata=strata, random_state=1)
df_strata = df_strata.dropna()

fig, ax = plt.subplots()
ax.imshow(strata.read(1, masked=True), extent=rasterio.plot.plotting_extent(strata))
df_strata.plot(ax=ax, markersize=2, color='red')
plt.show()
