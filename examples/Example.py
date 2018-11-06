import pyspatialml as ps
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
multiband = os.path.join(basedir, 'pyspatialml', 'tests', 'landsat_multiband.tif')
predictors = [band1, band2, band3, band4, band5, band7]

# Create a RasterStack instance
layer = ps.from_files(band1)
stack = ps.from_files([band1, band2, band3, band4, band5, band7])
stack.names
df = stack.to_pandas()

# Drop a layer
stack.drop(labels='lsat7_2000_70')
stack.names

# Add a layer
stack.append(ps.from_files(band7))
stack.names


# Load some training data in the form of a shapefile of point feature locations:
training_py = geopandas.read_file(os.path.join(basedir, 'pyspatialml', 'tests', 'landsat96_polygons.shp'))
training_pt = geopandas.read_file(os.path.join(basedir, 'pyspatialml', 'tests', 'landsat96_points.shp'))
training_px = rasterio.open(os.path.join(basedir, 'pyspatialml', 'tests', 'landsat96_labelled_pixels.tif'))
training_lines = deepcopy(training_py)
training_lines['geometry'] = training_lines.geometry.boundary

# Plot some training data
# plt.imshow(stack.lsat7_2000_70.read(masked=True),
#            extent=rasterio.plot.plotting_extent(stack.lsat7_2000_70.ds))
# plt.scatter(x=training_pt.bounds.iloc[:, 0],
#             y=training_pt.bounds.iloc[:, 1],
#             s=2, color='black')
# plt.show()

# Create a training dataset by extracting the raster values at the training point locations:
df_points = stack.extract_vector(response=training_pt, field='id')
df_polygons = stack.extract_vector(response=training_py, field='id')
df_lines = stack.extract_vector(response=training_lines, field='id')
df_raster = stack.extract_raster(response=training_px, value_name='id')
df_points.head()

layer.extract_vector(response=training_pt, field='id')

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
X = df_polygons.drop(columns=['id', 'geometry'])
y = df_polygons.id
lr.fit(X, y)

# spatial cross-validation
from sklearn.cluster import KMeans

# create 10 spatial clusters based on clustering of the training data point x,y coordinates
clusters = KMeans(n_clusters=34, n_jobs=-1)
clusters.fit(df_polygons.geometry.bounds.iloc[:, 0:2])

# cross validate
scores = cross_validate(
  lr, X, y, groups=clusters.labels_,
  scoring='accuracy',
  cv=3,  n_jobs=1)
scores['test_score'].mean()

# prediction
result = stack.predict(estimator=lr, dtype='int16', nodata=0)
result_prob = stack.predict(estimator=lr, predict_type='prob')

plt.imshow(result.read(masked=True)[0, :, :])
plt.show()

plt.imshow(result_prob.iloc[0].read(masked=True))
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
