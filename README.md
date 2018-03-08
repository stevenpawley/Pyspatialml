# Pyspatialml
Machine learning classification and regresssion modelling for spatial raster data.

## Description
Pyspatialml provides a suite of simple functions to enable machine learning classification and regression models to be applied to raster data. The package relies on the rasterio and GDAL packages for accessing raster data, and geopandas/shapely for vector data.

## Background

A supervised machine-learning workflow as applied to spatial raster data involves several steps:
1. Extracting training data.
2. Developing a machine learning classifier or regressor model. Pyspatialml uses scikit-learn for this purpose.
3. Performing the prediction on the raster data.

Training data consists of two components - a response feature and a set of predictors. In spatial data, the response feature is often represented by a locations when some property/state/concentration is already established. This data can be represented by point locations (e.g. arsenic concentrations soil samples), pixel locations where the pixel value represents the target of interest, or polygon features (e.g. labelled with landcover type). The predictors are represented by raster data, which contain variables that that in part may explaining the spatial distribution of the response variable (e.g., raster data representing soil types, soil properties, climatic data etc.).

![example](https://github.com/stevenpawley/Pyspatialml/blob/master/img/Pyspatialml_training.svg)

## Installation
```
pip install git+https://github.com/stevenpawley/Pyspatialml
```

## Usage

### Basic workflow

Import the extract and predict functions:
```
from pyspatialml.sampling import extract
from pyspatialml import predict
```

Stack a series of separate rasters as a virtual tile format raster:
```
from osgeo import gdal

predictor_files = [
  'raster1.tif',
  'raster2.tif',
  'raster3'.tif'
  ]

predictors = 'stack.vrt'

outds = gdal.BuildVRT(
    destName=predictors, srcDSOrSrcDSTab=predictor_files, separate=True,
    resolution='highest', resampleAlg='bilinear',
    outputBounds=(xmin, ymin, xmax, ymax),
    allowProjectionDifference=True,
    outputSRS='+proj=tmerc +lat_0=0 +lon_0=-115 +k=0.9992 +x_0=500000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
outds.FlushCache()
```

Load some training data in the form of a shapefile of point feature locations and extract the pixel values of the predictors:

```
import geopandas as gpd

training = gpd.read_file('training.shp')

X, y, xy = extract(
    raster=predictors, response_gdf=training, field='id')
```

The response_gdf argument of the extract function requires a Geopandas GeoDataFrame object, which contain either points or polygons. Alternatively if this response feature is represented by raster data (GDAL-supported, single-band raster), then the response_raster argument can be used:

```
training = gpd.read_file('training.tif')

X, y, xy = extract(
    raster=predictors, response_raster=training, field='id')
```

Note the extract function returns three numpy-arrays as a tuple, consisting of the extracted pixel values (X), the response variable value (y) and the sampled locations (2d numpy array of x,y values). These represent masked arrays with nodata values in the predictors being masked, and the equivalent entries in y and xy being masked on axis=0.

Next we can train a logistic regression classifier:

```
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# define the classifier with standardization of the input features in a pipeline
lr = Pipeline(
    [('scaling', StandardScaler()),
     ('classifier', LogisticRegressionCV(n_jobs=-1))])
````

After defining a classifier, a typical step consists of performing a cross-validation to evaluate the performance of the model. Scikit-learn provides the cross_validate function for this purpose. In comparison to non-spatial data, spatial data can be spatially correlated, which potentially can mean that geographically proximal samples may not represent truely independent samples if they are within the autocorrelation range of some of the predictors. This will lead to overly optimistic performance measures if samples in the training dataset / cross-validation partition are strongly spatially correlated with samples in the test dataset / cross-validation partition.

In this case, performing cross-validation using groups is useful, because these groups can represent spatial clusters of training samples, and samples from the same group will never occur in both the training and test partitions of a cross-validation. An example of creating random spatial clusters from point coordinates is provided here:

```
from sklearn.cluster import KMeans

# import training features
import geopandas as gpd
training_points = gpd.read_file('training_points.shp')

# extract training data from the predictors
X, y, xy = extract(
  raster=predictors, response_gdf=training_points, field='id')

# create 100 spatial clusters based on clustering of the training data point x,y coordinates
clusters = KMeans(n_clusters=100, n_jobs=-1)
clusters.fit(xy)

# cross validate
cross_validate(
  lr, X, y, groups=clusters.labels_,
  scoring=['accuracy', 'roc_auc',
  cv=5,  n_jobs=1)
```

Finally we might want to perform the prediction on the raster data. The estimator, raster and file_path fields are required. Predict_type can be either 'raw' to output a classification or regression result, or 'prob' to output class probabilities as a multi-band raster (a band for each class probability). In the latter case, indexes can also be supplied if you only want to output the probabilities for a particular class, or list of classes, by supplying the indices of those classes:

```
outfile = 'prediction.tif'
predict(estimator=lr, raster=predictors, file_path=outfile, predict_type='prob', indexes=1)
```

### Sampling Tools

For many spatial models, it is common to take a random or stratified random sample of the predictors to represent a single class (i.e. an environmental background or pseudo-absences in a binary classification model). Functions are supplied in the sampling module for this purpose:

```
from pyspatialml.sampling import random_sample, stratified_sample, sample

# extract training data using a random sample
xy = random_sample(size=1000, raster=predictors, random_state=1)
X = sample(xy, predictors)

# extract training data using a stratified random sample from a map containing categorical data
# here we are taking 50 samples per category
xy = stratified_sample(stratified='category_raster.tif', n=50)
X = sample(xy, predictors)
```

In some cases, we don't need all of the training data, but rather would spatially thin a point dataset. The filter_points function performs point-thinning based on a minimum distance buffer:

```
from pyspatialml.sampling import filter_points
import geopandas as gpd

training = gpd.read_file('training_points.shp')
training_xy = training.bounds.iloc[:, 2:].as_matrix()

thinned_points = filter_points(xy=training_xy, min_dist=500, remove='first')
```

We can also generate random points within polygons using the get_random_point_in_polygon function. This requires a shapely POLYGON geometry as an input, and returns a shapely POINT object:

```
from pyspatialml.sampling import get_random_point_in_polygon

polygons = gpd.read_file('training_polygons.shp')

# generate 5 random points in a single polygon
random_points = [get_random_point_in_polygon(polygons.geometry[0]) for i in range(5)]

# convert to a GeoDataFrame
random_points = gpd.GeoDataFrame(
  geometry=gpd.GeoSeries(random_points), crs=crs)
```

## Notes

Currently only Python 3 is supported.
