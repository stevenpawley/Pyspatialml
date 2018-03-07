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

# cross validate
cross_validate(
  lr, X, y, groups=groups,
  scoring=['accuracy', 'roc_auc',
  cv=5,  n_jobs=1)
```

Perform the prediction on the raster data. The estimator, raster and file_path fields are required. Predict_type can be either 'raw' to output a classification or regression result, or 'prob' to output class probabilities as a multi-band raster (a band for each class probability). In the latter case, indexes can also be supplied if you only want to output the probabilities for a particular class, or list of classes, by supplying the indices of those classes:

```
outfile = 'prediction.tif'
predict(estimator=lr, raster=predictors, file_path=outfile, predict_type='prob', indexes=1)
```

### Sampling Tools

For many spatial models, it is common to take a random or stratified random sample of the predictors to represent a single class (i.e. an environmental background or pseudo-absences in a binary classification model). Functions are supplied in the sampling module for this purpose:

```

```

## Notes

Currently only Python 3 is supported.
