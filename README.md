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

Import the extract and predict functions:
```
from pyspatialml.sampling import extract
from pyspatialml import predict
from osgeo import gdal
import rasterio
```

Stack a series of separate rasters as a virtual tile format raster:
```
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

## Notes

Currently only Python 3 is supported.
