# Pyspatialml
Machine learning classification and regresssion modelling for spatial raster data.

## Description
Pyspatialml provides a suite of functions to enable machine learning classification and regression models to be applied to raster data using the rasterio package. 

## Installation

pip install git+https://github.com/stevenpawley/Pyspatialml

## Usage

from pyspatialml.models import predict, specificity_score
from pyspatialml.sampling import extractPoints, get_random_point_in_polygon
