# Pyspatialml
Machine learning classification and regresssion modelling for spatial raster data.

## Description
Pyspatialml provides a suite of functions to enable machine learning classification and regression models to be applied to raster data. The package relies on the rasterio package for accessing raster data, and geopandas/shapely for vector data.

## Notes

Currently only Python 3 is supported.

## Installation

pip install git+https://github.com/stevenpawley/Pyspatialml

## Usage

from pyspatialml.models import predict
from pyspatialml.sampling import extract_points
from pyspatialml.utils import align_rasters


