Pyspatialml
===========

Pyspatialml is a Python package for applying scikit-learn machine learning
models to raster-based datasets. It is inspired by the famous  ``raster``
package in the R statistical programming language which has been extensively
used for applying statistical and machine learning models to geospatial raster
datasets.

Pyspatialml includes functions and classes for working with multiple raster
datasets and applying a typical machine learning workflow including raster data
manipulation, feature engineering, extraction of training data, and application
of the ``predict`` or ``predict_proba`` methods of scikit-learn estimator
objects to a stack of raster datasets.

Pyspatialml is built upon the ``rasterio`` Python package which performs all of
the heavy lifting, and is designed towork with the ``geopandas`` package for
related raster-vector data geoprocessing operations.

Contents
========

.. toctree::
    :maxdepth: 2

    introduction
    installation
    design
    quickstart
    plotting
    preprocessing
    geoprocessing
    sampling
    mlworkflow
    modules

