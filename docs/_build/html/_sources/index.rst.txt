pyspatialml: machine learning for raster datasets
=================================================

Pyspatialml is a Python package for applying scikit-learn machine learning
models to raster-based datasets. It is inspired by the famous 
`raster <https://cran.r-project.org/web/packages/raster/index.html>`_
package in the R statistical programming language which has been extensively
used for applying statistical and machine learning models to geospatial raster
datasets.

Pyspatialml includes functions and classes for working with multiple raster
datasets and applying typical machine learning workflows including raster data
manipulation, feature engineering on raster datasets, extraction of training data,
and application of the ``predict`` or ``predict_proba`` methods of 
scikit-learn estimator objects to a stack of raster datasets.

Pyspatialml is built upon the 
`rasterio <https://rasterio.readthedocs.io/en/latest/>`_ Python package which performs all of
the heavy lifting and is designed to work with the 
`geopandas <https://geopandas.org>`_ package for
related raster-vector data geoprocessing operations.

Documentation
=============

Getting Started
---------------


.. toctree::
    :maxdepth: 2

    installation
    quickstart
    mlworkflow


User Guide
----------


.. toctree::
    :maxdepth: 2

    introduction
    design
    plotting
    geoprocessing
    preprocessing
    sampling
    transformers
    estimators
    cross_validation


Reference Guide
---------------

.. toctree::
    :maxdepth: 1

    modules

