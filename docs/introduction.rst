Introduction
============

A supervised machine-learning workflow as applied to spatial raster data
typically involves several steps:

1. Extracting training data from a stack of raster images, or using an existing
   dataset containing measured values that are equivalent to data represented by
   the raster predictors.
2. Developing a machine learning classification or regression model. Pyspatialml
   uses scikit-learn or any other library with a compatible api for this
   purpose.
3. Performing the prediction on the raster data.

Training data consists of two components: (1) a response feature; and (2) a set
of predictors.

With spatial data, the response feature is often represented by
spatial locations when some property/state/concentration has already been
established. These data might be represented as point locations (e.g. arsenic
concentrations in soil samples), pixel locations where the pixel value
represents the target of interest, or polygon features (e.g. labelled with land
cover type).

The predictors are represented by raster data, which contain variables that that
in part may explaining the spatial distribution of the response variable
(e.g., raster data representing soil types, soil properties, climatic data
etc).

.. figure:: ../img/Pyspatialml_training.svg
    :width: 700px
    :align: center
    :height: 400px
    :alt: extracting training data
    :figclass: align-center

    Training data extraction
