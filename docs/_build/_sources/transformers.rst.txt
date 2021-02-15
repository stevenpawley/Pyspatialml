Transformers
************

Spatial Lag Transformer
=======================

A transformer to create spatial lag variables by using a
weighted mean/mode of the values of the K-neighboring observations. The
weighted mean/mode of the surrounding observations are appended as a new
feature to the right-most column in the training data. The `measure` parameter
should be set to 'mode' for classification, and 'mean' for regression.


::

    KNNTransformer(n_neighbors=7, weights="distance", measure="mean",
                   radius=1.0, algorithm="auto", leaf_size=30,
                   metric="minkowski", p=2, normalize=True, metric_params=None,
                   kernel_params=None, n_jobs=1)


GeoDistTransformer
==================

A common spatial feature engineering task is to create new features that
describe the proximity to some reference locations. The GeoDistTransformer
can be used to add these features as part of a machine learning pipeline.

::

    GeoDistTransformer(refs, log=False)


Where `refs` are an array of coordinates of reference locations in
(m, n-dimensional) order, such as
{n_locations, x_coordinates, y_coordinates, ...} for as many dimensions as
required. For example to calculate distances to a single x,y,z location:

::

    refs = [-57.345, -110.134, 1012]


And to calculate distances to three x,y reference locations:

::

    refs = [
        [-57.345, -110.134],
        [-56.345, -109.123],
        [-58.534, -112.123]
    ]


The supplied array has to have at least x,y coordinates with a
(1, 2) shape for a single location.
