Introduction
============

A supervised machine-learning workflow as applied to spatial raster data
typically involves several steps:

1. Using vector features or labelled pixels to extract training data from a stack of
   raster-based predictors (e.g. spectral bands, terrain derivatives, or climate grids).
   The training data represent locations when some property/state/concentration is already
   established, and might comprise point locations of arsenic concentrations, or
   labelled pixels with integer-encoded values that correspond to known landcover types.
2. Developing a machine learning classification or regression model on the training data.
   Pyspatialml is designed to use scikit-learn compatible api's for this purpose.
3. Applying the fitted machine learning model to make predictions on all of the pixels in
   the stack of raster data.

Pyspatialml is designed to make it easy to develop spatial prediction models on stacks of
2D raster datasets that are held on disk. Unlike using python's ``numpy`` module directly
where raster datasets need to be held in memory, the majority of functions within pyspatialml
work with raster datasets that are stored on disk and allow processing operations to be
performed on datasets that are too large to be loaded into memory.

Pyspatialml is designed to make it easy to work with typical raster data stacks consisting of
multiple 2D grids such as different spectal bands, maps etc. However, it's purpose is not to 
work with multidimensional datasets, i.e. those that have more than 3 dimensions such as 
spacetime cubes of multiband data. The [xarray](http://xarray.pydata.org/en/stable/index.html)
package can provide a structure for this type of data.
