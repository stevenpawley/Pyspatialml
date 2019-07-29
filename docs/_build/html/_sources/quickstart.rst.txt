Quickstart
==========

Initiating a Raster Object
##########################

We are going to use a set of Landsat 7 bands contained within the nc_dataset:
::

    import pyspatialml.datasets.nc_dataset as nc

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5,
                  nc.band7]

These raster datasets are aligned in terms of their extent and coordinate
reference systems. We can 'stack' these into a Raster class so that we can
perform machine learning related operations on the set of rasters:
::

    stack = Raster(predictors)

Upon 'stacking', syntactically-correct names for each RasterLayer are
automatically generated from the file_paths.

Indexing
########

Indexing of Raster objects is provided by several methods:

* Raster.loc[key] : provides key-based indexing based on the names of the
  RasterLayers, and always returns a RasterLayer object. Unlike a regular dict,
  a list or tuple of keys can be provided to return multiple layers.
* Raster.iloc[int, list, tuple, slice] : provides integer-based indexing or
  slicing, and always returns a RasterLayer object, or list of RasterLayers.
* Raster[key] : provides key-based indexing, but returns a new Raster object
  with the subset layers.
* Raster.name : attribute names can be used directly, and always returns a
  single RasterLayer object.

RasterLayer indexing which returns a RasterLayer:
::

    stack.iloc[0]  # index
    stack.iloc[0:3]  # slice
    stack.loc['lsat7_2000_10']  # key
    stack.loc[('lsat7_2000_10', 'lsat7_2000_20')]  # list or tuple of keys
    stack.lsat7_2000_10  # attribute

Iterate through RasterLayers:
::

    for name, layer in stack:
        print(layer)

Subset a Raster object:
::

    subset_raster = stack[['lsat7_2000_10', 'lsat7_2000_70']]
    subset_raster.names

Replace a RasterLayer with another:
::

    stack.iloc[0] = Raster(nc.band7).iloc[0]

Appending and Dropping Layers
#############################

Append layers from another Raster to the stack. Note that this occurs in-place.
Duplicate names are automatically given a suffix:
::

    stack.append(Raster(nc.band7))
    stack.names

Rename RasterLayers using a dict of old_name : new_name pairs:
::

    stack.names
    stack.rename({'lsat7_2000_30': 'new_name'})
    stack.names
    stack.new_name
    stack['new_name']
    stack.loc['new_name']

Drop a RasterLayer:
::

    stack.names
    stack.drop(labels='lsat7_2000_70_1')
    stack.names

Integration with Pandas
#######################

Data from a Raster object can converted into a Pandas dataframe, with each pixel
representing by a row, and columns reflecting the x, y coordinates and the
values of each RasterLayer in the Raster object:
::

    df = stack.to_pandas(max_pixels=50000, resampling='nearest')
    df.head()

The original raster is up-sampled based on max_pixels and the resampling method,
which uses all of resampling methods available in the underlying rasterio
library for decimated reads. The Raster.to_pandas method can be useful for
plotting datasets, or combining with a library such as plotnine to create
ggplot2-style plots of stacks of RasterLayers:
::

    from plotnine import *
    (ggplot(df.melt(id_vars=['x', 'y']), aes(x='x', y='y', fill='value')) +
     geom_tile() +
     facet_wrap('variable')
     )

Saving a Raster to File
#######################

Save a Raster:
::

    tmp_tif = tempfile.NamedTemporaryFile().name + '.tif'
    newstack = stack.write(file_path=tmp_tif, nodata=-9999)
    newstack.new_name.read()
    newstack = None
