Quickstart
**********

Initiating a Raster Object
==========================

We are going to use a set of Landsat 7 bands contained within the nc example
data:

::

    from pyspatialml import Raster
    import pyspatialml.datasets.nc as nc
    import matplotlib.pyplot as plt

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]


These raster datasets are aligned in terms of their extent and coordinate
reference systems. We can 'stack' these into a Raster class so that we can
perform machine learning related operations on the set of rasters:

::

    stack = Raster(predictors)


When a Raster object is created, the names to each layer are automatically
created based on syntactically-correct versions of the file basenames:

::

    stack.names


Color ramps and matplotlib.colors.Normalize objects can be assigned to each
RasterLayer in the object using the `cmap` and `norm` attributes for
convenient in plotting:

::

    stack.lsat7_2000_10.cmap = "Blues"
    stack.lsat7_2000_20.cmap = "Greens"
    stack.lsat7_2000_30.cmap = "Reds"
    stack.lsat7_2000_40.cmap = "RdPu"
    stack.lsat7_2000_50.cmap = "autumn"
    stack.lsat7_2000_70.cmap = "hot"

    stack.plot(
        title_fontsize=8,
        label_fontsize=6,
        legend_fontsize=6,
        names=["B1", "B2", "B3", "B4", "B5", "B7"],
        fig_kwds={"figsize": (8, 4)},
        subplots_kwds={"wspace": 0.3}
    )
    plt.show()


.. figure:: ../img/nc_stack.png
    :width: 700px
    :align: center
    :height: 400px
    :alt: nc stack
    :figclass: align-center

    Raster of stacked nc data

Subsetting and Indexing
=======================

Indexing of Raster objects is provided by several methods:

The ``Raster[keys]`` method enables key-based indexing using a name of a
RasterLayer, or a list of names. Direct subsetting of a Raster object instance
returns a RasterLayer if only a single label is used, otherwise it always
returns a new Raster object containing only the selected layers.

The ``Raster.iloc[int, list, tuple, slice]`` method allows a Raster object
instance to be subset using integer-based indexing or slicing. The ``iloc``
method returns a RasterLayer object if only a single index is used, otherwise
it always returns a new Raster object containing only the selected layers.

Subsetting of a Raster object instance can also occur by using attribute names
in the form of ``Raster.name_of_layer``. Because only a single RasterLayer can
be subset at one time using this approach, a RasterLayer object is always
returned.

Examples of methods to subset a Raster object:

::

    # subset based on position
    single_layer = stack.iloc[0]

    # subset using a slice
    new_raster_obj = stack.iloc[0:3]

    # subset using labels
    single_layer = stack['lsat7_2000_10']
    single_layer = stack.lsat7_2000_10

    # list or tuple of keys
    new_raster_obj = stack[('lsat7_2000_10', 'lsat7_2000_20')]


Iterate through RasterLayers individually:
::

    for name, layer in stack:
        print(layer)


Replace a RasterLayer with another:
::

    stack.iloc[0] = Raster(nc.band7).iloc[0]

Appending and Dropping Layers
=============================

Append layers from another Raster to the stack. Duplicate names are
automatically given a suffix.

::

    stack.append(Raster(nc.band7), in_place=True)
    stack.names

Rename RasterLayers using a dict of old_name : new_name pairs:

::

    stack.names
    stack.rename({'lsat7_2000_30': 'new_name'}, in_place=True)
    stack.names
    stack.new_name
    stack['new_name']

Drop a RasterLayer:
::

    stack.names
    stack.drop(labels='lsat7_2000_70_1', in_place=True)
    stack.names

Integration with Pandas
=======================

Data from a Raster object can converted into a ``Pandas.DataDrame``, with each
pixel representing by a row, and columns reflecting the x, y coordinates and
the values of each RasterLayer in the Raster object:

::

    import pandas as pd

    df = stack.to_pandas(max_pixels=50000, resampling='nearest')
    df.head()

The original raster is up-sampled based on max_pixels and the resampling
method, which uses all of resampling methods available in the underlying
rasterio library for decimated reads. The Raster.to_pandas method can be useful
for plotting datasets, or combining with a library such as ``plotnine`` to
create ggplot2-style plots of stacks of RasterLayers:

::

    from plotnine import *

    (ggplot(df.melt(id_vars=['x', 'y']), aes(x='x', y='y', fill='value')) +
     geom_tile() +
     facet_wrap('variable'))

Saving a Raster to File
=======================

Save a Raster:
::

    import tempfile

    tmp_tif = tempfile.NamedTemporaryFile().name + '.tif'
    newstack = stack.write(file_path=tmp_tif, nodata=-9999)
    newstack.new_name.read()
    newstack = None
