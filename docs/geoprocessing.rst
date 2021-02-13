Raster Geoprocessing
********************

Pyspatialml includes common geoprocessing methods that collectively operate on
stacks of raster datasets, such as cropping, reprojecting, masking etc. Most
of these methods are simple wrappers around underlying rasterio functions, but
applied to stacks of raster datasets.

Handling of Temporary Files
===========================

All of the geoprocessing methods have a `file_path` parameter to specify a file
path to save the results of th geoprocessing operation. However, pyspatialml is
designed for quick an interactive analyses on raster datasets, and if a file
path is not specified then the results are saved to a temporary file location
and a new Raster object is returned with the geoprocessing results.

Cropping a Raster object
========================

All layers within a Raster can be cropped to a new extent using the
``Raster.crop`` method.

::

    import geopandas as gpd
    from pyspatialml import Raster
    from pyspatialml.datasets import nc

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    stack = Raster(predictors)

    # crop to new extent (xmin, ymin, xmax, ymax)
    crop_bounds = training_py.loc[0, "geometry"].bounds
    stack_cropped = stack.crop(self.crop_bounds)


Masking a Raster
================

In comparison to cropping, masking can be used to set pixels that occur outside
of masking geometries to NaN, and optionally can also crop a Raster.

::

    import geopandas as gpd
    import pyspatialml.datasets.nc as nc
    from pyspatialml import Raster

    training_py = gpd.read_file(nc.polygons)
    mask_py = training_py.iloc[0:1, :]

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    # mask a Raster
    masked_object = stack.mask(mask_py)


Intersecting Layers in a Raster
===============================

The ``Raster.intersect`` method computes the geometric intersection of the
RasterLayers with the Raster object. This will cause nodata values in any of
the rasters to be propagated through all of the output rasters.

::

    import pyspatialml.datasets.nc as nc
    from pyspatialml import Raster

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)

    result = stack.intersect()

The intersect method is memory-safe, i.e. the intersection occurs during
windowed reading and writing of the Raster. The size and dimensions of the
windows can be changed using the `Raster.block_shapes` property.

Reprojecting
============

Reprojecting a raster using the ``Raster.to_crs`` method.

::

    stack_prj = stack.to_crs(crs={"init": "EPSG:4326"})

Other parameters that can be passed and their defaults are
resampling="nearest", file_path=None, driver="GTiff", nodata=None, n_jobs=1,
warp_mem_lim=0, progress=False, and other kwargs that are passed to the raster
format drivers.

Resampling
==========

The ``Raster.aggregate`` method is used to resample a raster to a different
resolution using the decimated reading approach in the rasterio library.

::

    stack.aggregate(out_shape, resampling="nearest", file_path=None,
                    driver="GTiff", dtype=None, nodata=None, **kwargs)

Apply
=====

Apply user-supplied function to a Raster object.

::

    stack.apply(self, function, file_path=None, driver="GTiff", dtype=None,
                nodata=None, progress=False, n_jobs=-1, function_args={},
                **kwargs)

Where `function` is a user-defined python that takes an numpy array as a
single argument, and can return either a 2d array that represents a single
raster dataset, such as NDVI, or can operate on a number of layers and can
return a raster with multiple layers in a 3d array in (layer, row, col)
order.

The apply function is memory-safe, i.e. it applies the function to windows
of the raster data that are read sequentially or in parallel
(with n_jobs != 1). The size and dimensions of the windows can be changed
using the `Raster.block_shapes` property.

Geoprocessing on RasterLayer objects
************************************

RasterLayer objects also support basic raster math operations using python's
magic methods, which supports all of the usual math operators. Calculations
on RasterLayers are also memory-safe, i.e. they occur using windowed reading
and writing of the data.

Because file paths cannot be specified, the results are automatically saved
to temporary files. For example:

::

    a = stack.iloc[0] + stack.iloc[1]
    b = stack.iloc[0] - stack.iloc[1]

    ndvi = (stack.iloc[3] - stack.iloc[2]) / (stack.iloc[3] + stack.iloc[2])

Note that because temporary files are used to ensure that the operations are
memory safe, complex calculations are performed duing multiple steps, which
may be inefficient. For a more computationally efficient calculation of NDVI,
use the `Raster.apply` method to pass a function that calculates NDVI on a
3d numpy array and apply it during windowed reading and writing.
