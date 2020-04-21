Design
######

The Raster Object
*****************

The main class that facilitates working with multiple raster datasets is the
``Raster`` class, which is inspired by the famous  'raster' package in the R
statistical programming language. The ``Raster`` object takes a list 
GDAL-supported raster datasets and 'stacks' them into a single Raster
object. The underlying file-based raster datasets are not physically-stacked,
but rather the Raster object internally represents each band within the datasets
as a ``RasterLayer``. This means that metadata regarding what each raster
dataset represents (e.g. the dataset's name) can be retained, and additional
raster datasets can easily be added or removed from the stack without physical
on-disk changes.

Note these raster datasets need to be spatially aligned in terms of their
extent, resolution and coordinate reference system. If they are not aligned,
then for convenience the ``pyspatialml.preprocessing.align_rasters`` function can be
used to resample a list of raster datasets.

Raster Object Initiation
------------------------

There are three methods of creating a new Raster object:

1. ``Raster(file_path=[raster1.tif, raster2.tif, raster3.tif])`` creates a
Raster object from existing file-based GDAL-supported datasets.

2. ``Raster(arr=new_numpy_array, crs=crs, transform=transform)`` creates a
Raster object from a 3D numpy array (band, row, column). The ``crs`` and
``transform`` arguments are optional but are required to provide coordinate
reference system information to the Raster object. The crs argument has to be
represented by ```rasterio.crs.CRS``` object, and the transform parameter requires
a ```affine.Affine``` object.

3. ``Raster(layers=[RasterLayer1, RasterLayer2, RasterLayer3])`` creates a
Raster object from a single or list of RasterLayer objects. RasterLayers are a
thin wrapper around rasterio.Band objects with additional methods. This is
mostly used internally. A RasterLayer itself is initiated directly from a
rasterio.Band object.

Generally, Pyspatialml intends users to work with the Raster object. However,
access to individual RasterLayer objects, or the underlying rasterio.band
datasets can be useful if pyspatialml is being used in conjunction with other
functions and methods in the ``rasterio`` package.

