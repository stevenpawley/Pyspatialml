Design
******

Pyspatialml provides access to raster datasets using two main structures,
comprising the `Raster` and `RasterLayer` classes.

Raster Objects
==============

The main class that facilitates working with multiple raster datasets is the
`Raster` class. The `Raster` object takes a list
GDAL-supported raster datasets and 'stacks' them into a single Raster
object. The underlying file-based raster datasets are not physically-stacked,
but rather the Raster object internally represents each band within the
datasets as a `RasterLayer`. This means that metadata regarding what
each raster dataset represents (e.g. the dataset's name) can be retained, and
additional raster datasets can be added or removed from the stack without
physical on-disk changes.

Note these raster datasets need to be spatially aligned in terms of their
extent, resolution and coordinate reference system.

Creating a new Raster object
----------------------------

There are several methods of creating a new Raster object:

1. ``Raster(src=[raster1.tif, raster2.tif, raster3.tif])`` creates a
Raster object from existing file-based GDAL-supported datasets, or a single
raster dataset. The file-based datasets can contain single or multiple bands.

2. ``Raster(src=new_numpy_array, crs=crs, transform=transform)`` creates a
Raster object from a 3D numpy array (band, row, column). The ``crs`` and
``transform`` arguments are optional but are required to provide coordinate
reference system information to the Raster object. The crs argument has to be
represented by ```rasterio.crs.CRS``` object, and the transform parameter
requires an ```affine.Affine``` object.

3. ``Raster(src=[RasterLayer1, RasterLayer2, RasterLayer3])`` creates a
Raster object from a single or list of RasterLayer objects. RasterLayers are a
thin wrapper around rasterio.Band objects with additional methods. This is
mostly used internally. A RasterLayer itself is initiated directly from a
rasterio.Band object.

4. A Raster can also be initated directly from a `rasterio.Band` object, or
a list of rasterio.Band objects.

Generally, Pyspatialml intends users to work with the Raster object. However,
access to individual RasterLayer objects, or the underlying rasterio.band
datasets can be useful if pyspatialml is being used in conjunction with other
functions and methods in the ``rasterio`` package.

Overview of attributes and methods
----------------------------------

Attributes
++++++++++

- `Raster.loc`: Access pyspatialml.RasterLayer objects within a Raster
  using a key, or a list of keys.
- `Raster.iloc`: Access pyspatialml.RasterLayer objects using an index
  position.
- `Raster.files`: A list of the on-disk files that the Raster object
  references.
- `Raster.dtypes`: A list of numpy dtypes for each RasterLayer.
- `Raster.nodatavals`: A list of the nodata values for each RasterLayer.
- `Raster.count`: The number of RasterLayers in the Raster.
- `Raster.res`: The resolution in (x, y) dimensions of the Raster.
- `Raster.meta`: A dict containing the raster metadata.
- `Raster.names`: A list of the RasterLayer names.
- `Raster.block_shape`: The default block_shape in (rows, cols) for
  reading windows of data in the Raster for out-of-memory processing.

Methods
+++++++

- `Raster.read`: eads data from the Raster object into a numpy array.
- `Raster.write`: Write the Raster object to a file.
- `Raster.predict_proba`: Apply class probability prediction of a scikit
  learn model to a Raster.
- `Raster.predict`: Apply prediction of a scikit learn model to a Raster.
- `Raster.append`: Method to add new RasterLayers to a Raster object.
- `Raster.drop`: Drop individual RasterLayers from a Raster object.
- `Raster.rename`: Rename a RasterLayer within the Raster object.
- `Raster.plot`: Plot a Raster object as a raster matrix.
- `Raster.mask`: Mask a Raster object based on the outline of shapes in
  a ``geopandas.GeoDataFrame``.
- `Raster.intersect`: Perform a intersect operation on the Raster object.
- `Raster.crop`: Crops a Raster object by the supplied bounds.
- `Raster.to_crs`: Reprojects a Raster object to a different crs.
- `Raster.aggregate`: Aggregates a raster to (usually) a coarser grid
  cell size.
- `Raster.apply`: Apply user-supplied function to a Raster object.
- `Raster.block_shapes`: Generator for windows for optimal reading and
  writing based on the raster.
- `Raster.copy`: Perform a shallow copy of a Raster object. Note only the
  object itself is copied, not the physical file locations that it is linked
  with.
- `Raster.sample`: Take a random or stratified sample from a Raster.
- `Raster.extract_vector`: Spatial query the pixels in the Raster using a
  geopandas.GeoDataFrame of point, line or polygon geometries.
- `Raster.extract_raster`: Spatial query the pixels in the Raster using a
  raster of labelled pixels.
- `Raster.extract_xy`: Extract pixel values using a list of x,y coordinates.
- `Raster.plot`: Plot a rasterplot matrix of all layers within the Raster.

RasterLayer
===========

A `RasterLayer` is an object that represents a single raster band. It is
based on a ``rasterio.band`` object with some additional attributes and
methods. A RasterLayer is used because the ``rasterio.Band.ds.read`` method
reads all bands from a multi-band dataset, whereas the RasterLayer read method
always refers to a single band.

A RasterLayer object is generally not intended to be initiated directly, but
rather is used internally as part of a `Raster` to represent individual
bands. However, a RasterLayer can be initiated from a `rasterio.band` object.

Overview of attributes and methods
----------------------------------

Methods encapsulated in RasterLayer objects represent those that typically
would only be applied to a single-band of a raster, i.e. sieve-clump, distance
to non-NaN pixels, or arithmetic operations on individual layers.

Attributes
++++++++++

- `RasterLayer.bidx`: The band index of the RasterLayer within the file
  dataset.
- `RasterLayer.dtype`: The data type of the RasterLayer.
- `RasterLayer.nodata`: The number that is used to represent nodata
  pixels in the RasterLayer.
- `RasterLayer.file`: The file path to the dataset.
- `RasterLayer.ds`: The underlying rasterio.band object.
- `RasterLayer.driver`: The name of the GDAL format driver.
- `RasterLayer.meta`: A python dict storing the RasterLayer metadata.
- `RasterLayer.cmap`: The name of matplotlib map, or a custom
  matplotlib.cm.LinearSegmentedColormap or ListedColormap object.
- `RasterLayer.norm`: A matplotlib.colors.Normalize to apply to the
  RasterLayer.
- `RasterLayer.count`: Number of layers; always equal to 1.
- `RasterLayer.name`: A name that is assigned to the RasterLayer, usually
  created when a parent Raster object is initiated.

Methods
+++++++

- `RasterLayer.close`: Closes a RasterLayer.
- `RasterLayer.read`: Reads data from the RasterLayer into a numpy array.
- `RasterLayer.write`: Write the RasterLayer to a GDAL-supported raster
  dataset.
- `RasterLayer.fill`: Fill nodata gaps in a RasterLayer. Wrapper around
  the ``rasterio.fill.fillnodata`` method.
- `RasterLayer.sieve`: Replace pixels with their largest neighbor.
  Wrapper around the ``rasterio.features.sieve`` method.
- `RasterLayer.distance`: Calculate euclidean grid distances to non-NaN
  pixels.
- `RasterLayer.plot`: Plot a RasterLayer using ``matplotlib.pyplot.imshow``.
- `RasterLayer.min`, `RasterLayer.max`, `RasterLayer.mean`,
  `RasterLayer.median`: Basic summary statistics for a layer.
