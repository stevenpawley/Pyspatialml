Design
******

Pyspatialml provides access to raster datasets using two main structures, comprising the
:class:`Raster` and :class:`RasterLayer` classes.

Raster Objects
==============

The main class that facilitates working with multiple raster datasets is the
:class:`Raster` class. The :class:`Raster` object takes a list 
GDAL-supported raster datasets and 'stacks' them into a single Raster
object. The underlying file-based raster datasets are not physically-stacked,
but rather the Raster object internally represents each band within the datasets
as a :class:`RasterLayer`. This means that metadata regarding what each raster
dataset represents (e.g. the dataset's name) can be retained, and additional
raster datasets can easily be added or removed from the stack without physical
on-disk changes.

Note these raster datasets need to be spatially aligned in terms of their
extent, resolution and coordinate reference system. If they are not aligned,
then for convenience the ``pyspatialml.preprocessing.align_rasters`` function can be
used to resample a list of raster datasets.

Creating a new Raster object
----------------------------

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

Overview of attributes and methods
----------------------------------

Attributes
++++++++++

- :attr:`Raster.loc`: Access pyspatialml.RasterLayer objects within a Raster using a key, or a list of keys.
- :attr:`Raster.iloc`: Access pyspatialml.RasterLayer objects using an index position.
- :attr:`Raster.files`: A list of the on-disk files that the Raster object references.
- :attr:`Raster.dtypes`: A list of numpy dtypes for each RasterLayer.
- :attr:`Raster.nodatavals`: A list of the nodata values for each RasterLayer.
- :attr:`Raster.count`: The number of RasterLayers in the Raster.
- :attr:`Raster.res`: The resolution in (x, y) dimensions of the Raster.
- :attr:`Raster.meta`: A dict containing the raster metadata.
- :attr:`Raster.names`: A list of the RasterLayer names.
- :attr:`Raster.block_shape`: The default block_shape in (rows, cols) for reading windows of data in the Raster for out-of-memory processing.

Methods
+++++++

- :attr:`~Raster.read`: eads data from the Raster object into a numpy array.
- :attr:`~Raster.write`: Write the Raster object to a file.
- :attr:`~Raster.predict_proba`: Apply class probability prediction of a scikit learn model to a Raster.
- :attr:`~Raster.predict`: Apply prediction of a scikit learn model to a Raster.
- :attr:`~Raster.append`: Method to add new RasterLayers to a Raster object.
- :attr:`~Raster.drop`: Drop individual RasterLayers from a Raster object.
- :attr:`~Raster.rename`: Rename a RasterLayer within the Raster object.
- :attr:`~Raster.plot`: Plot a Raster object as a raster matrix.
- :attr:`~Raster.mask`: Mask a Raster object based on the outline of shapes in a ``geopandas.GeoDataFrame``.
- :attr:`~Raster.intersect`: Perform a intersect operation on the Raster object.
- :attr:`~Raster.crop`: Crops a Raster object by the supplied bounds.
- :attr:`~Raster.to_crs`: Reprojects a Raster object to a different crs.
- :attr:`~Raster.aggregate`: Aggregates a raster to (usually) a coarser grid cell size.
- :attr:`~Raster.apply`: Apply user-supplied function to a Raster object.
- :attr:`~Raster.block_shapes`: Generator for windows for optimal reading and writing based on the raster.
- :attr:`~Raster.astype`: Not currently implemented.

RasterLayer
===========

A :class:`RasterLayer` is an object that represents a single raster band. It is based
on a ``rasterio.band`` object with some additional attributes and methods. A RasterLayer
is used because the ``rasterio.Band.ds.read`` method reads all bands from a multi-band dataset,
whereas the RasterLayer read method always refers to a single band.

A RasterLayer object is generally not intended to be initiated directly, but rather is used
internally as part of a :class:`Raster` to represent individual bands. However, a RasterLayer
can be initiated from a `rasterio.band` object.

Overview of attributes and methods
----------------------------------

Methods encapsulated in RasterLayer objects represent those that typically would
only be applied to a single-band of a raster, i.e. sieve-clump, distance to non-NaN
pixels, or arithmetic operations on individual layers.

Attributes
++++++++++

- :attr:`~RasterLayer.bidx`: The band index of the RasterLayer within the file dataset.
- :attr:`~RasterLayer.dtype`: The data type of the RasterLayer.
- :attr:`~RasterLayer.nodata`: The number that is used to represent nodata pixels in the RasterLayer.
- :attr:`~RasterLayer.file`: The file path to the dataset.
- :attr:`~RasterLayer.ds`: The underlying rasterio.band object.
- :attr:`~RasterLayer.driver`: The name of the GDAL format driver.
- :attr:`~RasterLayer.meta`: A python dict storing the RasterLayer metadata.
- :attr:`~RasterLayer.cmap`: The name of matplotlib map, or a custom matplotlib.cm.LinearSegmentedColormap or ListedColormap object.
- :attr:`~RasterLayer.norm`: A matplotlib.colors.Normalize to apply to the RasterLayer.
- :attr:`~RasterLayer.count`: Number of layers; always equal to 1.

Methods
+++++++

- :meth:`~RasterLayer.close`: Closes a RasterLayer.
- :meth:`~RasterLayer.read`: Reads data from the RasterLayer into a numpy array.
- :meth:`~RasterLayer.write`: Write the RasterLayer to a GDAL-supported raster dataset.
- :meth:`~RasterLayer.fill`: Fill nodata gaps in a RasterLayer. Wrapper around the ``rasterio.fill.fillnodata`` method.
- :meth:`~RasterLayer.sieve`: Replace pixels with their largest neighbor. Wrapper around the ``rasterio.features.sieve`` method.
- :meth:`~RasterLayer.distance`: Calculate euclidean grid distances to non-NaN pixels.
- :meth:`~RasterLayer.plot`: Plot a RasterLayer using ``matplotlib.pyplot.imshow``.
