Preprocessing and Feature Engineering
=====================================

Raster Math
###########

Simple raster arithmetic operations can be performed on RasterLayer
objects directly. These operations occur using on disk processing by
reading small blocks of data at a time, and writing to temporary files to store the
results. The size of the blocks uses the GDAL file format defaults for block width
and height.

::

    from pyspatialml import Raster
    import pyspatialml.datasets.nc as nc
    import math

    # initiate a Raster object
    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
    stack = Raster(predictors)


Quick arithmetic operations on two layers:

::

    addition = stack.lsat7_2000_10 + stack.lsat7_2000_20
    subtraction = stack.lsat7_2000_10 - stack.lsat7_2000_20
    multiplication = stack.lsat7_2000_10 * stack.lsat7_2000_20
    division = stack.lsat7_2000_10 / stack.lsat7_2000_20



Overlay operations:

::

    intersection = stack.lsat7_2000_10 & stack.lsat7_2000_20
    union = stack.lsat7_2000_10 | stack.lsat7_2000_20
    xor = stack.lsat7_2000_10 ^ stack.lsat7_2000_20


Other operations:

::

    rounded = round(stack.lsat7_2000_10, 1)
    round_down = math.floor(stack.lsat7_2000_10)
    round_up = math.ceil(stack.lsat7_2000_10)
    trunc = math.trunc(stack.lsat7_2000_10)
    absolute = abs(stack.lsat7_2000_10)
    unary_pos = +stack.lsat7_2000_10
    unary_neg = -stack.lsat7_2000_10

More complex operations should be performed using the Raster ``calc`` method.
This is preferred because a user-defined function can be supplied to the method,
and multiple calculations can be performed in one step within needing to repeatedly
write intermediate results to temporary files. The user-defined calculate is memory-safe
because it is also applied to the Raster object by reading and writing in windows. The
size of the windows is set by the ``Raster_obj.block_shape`` attribute.

::

    stack.block_shape = (100, 100)

    # user-defined function that outputs a 2d array
    def compute_outputs_2d_array(arr):
        new_arr = arr[0, :, :] + arr[1, :, :]
        return new_arr

    result_2d = stack.calc(
        function=compute_outputs_2d_array,
        file_path=None,
        driver='GTiff',
        dtype=None,
        nodata=None,
        progress=False
    )


The user-defined needs to produce either a 2d numpy array with [rows, cols] or a
3d array with [band, rows, cols]:

::

    # user-defined function that outputs a 3d array representing a raster with multiple bands
    def compute_outputs_3d_array(arr):
        arr[0, :, :] = arr[0, :, :] + arr[1, ::]
        return arr

    result_3d = stack.calc(
        function=compute_outputs_3d_array,
        file_path=None,
        driver='GTiff',
        dtype=None,
        nodata=None,
        progress=False
    )

One-Hot Encoding
################

Although scikit-learn offers different one-hot encoding methods to deal with
categorical datasets, these can be inefficient when repeatedly applying models
to the same raster dataset, because the one-hot encoded transformation is applied
repeatedly on-the-fly. Pyspatialml includes a simple function to split a categorical
raster into a set of binary raster maps for each category:

::

    from pyspatialml import Raster
    from pyspatialml.preprocessing import one_hot_encode
    import pyspatialml.datasets.nc as nc

    categorical_raster = Raster(nc.ffreq)

    ohe_raster = one_hot_encode(
        layer=categorical_raster.ffreq,
        categories=None,
        file_path=None,
        driver='GTiff'
    )

The optional ``categories`` parameter allows a list of categories to be
supplied so that only these categories are encoded.

Generating Grids of Spatial Coordinates
#######################################

For certain types of spatial predictions, it is advantageous to include the model
region's spatial coordinates as predictors. In addition to using x, y coordinates,
coordinates that are defined relative to other reference points can be used to account
for non-linear spatial relationships, such as distances from the corners and centre
of the model region (i.e. the euclidean distance fields approach, Behrens et al., 2018),
rotations of the x, y coordinates, or distances from other spatial features.

Pyspatialml provides several function to quickly calculate these distance measures and
include them as additional predictors along with other raster-based data:

::

    from pyspatialml.preprocessing import (
        xy_coordinates,
        rotated_coordinates,
        distance_to_corners
    )

    # calculate coordinate grids an existing RasterLayer as a template
    xy_grids = xy_coordinates(layer=stack.iloc[0])
    angled_grids = rotated_coordinates(layer=stack.iloc[0])
    edm_grids = distance_to_corners(layer=stack.iloc[0])

    # append the coordinate grids to the Raster
    stack_new = stack.append(xy_grids, angled_grids, edm_grids, in_place=False)


Alternatively, distance measures to reference points in the raster can be
calculated using the ``pyspatialml.preprocessing.distance_to_samples`` function.
This function takes an existing RasterLayer to use as a template, and lists of
row and column indices to calculate distance measures to each row, col pair:

::

    from pyspatialml.preprocessing import distance_to_samples

    # row, col indices for top-left, top-right, lower-left and lower-right corners of raster
    # to use as example reference points:

    row_pos = [0, 0, 442, 442]
    col_pos = [0, 488, 0, 488]

    sample_grids = distance_to_samples(
        layer=stack.iloc[0],
        rows=row_pos,
        cols=col_pos
    )
