Preprocessing and Feature Engineering
=====================================

Raster Math
###########

Simple raster arithmetic operations can be performed directly on RasterLayer
objects directly

More complex operations should be performed using the ``calc`` method:
::

    raster.calc(function, file_path=None, driver='GTiff', dtype=None,
                nodata=None, progress=False)

One-Hot Encoding
################

::

    ohe_raster = one_hot_encode(
        layer, categories=None, file_path=None, driver='GTiff')

Generating Grids of Spatial Coordinates Information
###################################################

::

    xy_grids = xy_coordinates(layer, file_path=None, driver='GTiff')
    angled_grids = rotated_coordinates(layer, n_angles=8, file_path=None, driver='GTiff')
    edm_grids = distance_to_corners(layer, file_path=None, driver='GTiff')
    sample_grids = distance_to_samples(layer, rows, cols, file_path=None, driver='GTiff')

Handling of Temporary Files
###########################
