from osgeo import gdal
from pyspatialml.sampling import extract
from pyspatialml import predict
import geopandas as gpd
import rasterio
import os

# project base directory
projectdir = '/Users/steven/Github/pyspatialml/tests'

# raster data
band1 = os.path.join(projectdir, 'lsat7_2000_10.tif')
band2 = os.path.join(projectdir, 'lsat7_2000_20.tif')
band3 = os.path.join(projectdir, 'lsat7_2000_30.tif')
band4 = os.path.join(projectdir, 'lsat7_2000_40.tif')
band5 = os.path.join(projectdir, 'lsat7_2000_50.tif')
band7 = os.path.join(projectdir, 'lsat7_2000_70.tif')

# vector data
training_points = os.path.join(projectdir, 'landclass96_roi.shp')

# prepare virtual raster
predictors = [band1, band2, band3, band4, band5, band7]
vrt_file = os.path.join(projectdir, 'landsat.vrt')
outds = gdal.BuildVRT(
    destName=vrt_file, srcDSOrSrcDSTab=predictors, separate=True,
    resolution='highest', resampleAlg='bilinear')
outds.FlushCache()

# read vector data
training_gpd = gpd.read_file(training_points)

# extract training data
X, y, xy = extract(raster=vrt_file, response_gdf=training_gpd, field='id')




# extract training data from points
X, y, xy = extract(raster=vrt_file, response_gdf=training_points, field='id')

# extract training data from polygons
X, y, xy = extract(raster=training_stack, response_gdf=training_polygons, field='ID')
X, y, xy = extract(raster=training_stack, response_gdf=training_polygons)

# extract training data from labelled pixels
X, y, xy = extract(raster=training_stack, response_raster=training_pixels)

spec_ind = indices(raster=training_stack, blue=1, green=2, red=3, nir=4)
spec_ind.dvi(s=0.5, file_path='C:/GIS/Tests/test_spectral2.tif')