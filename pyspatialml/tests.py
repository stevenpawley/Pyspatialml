# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:55:52 2018

@author: Steven Pawley
"""

import geopandas as gpd

training_stack = 'C:/GIS/Tests/ir_bushehr_28feb04_ps.tif'
training_vrt = 'C:/GIS/Tests/ir_bushehr_28feb04_ps.vrt'
training_polygons = 'C:/GIS/Tests/Training_polygons.shp'
training_points = 'C:/GIS/Tests/Training_points.shp'
training_pixels = 'C:/GIS/Tests/Training_pixels.sdat'

# read vector data
training_polygons = gpd.read_file(training_polygons)
training_points = gpd.read_file(training_points)

# extract training data from points
X, y, xy = extract(raster=training_stack, response_gdf=training_points)
X, y, xy = extract(raster=training_stack, response_gdf=training_points, field='ID')
X, y, xy = extract(raster=training_vrt, response_gdf=training_points)

# extract training data from polygons
X, y, xy = extract(raster=training_stack, response_gdf=training_polygons, field='ID')
X, y, xy = extract(raster=training_stack, response_gdf=training_polygons)

# extract training data from labelled pixels
X, y, xy = extract(raster=training_stack, response_raster=training_pixels)

spec_ind = indices(raster=training_stack, blue=1, green=2, red=3, nir=4)
spec_ind.dvi(s=0.5, file_path='C:/GIS/Tests/test_spectral2.tif')