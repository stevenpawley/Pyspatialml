#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:05:40 2018

@author: steven
"""

from pyspatialml.sampling import extract
from pyspatialml import predict
import geopandas as gpd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

training_pts = gpd.read_file(
    '/Users/steven/GIS/TestGISdata/Digital-Globe-Quickbird/TileGrid_Training_Points.shp')

class_summary = pd.DataFrame(
    training_pts.VALUE.value_counts(sort=False).sort_index())

X, y, _ = extract(
    raster='/Users/steven/GIS/TestGISdata/Digital-Globe-Quickbird/TileGrid.sdat',
    response_gdf=training_pts, field='VALUE')
                  
class_summary['Sampled'] = np.bincount(y[~y.mask].astype('int'))
class_summary