#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rasterio
from rasterio.warp import reproject
from tempfile import NamedTemporaryFile as tmpfile

def read_memmap(src, bands=None):
    """Read a rasterio._io.RasterReader object into a numpy.memmap file
    
    Parameters
    ----------
    src : rasterio._io.RasterReader
        Rasterio raster that is open.
    bands : int or list, optional (default = None)
        Optionally specify to load a single band, a list of bands, or None for
        all bands within the raster.
    
    Returns
    -------
    arr : array-like
        2D or 3D numpy.memmap containing raster data. Nodata values are 
        represented by np.nan. By default a 3D numpy.memmap object is returned
        unless a single band is specified, in which case a 2D numpy.memmap
        object is returned.
    """
    
    # default loads all bands as 3D numpy array
    if bands is None:
        bands = range(1, src.count+1, 1)
        arr_shape = (src.count, src.shape[0], src.shape[1])

    # if a single band is specified use 2D numpy array 
    elif len(bands) == 1:
        arr_shape = (src.shape[0], src.shape[1])

    # use 3D numpy array for a subset of bans
    else:
        arr_shape = (len(bands), src.shape[0], src.shape[1])
        
    # numpy.memmap using float32
    arr = np.memmap(filename=tmpfile(), shape=arr_shape, dtype='float32')
    arr[:] = src.read(bands)
    arr[arr==src.nodata] = np.nan
    
    return arr


def reclass_nodata(input, output, src_nodata=None, dst_nodata=-99999, intern=False):
    """Reclassify raster no data values and save to a new raster
    
    Parameters
    ----------
    input : str
        File path to raster that is to be reclassified
    output : str
        File path to output raster
    src_nodata : int or float, optional
        The source nodata value. Pixels with this value will be reclassified
        to the new dst_nodata value.
        If not set, it will default to the nodata value stored in the source
        image.
    dst_nodata : int or float, optional (default -99999)
        The nodata value that the outout raster will receive after
        reclassifying pixels with the src_nodata value.
    itern : bool, optional (default=False)
        Optionally return the reclassified raster as a numpy array
    
    Returns
    -------
    out: None, or 2D numpy array of raster with reclassified nodata pixels 
    """
    
    r = rasterio.open(input)
    width, height, transform, crs = r.width, r.height, r.transform, r.crs
    img_ar = r.read(1)
    if src_nodata is None:
        src_nodata = r.nodata
        
    img_ar[img_ar == src_nodata] = dst_nodata
    r.close()
    
    with rasterio.open(path=output, mode='w', driver='GTiff', width=width,
                       height=height, count=1, transform=transform, crs=crs,
                       dtype=str(img_ar.dtype), nodata=dst_nodata) as dst:
        dst.write(img_ar, 1)

    if intern is True:
        return (img_ar)
    else:
        return()
    

def align_rasters(rasters, template, outputdir, method="Resampling.nearest",
                  src_nodata = None, dst_nodata = None):

    """Aligns a list of rasters (paths to files) to a template raster.
    The rasters to that are to be realigned are assumed to represent
    single band raster files.
    
    Nodata values are also reclassified to the template raster's nodata values
    
    Parameters
    ----------
    rasters : List of str
        List containing file paths to multiple rasters that are to be realigned.
    template : str
        File path to raster that is to be used as a template to transform the
        other rasters to.
    outputdir : str
        Directory to output the realigned rasters. This should not be the
        existing directory unless it is desired that the existing rasters to be
        realigned should be overwritten.
    method : str
        Resampling method to use. One of the following:
            Resampling.average,
            Resampling.bilinear,
            Resampling.cubic,
            Resampling.cubic_spline,
            Resampling.gauss,
            Resampling.lanczos,
            Resampling.max,
            Resampling.med,
            Resampling.min,
            Resampling.mode,
            Resampling.nearest,
            Resampling.q1,
            Resampling.q3
    src_nodata : int or float, optional
        The source raster nodata value. Pixels with this value will be
        transformed to the new dst_nodata value.
        If not set, it will default to the nodata value stored in the source
        image.
    dst_nodata : int or float
        The nodata value that the outout raster will receive after realignment
        If not set, the source rasters nodata value will be used, or the
        GDAL default of 0
    """
    
    # check resampling methods
    methods = dir(rasterio.enums.Resampling)
    methods = [i for i in methods if i.startswith('__') is False]
    methods = ['Resampling.' + i for i in methods]
    
    if method not in methods:
        raise ValueError('Invalid resampling method: ' + method + os.linesep +
                         'Valid methods are: ' + str(methods))

    # open raster to be used as the template and create numpy array
    template = rasterio.open(template)
    kwargs = template.meta.copy()

    for raster in rasters:
        
        output = os.path.join(outputdir, os.path.basename(raster))
        
        with rasterio.open(raster) as src:
            with rasterio.open(output, 'w', **kwargs) as dst:
                reproject(source=rasterio.band(src, 1),
                          destination=rasterio.band(dst, 1),
                          dst_transform=template.transform,
                          dst_nodata=template.nodata,
                          dst_crs=template.nodata)
        
    template.close()

    return()
    
