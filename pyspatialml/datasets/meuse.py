import os

chnl_dist = os.path.join(os.path.dirname(__file__), 'chnl_dist.tif')
dem = os.path.join(os.path.dirname(__file__), 'dem.tif')
dist = os.path.join(os.path.dirname(__file__), 'dist.tif')
ffreq = os.path.join(os.path.dirname(__file__), 'ffreq.tif')
landimg2 = os.path.join(os.path.dirname(__file__), 'landimg2.tif')
landimg3 = os.path.join(os.path.dirname(__file__), 'landimg3.tif')
landimg4 = os.path.join(os.path.dirname(__file__), 'landimg4.tif')
mrvbf = os.path.join(os.path.dirname(__file__), 'mrvbf.tif')
rsp = os.path.join(os.path.dirname(__file__), 'rsp.tif')
slope = os.path.join(os.path.dirname(__file__), 'slope.tif')
soil = os.path.join(os.path.dirname(__file__), 'soil.tif')
twi = os.path.join(os.path.dirname(__file__), 'twi.tif')
meuse = os.path.join(os.path.dirname(__file__), 'meuse.shp')

predictors = [
    chnl_dist,
    dem,
    dist,
    ffreq,
    landimg2,
    landimg3,
    landimg4,
    mrvbf,
    rsp,
    slope,
    soil,
    twi]
