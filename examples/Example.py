from osgeo import gdal
from pyspatialml import predict
from pyspatialml.sampling import extract
import os
import geopandas
import rasterio.plot
import matplotlib.pyplot as plt

# First, import the extract and predict functions:
os.chdir(os.path.join('.', 'Examples'))

band1 = 'lsat7_2000_10.tif'
band2 = 'lsat7_2000_20.tif'
band3 = 'lsat7_2000_30.tif'
band4 = 'lsat7_2000_40.tif'
band5 = 'lsat7_2000_50.tif'
band7 = 'lsat7_2000_70.tif'
predictors = [band1, band2, band3, band4, band5, band7]

# stack the bands into a single virtual tile format dataset:
vrt_file = 'landsat.vrt'
outds = gdal.BuildVRT(
    destName=vrt_file, srcDSOrSrcDSTab=predictors, separate=True,
    resolution='highest', resampleAlg='bilinear')
outds.FlushCache()

# load the vrt as a rasterio dataset:
src = rasterio.open(vrt_file)

# Load some training data in the form of a shapefile of point feature locations:
training_py = geopandas.read_file('landsat96_polygons.shp')
training_pt = geopandas.read_file('landsat96_points.shp')
training_px = rasterio.open('landsat96_labelled_pixels.tif')

# Show training data and a single raster band using numpy and matplotlib:
srr_arr = src.read(4, masked=True)
plt.imshow(srr_arr, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
plt.scatter(x=training_pt.bounds.iloc[:, 0],
            y=training_pt.bounds.iloc[:, 1],
            s=2, color='black')
plt.show()

# Create a training dataset by extracting the raster values at the training point locations:
X, y, xy = extract(dataset=src, response=training_pt, field='id')
X.shape


# A geodataframe containing polygon features can also be supplied to the extract function:
X, y, xy = extract(dataset=src, response=training_py, field='id')
X.shape

# The response argument of the extract function can also take a raster data
# (GDAL-supported, single-band raster) where the training data are represented by labelled pixels:
X, y, xy = extract(dataset=src, response=training_px)
X.shape


# The training data is returned as a masked array, with training points that intersect nodata
# values in the predictor rasters being masked. This can cause problems with sklearn,
# so here we use only the valid entries:
X = X[~X.mask.any(axis=1)]
y = y[~y.mask]
xy = xy[~xy.mask.any(axis=1)]

# Next we can train a logistic regression classifier:
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# define the classifier with standardization of the input features in a pipeline
lr = Pipeline(
    [('scaling', StandardScaler()),
     ('classifier', LogisticRegressionCV(n_jobs=-1))])

# fit the classifier
lr.fit(X, y)

# spatial cross-validation
from sklearn.cluster import KMeans

# create 10 spatial clusters based on clustering of the training data point x,y coordinates
clusters = KMeans(n_clusters=10, n_jobs=-1)
clusters.fit(xy)

# cross validate
scores = cross_validate(
  lr, X, y, groups=clusters.labels_,
  scoring='accuracy',
  cv=3,  n_jobs=1)
scores['test_score'].mean()

# prediction
outfile = 'classification.tif'
result = predict(estimator=lr, dataset=src, file_path='classification.tif',
                 dtype='int16', nodata=0)

plt.imshow(result.read(1, masked=True))
plt.show()

# sampling
from pyspatialml.sampling import sample

# extract training data using a random sample
X, xy = sample(size=1000, dataset=src, random_state=1)

row, col = rasterio.transform.rowcol(src.transform, xs=xy[:, 0], ys=xy[:, 1])
plt.imshow(src.read(1, masked=True))
plt.scatter(x=col, y=row)
plt.show()

# extract training data using a stratified random sample from a map containing categorical data
# here we are taking 50 samples per category
strata = rasterio.open('strata.tif')
X, xy = sample(size=5, dataset=src, strata=strata, random_state=1)

row, col = rasterio.transform.rowcol(src.transform, xs=xy[:, 0], ys=xy[:, 1])
plt.imshow(strata.read(1, masked=True))
plt.scatter(x=col, y=row, color='white')
plt.show()





# pyraster approach
ras = PyRaster(predictors)

pred = ras.predict(estimator=clf, file_path='classification.tif')
plt.imshow(pred.read(masked=True))