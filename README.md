[![Codecov test coverage](https://codecov.io/gh/stevenpawley/pyspatialml/branch/master/graph/badge.svg)](https://codecov.io/gh/stevenpawley/pyspatialml?branch=master)

# Pyspatialml
Machine learning classification and regression modelling for spatial raster data.

## Description
`Pyspatialml` is a Python module for applying scikit-learn machine learning models to 'stacks' of raster datasets.
Pyspatialml includes functions and classes for working with multiple raster datasets and performing a typical machine
learning workflow consisting of extracting training data and applying the predict or predict_proba methods of
scikit-learn estimators to a stack of raster datasets. Pyspatialml is built upon the `rasterio` Python module for
all of the heavy lifting, and is also designed for working with vector data using the `geopandas` module.

For more information read the documents page at: https://stevenpawley.github.io/Pyspatialml/

## Rationale
A typical supervised machine-learning workflow involving raster datasets consists of several steps:

1. Using vector features or labelled pixels to extract training data from a stack of raster-based predictors (e.g.
spectral bands, terrain derivatives, or climate grids). The training data represent locations when some 
property/state/concentration is already established, and might comprise point locations of arsenic concentrations, or
labelled pixels with integer-encoded values that correspond to known landcover types.

2. Developing a machine learning classification or regression model on the training data. Pyspatialml is designed to use
scikit-learn compatible api's for this purpose.

3. Applying the fitted machine learning model to make predictions on all of the pixels in the stack of raster data.

Pyspatialml is designed to make it easy to develop spatial prediction models on stacks of 2D raster datasets that are
held on disk. Unlike using python's `numpy` module directly where raster datasets need to be held in memory, 
the majority of functions within pyspatialml work with raster datasets that are stored on disk and allow processing
operations to be performed on datasets that are too large to be loaded into memory.  

## Design

### Raster and RasterLayer objects

The main class that facilitates working with multiple raster datasets is the `Raster` class, which is inspired by
the famous  ```raster``` package in the R statistical programming language. The `Raster` object takes a list of file
paths to GDAL-supported raster datasets and 'stacks' them into a single Raster object. The underlying file-based raster
datasets are not physically-stacked, but rather the Raster object internally represents each band within the datasets as
a `RasterLayer`. This means that metadata regarding what each raster dataset represents (e.g. the dataset's name)
can be retained, and additional raster datasets can be added or removed from the stack without physical on disk changes.

Note these raster datasets need to be spatially aligned in terms of their extent, resolution and coordinate reference
system.

### Usage

There are four methods of creating a new Raster object:

1. `Raster([raster1.tif, raster2.tif, raster3.tif])` creates a Raster object from existing file-based
GDAL-supported datasets.

2. `Raster(new_numpy_array, crs=crs, transform=transform)` creates a Raster object from a 3D numpy array (band,
row, column). The `crs` and `transform` arguments are optional but are required to provide coordinate reference
system information to the Raster object. The crs argument has to be represented by `rasterio.crs.CRS` object, and
the transform parameter requires a `affine.Affine` object.

3. `Raster([RasterLayer1, RasterLayer2, RasterLayer3])` creates a Raster object from a single or list of
RasterLayer objects. RasterLayers are a thin wrapper around rasterio.Band objects with additional methods. This is
mostly used internally. A RasterLayer itself is initiated directly from a rasterio.Band object.

4. `Raster([src0, src1, src2])` where the list elements are `rasterio.io.datasetreader` objects, i.e. raster
datasets that have been opened using the `rasterio.open` method.

Generally, pyspatialml intends users to work with the Raster object. However, access to individual RasterLayer
objects, or the underlying rasterio.band datasets can be useful if pyspatialml is being used in conjunction with other
functions and methods in the Rasterio package.

## Installation

The package is now available on PyPI, but can also be installed from GitHub directly via:

```
pip install git+https://github.com/stevenpawley/Pyspatialml
```

## Quickstart

This is an example using the imagery data that is bundled with the package. This data is derived from the GRASS GIS
North Carolina dataset and comprises Landsat 7 VNIR and SWIR bands along with some land cover training data that were
derived from a land cover classification from an earlier date.

First, import the extract and predict functions:

```
from pyspatialml import Raster
from copy import deepcopy
import os
import tempfile
import geopandas
import rasterio.plot
import matplotlib.pyplot as plt
```

### Stacking 

We are going to use a set of Landsat 7 bands contained within the nc_dataset:

```
import pyspatialml.datasets.nc as nc
predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
```

These raster datasets are aligned in terms of their extent and coordinate reference systems. We can 'stack' these into a
Raster class so that we can perform machine learning related operations on the set of rasters:

```
stack = Raster(predictors)
```

Upon 'stacking', syntactically-correct names for each RasterLayer are automatically generated from the file_paths.

### Indexing

Indexing of Raster objects is provided by several methods:

- Raster[keys] : subsets a Raster using key-based indexing based on the names of the RasterLayers. If a single key is
supplied then a RasterLayer is returned, otherwise a Raster object is returned contained the subsetted layers.
- Raster.iloc[int, list, tuple, slice] : subsets a Raster using integer-based indexing or slicing. If a single key is
supplied then a RasterLayer is returned, otherwise a Raster object is returned contained the subsetted layers.
- Raster.name : attribute names can be used directly, and always returns a single RasterLayer object.

RasterLayer indexing which returns a RasterLayer:

```
# index by integer position
rasterlayer = stack.iloc[0]
rasterstack = stack.iloc[0:3]

# index by name
rasterlayer = stack['lsat7_2000_10']
rasterstack = stack[('lsat7_2000_10', 'lsat7_2000_20')]

# index by atttribute
rasterlayer = stack.lsat7_2000_10
```

Iterate through RasterLayers:

```
for name, layer in stack:
    print(layer)
```

Subset a Raster object:

```
subset_raster = stack[['lsat7_2000_10', 'lsat7_2000_70']]
subset_raster.names
```

Replace a RasterLayer with another:

```
stack.iloc[0] = Raster(nc.band7).iloc[0]
```

Append layers from another Raster to the stack. Note that this occurs in-place. Duplicate names are automatically given
a suffix:

```
stack.append(Raster(nc.band7))
stack.names
```

Rename RasterLayers using a dict of old_name : new_name pairs:

```
stack.names
stack.rename({'lsat7_2000_30': 'new_name'}, in_place=True)
stack.names
stack.new_name
stack['new_name']
stack.loc['new_name']
```

We can also change all of the column names by replacing them:

```
stack.names = ["band1", "band2", "band3", "band4", "band5", "band7"]
```

Drop a RasterLayer:

```
stack.names
stack.drop(labels='band1', in_place=True)
stack.names
```

Save a Raster:

```
tmp_tif = tempfile.NamedTemporaryFile().name + '.tif'
newstack = stack.write(file_path=tmp_tif, nodata=-9999)
newstack.band2.read()
```

### Plotting

Basic plotting has been added to as a method to RasterLayer and Raster options. More controls on plotting will be added
in the future. Currently you can set a matplotlib cmap for each RasterLayer using the RasterLayer.cmap attribute. 

Plot a single RasterLayer:

```
from matplotlib.colors import Normalize

stack = Raster(predictors)
stack.lsat7_2000_10.cmap = 'plasma'
stack.lsat7_2000_10.norm = Normalize(20, 210)
stack.lsat7_2000_10.plot()
```

Plot all RasterLayers in a Raster object:

```
stack.plot()
```

### Integration with Pandas

Data from a Raster object can converted into a Pandas dataframe, with each pixel representing by a row, and columns
reflecting the x, y coordinates and the values of each RasterLayer in the Raster object:

```
df = stack.to_pandas(max_pixels=50000, resampling='nearest')
df.head()
```

The original raster is up-sampled based on max_pixels and the resampling method, which uses all of resampling methods
available in the underlying rasterio library for decimated reads. The Raster.to_pandas method can be useful for plotting
datasets, or combining with a library such as plotnine to create ggplot2-style plots of stacks of RasterLayers:

```
from plotnine import *
(ggplot(df.melt(id_vars=['x', 'y']), aes(x='x', y='y', fill='value')) +
geom_tile() + facet_wrap('variable'))
```

## A Machine Learning Workflow

### Extract Training Data

Load some training data in the form of polygons, points and labelled pixels in geopandas GeoDataFrame objects. We will
also generate some line geometries by converting the polygon boundaries into linestrings. All of these geometry types
can be used to spatially query pixel values in a Raster object, however each GeoDataFrame must contain only one type of
geometry (i.e. either shapely points, polygons or linestrings).

```
training_py = geopandas.read_file(nc.polygons)
training_pt = geopandas.read_file(nc.points)
training_px = rasterio.open(nc.labelled_pixels)
training_lines = deepcopy(training_py)
training_lines['geometry'] = training_lines.geometry.boundary
```

Show training data points and a single raster band using numpy and matplotlib:

```
stack = Raster(predictors)
plt.imshow(stack.lsat7_2000_70.read(masked=True),
           extent=rasterio.plot.plotting_extent(stack.lsat7_2000_70))
plt.scatter(x=training_pt.bounds.iloc[:, 0],
            y=training_pt.bounds.iloc[:, 1],
            s=2, color='black')
plt.show()
```

Pixel values in the Raster object can be spatially queried using the `extract_vector` and `extract_raster`
methods. In addition, the `extract_xy` method can be used to query pixel values using a 2d array of x and y
coordinates.

The `extract_vector` method accepts a Geopandas GeoDataFrame as the `gdf` argument. For
GeoDataFrames containing shapely point geometries, the closest pixel to each point is sampled. For shapely polygon
geometries, all pixels whose centres are inside the polygon are sampled. For shapely linestring geometries, every pixel
touched by the line is sampled. For all geometry types, pixel values are queries for each geometry separately. This
means that overlapping polygons or points that fall within the same pixel with cause the same pixel to be sampled
multiple times.

By default, the extract functions return a Geopandas GeoDataFrame of point geometries and the DataFrame containing the
extracted pixels, with the column names set by the names of the raster datasets in the Raster object. The user can also
use the `return_array=True` argument, which instead of returning a DataFrame will return three masked numpy arrays
(id, X, coordinates) containing geodataframe indices, the extracted pixel values, and the spatial coordinates of the sampled
pixels. These arrays are masked arrays with nodata values in the RasterStack datasets being masked.

The `extract_raster` method can also be used to spatially query pixel values from a Raster object using another
raster containing labelled pixels. This raster has to be spatially aligned with the Raster object. This method also returns
the values of the labelled pixels along with the queried pixel values.

```
# Create a training dataset by extracting the raster values at the training point locations:
df_points = stack.extract_vector(training_pt)
df_polygons = stack.extract_vector(training_py)
df_lines = stack.extract_vector(training_lines)
df_raster = stack.extract_raster(training_px)

df_points.head()

# join the extracted pixel data back with the training data
df_points = df_points.droplevel(0).merge(
    training_pt.loc[:, ("id")], 
    left_index=True, 
    right_index=True
)
df_points = df_points.dropna()
```

### Model Training

Next we can train a logistic regression classifier:

```
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
X = df_points.drop(columns=["geometry", "id"])
y = df_points.id
lr.fit(X, y)
````

After defining a classifier, a typical step consists of performing a cross-validation to evaluate the performance of the
model. Scikit-learn provides the cross_validate function for this purpose. In comparison to non-spatial data, spatial
data can be spatially correlated, which potentially can mean that geographically proximal samples may not represent
independent samples if they are within the autocorrelation range of some of the predictors. This will lead to overly
optimistic performance measures if samples in the training dataset / cross-validation partition are strongly spatially
correlated with samples in the test dataset / cross-validation partition.

In this case, performing cross-validation using groups is useful, because these groups can represent spatial clusters of
training samples, and samples from the same group will never occur in both the training and test partitions of a
cross-validation. An example of creating random spatial clusters from point coordinates is provided here:

```
# spatial cross-validation
from sklearn.cluster import KMeans

# create 10 spatial clusters based on clustering of the training data point x,y coordinates
clusters = KMeans(n_clusters=34, n_jobs=-1)
clusters.fit(df_points.geometry.bounds.iloc[:, 0:2])

# cross validate
scores = cross_validate(
  lr, X, y, groups=clusters.labels_,
  scoring='accuracy',
  cv=3,  n_jobs=1)
scores['test_score'].mean()
```

### Raster Prediction

Prediction on the Raster object is performed using the `predict` and `predict_proba` methods. The `estimator` is the only required
argument. If the `file_path` argument is not specified then the result is automatically written to a temporary file.
The predict method returns an rasterio.io.DatasetReader object which is open. For probability prediction,
`indexes` can also be supplied if you only want to output the probabilities for a particular class, or list of
classes, by supplying the indices of those classes:

```
# prediction
result = stack.predict(estimator=lr, dtype='int16', nodata=0)
result_probs = stack.predict_proba(estimator=lr)

# plot classification result
result.plot()
plt.show()

# plot class probabilities
result_probs.plot()
plt.show()
```

## Sampling Tools

For many spatial models, it is common to take a random sample of the predictors to represent a single class (i.e. an
environmental background or pseudo-absences in a binary classification model). The sample function is supplied in the
sampling module for this purpose:
```
# extract training data using a random sample
df_rand = stack.sample(size=100, random_state=1)
df_rand.plot()
```

The sample function also enables stratified random sampling based on passing a categorical raster dataset to the strata
argument. The categorical raster should spatially overlap with the dataset to be sampled, but it does not need to be of
the same grid resolution. This raster should be passed as another `Raster` dataset containing a single categorical layer:

```
strata = Raster(nc.strata)
df_strata = stack.sample(size=5, strata=strata, random_state=1)
df_strata = df_strata.dropna()

fig, ax = plt.subplots()
ax.imshow(strata.read(1, masked=True), extent=rasterio.plot.plotting_extent(strata), cmap='tab10')
df_strata.plot(ax=ax, markersize=20, color='white')
plt.show()
```

## Vector Data Tools

In some cases, we don't need all of the training data, but rather would spatially thin a point dataset. The
filter_points function performs point-thinning based on a minimum distance buffer on a geopandas dataframe containing
point geometries:

```
from pyspatialml.vector import filter_points

thinned_points = filter_points(training_pt, min_dist=500, remove='first')
thinned_points.shape
```

We can also generate random points within polygons using the get_random_point_in_polygon function. This requires a
shapely POLYGON geometry as an input, and returns a shapely POINT object:

```
from pyspatialml.vector import get_random_point_in_polygon

# generate 5 random points in a single polygon
random_points = [get_random_point_in_polygon(training_py.geometry[0]) for i in range(5)]

# convert to a GeoDataFrame
random_points = geopandas.GeoDataFrame(
  geometry=geopandas.GeoSeries(random_points))
```
