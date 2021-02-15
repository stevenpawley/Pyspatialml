from pyspatialml import Raster
from pyspatialml.datasets import nc
from copy import deepcopy
import geopandas
import rasterio.plot
import matplotlib.pyplot as plt

training_py = geopandas.read_file(nc.polygons)
training_pt = geopandas.read_file(nc.points)
training_px = rasterio.open(nc.labelled_pixels)
training_lines = deepcopy(training_py)
training_lines['geometry'] = training_lines.geometry.boundary

predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
stack = Raster(predictors)

# Extract data from rasters at the training point locations:
df_points = stack.extract_vector(training_pt)
df_polygons = stack.extract_vector(training_py)
df_lines = stack.extract_vector(training_lines)

# Join the extracted values with other columns from the training data
df_points["id"] = training_pt["id"].values
df_points = df_points.dropna()
df_points.head()

df_polygons = df_polygons.merge(
    right=training_py.loc[:, ["label", "id"]],
    left_on="geometry_idx",
    right_on="index",
    right_index=True
)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define the classifier with standardization of the input features in a
# pipeline
lr = Pipeline(
    [('scaling', StandardScaler()),
     ('classifier', LogisticRegressionCV(n_jobs=-1))])

# remove NaNs from training data
df_polygons = df_polygons.dropna()

# fit the classifier
X = df_polygons.drop(columns=["id", "label", "geometry"]).values
y = df_polygons["id"].values
lr.fit(X, y)

# prediction
result = stack.predict(estimator=lr, in_memory=True, nodata=0)
result_probs = stack.predict_proba(estimator=lr, in_memory=True)

# plot classification result
result.iloc[0].cmap = "Dark2"
result.iloc[0].categorical = True
result.iloc[0].dtype
result.plot()
plt.show()
