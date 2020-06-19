from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.extmath import weighted_mode
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy


class SpatialLagBase(ABC, BaseEstimator, RegressorMixin):
    """Base class for spatial lag estimators.

    A spatial lag estimator uses a weighted mean/mode of the values of the K-neighboring
    observations to augment the base_estimator. The weighted mean/mode of the
    surrounding observations are appended as a new feature to the right-most column in
    the training data.

    The K-neighboring observations are determined using the distance metric specified in
    the `metric` argument. The default metric is minkowski, and with p=2 is equivalent
    to the standard Euclidean metric.

    Parameters
    ----------
    base_estimator : 

    n_neighbors : int, default = 7
        Number of neighbors to use by default for kneighbors queries.
    
    weights : {‘uniform’, ‘distance’} or callable, default=’distance’
        Weight function used in prediction. Possible values:

            - ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
            - ‘distance’ : weight points by the inverse of their distance. in this case,
              closer neighbors of a query point will have a greater influence than
              neighbors which are further away.
            - [callable] : a user-defined function which accepts an array of distances,
              and returns an array of the same shape containing the weights.

    radius : float, default=1.0
        Range of parameter space to use by default for radius_neighbors queries.

    algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        Algorithm used to compute the nearest neighbors:

            - ‘ball_tree’ will use BallTree
            - ‘kd_tree’ will use KDTree
            - ‘brute’ will use a brute-force search.
            - ‘auto’ will attempt to decide the most appropriate algorithm based on the
              values passed to fit method.
            - Note: fitting on sparse input will override the setting of this parameter,
              using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree. This can affect the speed of the construction
        and query, as well as the memory required to store the tree. The optimal value depends
        on the nature of the problem.
    
    metric : str or callable, default=’minkowski’
        The distance metric to use for the tree. The default metric is minkowski, and
        with p=2 is equivalent to the standard Euclidean metric. See the documentation
        of DistanceMetric for a list of available metrics. If metric is “precomputed”,
        X is assumed to be a distance matrix and must be square during fit. X may be a
        sparse graph, in which case only “nonzero” elements may be considered neighbors.

    p : int, default=2
        Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances.
        When p = 1, this is equivalent to using manhattan_distance (l1), and
        euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    
    feature_indices : list, default=None
        By default, the nearest neighbors are determined from the distance metric calculated
        using all of the features. If `feature_indices` are supplied then the distance
        calculation is restricted to the specific column indices. For spatial data, these
        might represent the x,y coordinates for example.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors. See Glossary
        for more details.
    """
    def __init__(
        self,
        base_estimator,
        n_neighbors=7,
        weights="distance",
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        feature_indices=None,
        n_jobs=1,
    ):

        self.base_estimator = base_estimator
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.feature_indices = feature_indices
        self.n_jobs = n_jobs

        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        
        self.y_ = None

    @abstractmethod
    def _distance_weighting(self, neighbor_vals, neighbor_dist):
        pass        

    @abstractmethod
    def _uniform_weighting(self, neighbor_vals):
        pass
    
    @abstractmethod
    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        pass
    
    def fit(self, X, y):
        # some checks
        self.base_estimator = clone(self.base_estimator)

        self.y_ = deepcopy(y)
        distance_data = deepcopy(X)
        
        # use only selected columns in the data for the distances
        if self.feature_indices is not None:
            distance_data = distance_data[:, self.feature_indices]

        # fit knn and get values of neighbors
        self.knn.fit(distance_data)
        neighbor_dist, neighbor_ids = self.knn.kneighbors()

        # mask any zero distances
        neighbor_dist = np.ma.masked_equal(neighbor_dist, 0)
        
        # get y values of neighbouring points
        neighbor_vals = np.array([y[i] for i in neighbor_ids])
        neighbor_vals = np.ma.masked_array(neighbor_vals, mask=neighbor_dist.mask)
        
        # calculated weighted means
        if self.weights == "distance":
            new_X = self._distance_weighting(neighbor_vals, neighbor_dist)
        
        elif self.weights == "uniform":
            new_X = self._uniform_weighting(neighbor_vals)
        
        elif callable(self.weights):
            new_X = self._custom_weighting(neighbor_vals, neighbor_dist)
        
        # fit base estimator on augmented data
        self.base_estimator.fit(np.column_stack((X, new_X)), y)

        return self

    def predict(self, X, y=None):
        distance_data = deepcopy(X)

        # use only selected columns in the data for the distances
        if self.feature_indices is not None:
            distance_data = distance_data[:, self.feature_indices]

        # get distances from training points to new data
        neighbor_dist, neighbor_ids = self.knn.kneighbors(X=distance_data)
        
        # mask zero distances
        neighbor_dist = np.ma.masked_equal(neighbor_dist, 0)
        
        # get values of closest training points to new data
        neighbor_vals = np.array([self.y_[i] for i in neighbor_ids])
        neighbor_vals = np.ma.masked_array(neighbor_vals, mask=neighbor_dist.mask)

        # calculated weighted means
        if self.weights == "distance":
            new_X = self._distance_weighting(neighbor_vals, neighbor_dist)
        
        if self.weights == "uniform":
            new_X = self._uniform_weighting(neighbor_vals)
        
        elif callable(self.weights):
            new_X = self._custom_weighting(neighbor_vals, neighbor_dist)
        
        # fit base estimator on augmented data
        preds = self.base_estimator.predict(np.column_stack((X, new_X)))

        return preds


class SpatialLagRegressor(SpatialLagBase):    
    @staticmethod
    def _distance_weighting(neighbor_vals, neighbor_dist):
            neighbor_weights = 1 / neighbor_dist
            X = np.ma.average(neighbor_vals, weights=neighbor_weights, axis=1)
            return X
        
    @staticmethod
    def _uniform_weighting(self, neighbor_vals):
        X = np.ma.average(neighbor_vals, axis=1)
        return X

    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        neighbor_weights = self.weight(neighbor_dist)
        new_X = np.ma.average(neighbor_vals, weights=neighbor_weights, axis=1)


class SpatialLagClassifier(SpatialLagBase):    
    @staticmethod
    def _distance_weighting(neighbor_vals, neighbor_dist):
            neighbor_weights = 1 / neighbor_dist
            X = weighted_mode(neighbor_vals, neighbor_weights, axis=1)
            return X
        
    @staticmethod
    def _uniform_weighting(self, neighbor_vals):
        X = mode(neighbor_vals, axis=1)
        return X

    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        neighbor_weights = self.weight(neighbor_dist)
        new_X = weighted_mode(neighbor_vals, neighbor_weights, axis=1)
