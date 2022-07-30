import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import weighted_mode


class KNNTransformer(BaseEstimator, TransformerMixin):
    """Transformer to generate new lag features by weighted aggregation
    of K-neighboring observations.

    A lag transformer uses a weighted mean/mode of the values of the
    K-neighboring observations to generate new lagged features. The
    weighted mean/mode of the surrounding observations are appended
    as a new feature to the right-most column in the training data.

    The K-neighboring observations are determined using the distance
    metric specified in the `metric` argument. The default metric is
    minkowski, and with p=2 is equivalent to the standard Euclidean
    metric.

    Parameters
    ----------
    n_neighbors : int, default = 7
        Number of neighbors to use by default for kneighbors queries.

    weights : {‘uniform’, ‘distance’} or callable, default=’distance’
        Weight function used in prediction. Possible values:

            - ‘uniform’ : uniform weights. All points in each
              neighborhood are weighted equally.
            - ‘distance’ : weight points by the inverse of their
               distance. In this case, closer neighbors of a query
               point will have a greater influence than neighbors
               which are further away.
            - [callable] : a user-defined function which accepts an
              array of distances, and returns an array of the same
              shape containing the weights.

    measure : {'mean', 'mode'}
        Function that is used to apply the weights to `y`. Use 'mean'
        if the target variable is continuous and 'mode' if the target
        variable is discrete.

    radius : float, default=1.0
        Range of parameter space to use by default for radius_neighbors
        queries.

    algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        Algorithm used to compute the nearest neighbors:

            - ‘ball_tree’ will use BallTree
            - ‘kd_tree’ will use KDTree
            - ‘brute’ will use a brute-force search.
            - ‘auto’ will attempt to decide the most appropriate
               algorithm based on the values passed to fit method.
            - Note: fitting on sparse input will override the setting
              of this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : str or callable, default=’minkowski’
        The distance metric to use for the tree. The default metric is
        minkowski, and with p=2 is equivalent to the standard
        Euclidean metric. See the documentation of DistanceMetric for
        a list of available metrics. If metric is “precomputed”, X is
        assumed to be a distance matrix and must be square during fit.
        X may be a sparse graph, in which case only “nonzero” elements
        may be considered neighbors.

    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this
        is equivalent to using manhattan_distance (l1), and
        euclidean_distance (l2) for p = 2. For arbitrary p,
        minkowski_distance (l_p) is used.

    normalize : bool, default=True
        Whether to normalize the inputs using
        sklearn.preprocessing.Normalizer

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    kernel_params : dict, default=None
        Additional keyword arguments to pass to a custom kernel
        function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. None
        means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors. See Glossary for more details.
    """

    def __init__(
        self,
        n_neighbors=7,
        weights="distance",
        measure="mean",
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        normalize=True,
        metric_params=None,
        kernel_params=None,
        n_jobs=1,
    ):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.measure = measure
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.kernel_params = kernel_params
        self.normalize = normalize
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

    def fit(self, X, y=None):
        """Fit the base_estimator with features from X
        {n_samples, n_features} and with an additional spatially lagged
        variable added to the right-most column of the training data.

        During fitting, the k-neighbors to each training point are
        used to estimate the spatial lag component. The training point
        is not included in the calculation, i.e. the training point is
        not considered its own neighbor.

        Parameters
        ----------
        X : array-like of sample {n_samples, n_features} using for model
            fitting The training input samples

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real
            numbers in regression).
        """
        # some checks
        if self.kernel_params is None:
            self.kernel_params = {}

        if y.ndim == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[1]

        # fit knn and get values of neighbors
        if self.normalize is True:
            scaler = Normalizer()
            X = scaler.fit_transform(X)
            self.scaler_ = scaler

        self.knn.fit(X)
        self.y_ = y.copy()

        return self

    def transform(self, X, y=None):
        """Transform method for spatial lag models.

        Augments new observations with a spatial lag variable created
        from a weighted mean/mode (regression/classification) of
        k-neighboring observations.

        Parameters
        ----------
        X : array-like of sample {n_samples, n_features}
            New samples for the prediction.

        y : None
            Not used.
        """
        # get distances from training points to new data
        if self.normalize is True:
            X = self.scaler_.transform(X)

        neighbor_dist, neighbor_ids = self.knn.kneighbors(X=X)

        # mask zero distances
        neighbor_dist = np.ma.masked_equal(neighbor_dist, 0)

        # get values of closest training points to new data
        neighbor_vals = np.array([self.y_[i] for i in neighbor_ids])

        # mask neighbor values with zero distances
        mask = neighbor_dist.mask

        if mask.all() == False:
            mask = np.zeros(neighbor_dist.shape, dtype=np.bool)
            mask[:] = False

        if neighbor_vals.ndim == 2:
            neighbor_vals = np.ma.masked_array(neighbor_vals, mask)
        else:
            n_outputs = neighbor_vals.shape[2]
            mask = np.repeat(mask[:, :, np.newaxis], n_outputs, axis=2)
            neighbor_vals = np.ma.masked_array(neighbor_vals, mask=mask)

        # calculated weighted means
        if self.weights == "distance":
            new_X = self._distance_weighting(neighbor_vals, neighbor_dist)

        elif self.weights == "uniform":
            new_X = self._uniform_weighting(neighbor_vals)

        elif callable(self.weights):
            new_X = self._custom_weighting(neighbor_vals, neighbor_dist)

        return np.column_stack((X, new_X))

    def _apply_weights(self, neighbor_vals, neighbor_weights):
        # weighted mean/mode of neighbors for a single regression target
        if neighbor_vals.ndim == 2:
            if self.measure == "mean":
                X = np.ma.average(neighbor_vals, weights=neighbor_weights, axis=1)
            else:
                X, _ = weighted_mode(neighbor_vals, neighbor_weights, axis=1)

        # weighted mean of neighbors for a multi-target regression
        # neighbor_vals = (n_samples, n_neighbors, n_targets)
        else:
            X = np.zeros((neighbor_vals.shape[0], neighbor_vals.shape[2]))

            if self.measure == "mean":
                for i in range(neighbor_vals.shape[-1]):
                    X[:, i] = np.ma.average(
                        neighbor_vals[:, :, i], weights=neighbor_weights, axis=1
                    )
            else:
                for i in range(neighbor_vals.shape[-1]):
                    X[:, i], _ = weighted_mode(
                        neighbor_vals[:, :, i], neighbor_weights, axis=1
                    )

        return X

    def _distance_weighting(self, neighbor_vals, neighbor_dist):
        weights = 1 / neighbor_dist
        return self._apply_weights(neighbor_vals, weights)

    def _uniform_weighting(self, neighbor_vals):
        weights = np.ones((neighbor_vals.shape[0], neighbor_vals.shape[0]))
        return self._apply_weights(neighbor_vals, weights)

    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        weights = self.weights(neighbor_dist, **self.kernel_params)
        return self._apply_weights(neighbor_vals, weights)

    def _distance_weighting(self, neighbor_vals, neighbor_dist):
        weights = 1 / neighbor_dist
        return self._apply_weights(neighbor_vals, weights)

    def _uniform_weighting(self, neighbor_vals):
        weights = np.ones((neighbor_vals.shape[0], neighbor_vals.shape[0]))
        return self._apply_weights(neighbor_vals, weights)

    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        weights = self.weights(neighbor_dist, **self.kernel_params)
        return self._apply_weights(neighbor_vals, weights)


class GeoDistTransformer(BaseEstimator, TransformerMixin):
    """Transformer to add new features based on geographical distances
    to reference locations.

    Parameters
    ----------
    refs : ndarray
        Array of coordinates of reference locations in
        (m, n-dimensional) order, such as {n_locations,
        x_coordinates, y_coordinates, ...} for as many dimensions as
        required. For example to calculate distances to a single x,y,z
        location:

        refs = [-57.345, -110.134, 1012]

        And to calculate distances to three x,y reference locations:

        refs = [
            [-57.345, -110.134],
            [-56.345, -109.123],
            [-58.534, -112.123]
        ]

        The supplied array has to have at least x,y coordinates with a
        (1, 2) shape for a single location.

    minimum : bool, default is False
        Optionally calculate the minimum distance to the combined
        reference locations, resulting in a single new feature,
        rather than a new feature for each individual reference
        location.

    log : bool (opt), default=False
        Optionally log-transform the distance measures.

    Returns
    -------
    X_new : ndarray
        Array of shape (n_samples, n_features) with new geodistance
        features appended to the right-most columns of the array.
    """

    def __init__(self, refs, minimum=False, log=False):
        self.refs = refs
        self.log = log
        self.refs_ = None
        self.minimum = minimum

    def fit(self, X, y=None):
        self.refs_ = np.asarray(self.refs)

        if self.refs_.ndim < 2:
            raise ValueError(
                "`refs` has to be a m,n-dimensional array with  at least two dimensions"
            )

        return self

    def transform(self, X, y=None):
        if self.minimum is False:
            dists = cdist(self.refs_, X).transpose()

        if self.minimum is True:
            tree = cKDTree(self.refs_)
            dists, _ = tree.query(X)

        if self.log is True:
            dists = np.log(dists)

        return np.column_stack((X, dists))


def _apply_transformer(img, transformer):
    img = np.ma.masked_invalid(img)
    mask = img.mask.copy()

    # reshape into 2D array
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    flat_pixels = img.reshape((rows * cols, n_features))
    flat_pixels = flat_pixels.filled(0)

    # predict and replace mask
    result = transformer.transform(flat_pixels)

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result = result.reshape((n_features, rows, cols))
    result = np.ma.masked_array(data=result, mask=mask, copy=True)
    
    return result
