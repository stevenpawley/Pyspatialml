from sklearn.base import (BaseEstimator, RegressorMixin, ClassifierMixin, clone, is_classifier, is_regressor)
from sklearn.utils.extmath import weighted_mode
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy


class SpatialLagBase(ABC, BaseEstimator):
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
    base_estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface. Either estimator
        needs to provide a score function, or scoring must be passed.

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

    kernel_params : dict, default=None
        Additional keyword arguments to pass to a custom kernel function.

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
            kernel_params=None,
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
        self.kernel_params = kernel_params
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

    @abstractmethod
    def _validate_base_estimator(self):
        pass

    def fit(self, X, y):
        """Fit the base_estimator with features from X {n_samples, n_features} and with an
        additional spatially lagged variable added to the right-most column of the
        training data.

        During fitting, the k-neighbors to each training point are used to
        estimate the spatial lag component. The training point is not included in the
        calculation, i.e. the training point is not considered its own neighbor.

        Parameters
        ----------
        X : array-like of sample {n_samples, n_features} using for model fitting
            The training input samples

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).
        """
        # some checks
        self.base_estimator = clone(self.base_estimator)
        self._validate_base_estimator()

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
        """Predict method for spatial lag models.

        Augments new osbservations with a spatial lag variable created from a weighted
        mean/mode (regression/classification) of k-neighboring observations.

        Parameters
        ----------
        X : array-like of sample {n_samples, n_features}
            New samples for the prediction.

        y : None
            Not used.
        """
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

        elif self.weights == "uniform":
            new_X = self._uniform_weighting(neighbor_vals)

        elif callable(self.weights):
            new_X = self._custom_weighting(neighbor_vals, neighbor_dist)

        # fit base estimator on augmented data
        preds = self.base_estimator.predict(np.column_stack((X, new_X)))

        return preds


class SpatialLagRegressor(RegressorMixin, SpatialLagBase):
    @staticmethod
    def _distance_weighting(neighbor_vals, neighbor_dist):
        neighbor_weights = 1 / neighbor_dist
        X = np.ma.average(neighbor_vals, weights=neighbor_weights, axis=1)
        return X

    @staticmethod
    def _uniform_weighting(neighbor_vals):
        X = np.ma.average(neighbor_vals, axis=1)
        return X

    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        neighbor_weights = self.weights(neighbor_dist, **self.kernel_params)
        new_X = np.ma.average(neighbor_vals, weights=neighbor_weights, axis=1)
        return new_X

    def _validate_base_estimator(self):
        if not is_regressor(self.base_estimator):
            raise ValueError(
                "'base_estimator' parameter should be a regressor. Got {}"
                    .format(self.base_estimator)
            )


class SpatialLagClassifier(ClassifierMixin, SpatialLagBase):
    @staticmethod
    def _distance_weighting(neighbor_vals, neighbor_dist):
        neighbor_weights = 1 / neighbor_dist
        X = weighted_mode(neighbor_vals, neighbor_weights, axis=1)
        return X

    @staticmethod
    def _uniform_weighting(neighbor_vals):
        X = mode(neighbor_vals, axis=1)
        return X

    def _custom_weighting(self, neighbor_vals, neighbor_dist):
        neighbor_weights = self.weights(neighbor_dist, **self.kernel_params)
        new_X = weighted_mode(neighbor_vals, neighbor_weights, axis=1)
        return new_X

    def _validate_base_estimator(self):
        if not is_classifier(self.base_estimator):
            raise ValueError(
                "'base_estimator' parameter should be a classifier. Got {}"
                    .format(self.base_estimator)
            )


class ThresholdClassifierCV(BaseEstimator, ClassifierMixin):
    """
    Metaclassifier to perform cutoff threshold optimization

    This implementation is restricted to binary classification problems

    Notes
    -----

    - The training data are partitioned in k-1, and k sets
    - The metaclassifier trains the BaseEstimator on the k-1 partitions
    - The Kth paritions are used to determine the optimal cutoff taking
      the mean of the thresholds that maximize the scoring metric
    - The optimal cutoff is applied to all classifier predictions
    """

    def __init__(self, estimator, thresholds=np.arange(0.1, 0.9, 0.01),
                 scoring=None, refit=False, cv=3, random_state=None):

        """Initialize a ThresholdClassifierCV instance

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        thresholds : threshold values to search for optimal cutoff, for
            example a list or array of cutoff thresholds to use for scoring

        scoring : callable, dict
            A callable or dict of key : callable pairs of scoring metrics to
            evaluate at the cutoff thresholds

        refit : string, or None
            String specifying the key name of the metric to use to determine
            the optimal cutoff threshold. Only required when multiple scoring
            metrics are used

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.

            Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.

            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        random_state : int, RandomState instance or None, optional (default=0)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Notes
        -----
        Rules for creating a custom scikit-learn estimator
        - All arguments of __init__must have default value, so it's possible to initialize
           the classifier just by typing MyClassifier()
        - No confirmation of input parameters should be in __init__ method
           That belongs to fit method.
        - All arguments of __init__ method should have the same name as they will have as the
           attributes of created object
        - Do not take data as argument here! It should be in fit method

        TODO
        ----
        Parallelize the cross validation loop used to find the optimal threshold
        in the fit method

        Change the behaviour of the scoring parameters so that it accepts arguments
        in the same manner as scikit learn functions such as GridSearchCV, i.e. it
        accepts string, callable, list/tuple, or dict
        """

        # get dict of arguments from function call
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _find_threshold(self, X, y=None):
        """
        Finds the optimal cutoff threshold based on maximizing/minimizing the
        scoring method
        """

        estimator = self.estimator
        thresholds = self.thresholds
        scorers = self.scorers_
        refit = self.refit

        y_pred = estimator.predict_proba(X)
        scores = dict([(dct, np.zeros((len(thresholds)))) for dct in scorers])

        for i, cutoff in enumerate(thresholds):
            for name, scorer in scorers.items():
                scores[name][i] = self._scorer_cutoff(y, y_pred, scorer, cutoff)

        top_score = scores[refit][scores[refit].argmax()]
        top_threshold = thresholds[scores[refit].argmax()]
        return top_threshold, top_score, scores

    def score(self, X, y):
        """
        Overloading of classifier score method
        score method is required for compatibility with GridSearch
        The scoring metric should be one that can be maximized (bigger=better)
        """

        _threshold, score, _threshold_scores = self._find_threshold(X, y)

        return score

    @staticmethod
    def _scorer_cutoff(y, y_pred, scorer, cutoff):
        """
        Helper method to binarize the probability scores based on a cutoff
        """

        y_pred = (y_pred[:, 1] > cutoff).astype(int)
        return scorer(y, y_pred)

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run fit method with all sets of parameters

        Args
        ----
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning

        groups : array-like, shape = [n_samples], optional
            Training vector groups for cross-validation

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Notes
        -----

        Rules

        - Parameters are checked during the fit method
        - New attributes created during fitting should end in _, i.e. fitted_
        - Fit method needs to return self for compatibility reasons with sklearn
        - The response vector, i.e. y, should be initiated with None
        """

        # check estimator and cv methods are valid
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        # check for binary response
        if len(np.unique(y)) > 2:
            raise ValueError('Only a binary response vector is currently supported')

        # check that scoring metric has been specified
        if self.scoring is None:
            raise ValueError('No score function is defined')

        # convert scoring callables to dict of key: scorer pairs
        if callable(self.scoring):
            self.scorers_ = {'score': self.scoring}
            self.refit = 'score'

        elif isinstance(self.scoring, dict):
            self.scorers_ = self.scoring

            if self.refit is False:
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to determine which scoring method "
                                 "is used to optimize the classifiers "
                                 "cutoff threshold")

        # cross-validate the probability threshold
        self.best_threshold_ = 0
        self.threshold_scores_ = dict([(k, np.zeros((len(self.thresholds)))) for k in self.scorers_])

        for train, cal in cv.split(X, y, groups):
            X_train, y_train, X_cal, y_cal = X[train], y[train], X[cal], y[cal]

            if groups is not None:
                groups_train, groups_cal = groups[train], groups[cal]
                estimator.fit(X_train, y_train, groups_train, **fit_params)
            else:
                estimator.fit(X_train, y_train, **fit_params)

            # find cutoff threshold on calibration set
            # unless a single cutoff threshold is specified, which sets the classifier to this
            # threshold
            if isinstance(self.thresholds, (list, np.array)):
                best_threshold, _score, threshold_scores = self._find_threshold(X_cal, y_cal)

                # sum the scores
                self.best_threshold_ += best_threshold
                self.threshold_scores_ = {
                    key: (self.threshold_scores_[key] + threshold_scores[key]) for key in self.threshold_scores_}

                # average the scores per cross validation fold
                self.best_threshold_ /= cv.get_n_splits()
                self.threshold_scores_ = dict([(k, v / cv.get_n_splits()) for (k, v) in self.threshold_scores_.items()])

            else:
                self.best_threshold_ = self.threshold

        self.classes_ = self.estimator.classes_

    def predict(self, X, y=None):
        y_score = self.estimator.predict_proba(X)
        return np.array(y_score[:,1] >= self.best_threshold_)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
