Estimators
**********

Spatial Lag Estimators
======================

A meta-estimator to perform spatial lag regression/classification by using a
weighted mean/mode of the values of the K-neighboring observations to augment
the `base_estimator`. The weighted mean/mode of the surrounding observations
are appended as a new feature to the right-most column in the training data.

For classification, the `SpatialLagClassifier` class is used, where the spatially
lagged feature is created from a weighted mode of the surrounding observations. For
regression the `SpatialLagRegressor` is used, where the spatially lagged feature is 
created from a weighted mean of the surrounding observations.

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
    
    feature_indices : list, default=None
        By default, the nearest neighbors are determined from the distance metric calculated
        using all of the features. If `feature_indices` are supplied then the distance
        calculation is restricted to the specific column indices. For spatial data, these
        might represent the x,y coordinates for example.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors. See Glossary
        for more details.

Methods
-------

    fit(self, X, y)
        Fit the base_estimator with features from X {n_samples, n_features} and with an
        additional spatially lagged variable added to the right-most column of the 
        training data. During fitting, the k-neighbors to each training point are used to
        estimate the spatial lag component. The training point is not included in the
        calculation, i.e. the training point is not considered its own neighbor.
    
    predict(self, X, y=None)
        Prediction method for new data X.

ThresholdClassifierCV
=====================

A meta-classifier to perform cutoff threshold optimization for binary 
classification models.

During the `fit` method, the training data are partitioned into k-1, and k sets.
The metaclassifier trains the `base_estimator` on the k-1 partitions, and 
the Kth paritions are used to determine the optimal cutoff, taking the mean
of the thresholds that maximize the scoring metric. The optimal cutoff is 
threshold is them applied to all classifier predictions when using the
`predict` method.

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

Methods
-------
    
    fit(self, X, y=None, groups=None, **fit_params)

    predict(self, X, y=None)

    predict_proba(self, X, y=None)

