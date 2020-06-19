import numpy as np
import inspect
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.preprocessing import binarize
from collections import OrderedDict


class CrossValidateThreshold(object):
    """
    Perform cross-validation and calculate scores using a cutoff threshold that
    attains a minimum true positive rate.
    """

    def __init__(self, estimator, scoring, cv=3, positive=1, n_jobs=1):
        """Initiate a class instance

        Args
        ----
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        scoring : dict
            Dict containing name and scoring callable

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

        positive : int, default=1
            Index of the positive class

        n_jobs : int, default=1
            Number of processing cores for multiprocessing
        """

        self.scoring = scoring
        self.cv = cv
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.positive = positive

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
        """

        # check estimator and cv methods are valid
        self.cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        # check for binary response
        if len(np.unique(y)) > 2:
            raise ValueError('Only a binary response vector is currently supported')

        # check that scoring metric has been specified
        if self.scoring is None:
            raise ValueError('No score function is defined')

        # perform cross validation prediction
        self.y_pred_ = cross_val_predict(
            estimator=self.estimator, X=X, y=y, groups=groups, cv=self.cv,
            method='predict_proba', n_jobs=self.n_jobs, **fit_params)
        self.y_true = y

        # add fold id to the predictions
        self.test_idx_ = [indexes[1] for indexes in self.cv.split(X, y, groups)]

    def score(self, tpr_threshold=None, cutoff_threshold=None):
        """
        Calculates the scoring metrics using a cutoff threshold that attains a true positive rate
        that is equal or greater than the desired tpr_threshold

        Args
        ----
        tpr_threshold : float
            Minimum true positive rate to achieve
        cutoff_threshold : float
            As an alternative to using a minimum true positive, a probability cutoff threshold
            can be specified to calculate the scoring
        """

        if tpr_threshold is None and cutoff_threshold is None:
            raise ValueError('Either tpr_threshold or cutoff_threshold must be specified')

        scores = OrderedDict((k, []) for (k, v) in self.scoring.items())
        self.thresholds_ = []
        self.tpr_ = []
        self.fpr_ = []
        self.roc_thresholds_ = []

        for idx in self.test_idx_:
            # split fold
            y_true = self.y_true[idx]
            y_pred_ = self.y_pred_[idx, :]

            # get roc curve data
            fpr, tpr, thresholds = roc_curve(
                y_true, y_pred_[:, self.positive])

            self.fpr_.append(fpr)
            self.tpr_.append(tpr)
            self.roc_thresholds_.append(thresholds)

            # calculate cutoff that produces tpr >= threshold
            if cutoff_threshold is None:
                opt_threshold = thresholds[np.where(tpr >= tpr_threshold)[0].min()]
                self.thresholds_ = np.append(self.thresholds_, opt_threshold)
            else:
                opt_threshold = cutoff_threshold

            # calculate performance metrics
            y_pred_opt = binarize(y_pred_, opt_threshold)

            # calculate scores
            for name, score_func in self.scoring.items():
                score_func = self.scoring[name]
                scores[name] = np.append(scores[name], score_func(y_true, y_pred_opt[:, self.positive]))

        return scores


def spatial_loocv(estimator, X, y, coordinates, size, radius,
                  random_state=None):
    """
    Spatially buffered leave-One-out cross-validation
    Uses a circular spatial buffer to separate samples that are used to train
    the estimator, from samples that are used to test the prediction accuracy

    Args
    ----
    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning

    coordinates : 2d-array like
        Spatial coordinates, usually as xy, that correspond to each sample in X

    size : int
        Sample size to process (number of leave-one-out runs)

    radius : int or float
        Radius for the spatial buffer around test point

    random_state : int
        random_state is the seed used by the random number generator

    Returns
    -------
    y_test : 1d array-like
        Response variable values in the test partitions

    y_pred : 1d array-like
        Predicted response values by the estimator

    y_prob : array-like
        Predicted probabilities by the estimator

    Notes
    -----
    This python function was adapted from R code
    https://davidrroberts.wordpress.com/2016/03/11/spatial-leave-one-out-sloo-cross-validation/
    """

    # determine number of classes and features
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    rstate = RandomState(random_state)

    # randomly select the testing points
    ind = rstate.choice(range(X.shape[0]), size)
    X_test = X[ind, :]
    y_test = y[ind]
    coordinates_test = coordinates[ind, :]

    # variables to store predictions and probabilities
    y_pred = np.empty((0,))
    y_prob = np.empty((0, n_classes))

    # loop through the testing points
    for i in range(size):
        # Training data (test point & buffer removed)
        # empty numpy arrays to append training that is > radius from test loc
        X_train = np.empty((0, n_features))
        y_train = np.empty((0))

        # loop through each point in the original training data
        # and append to X_train, y_train if coordinates > minimum radius
        for j in range(X.shape[0]):
            if math.sqrt((coordinates[j, 0] - coordinates_test[i, 0])**2 +
                         (coordinates[j, 1] - coordinates_test[i, 1])**2) \
                          > radius:
                X_train = np.vstack((X_train, X[j]))
                y_train = np.append(y_train, y[j])

        # Build the model
        estimator.fit(X_train, y_train)

        # Predict on test point
        y_pred = np.append(y_pred, estimator.predict(X_test[i].reshape(1, -1)))
        y_prob = np.vstack((
            y_prob, estimator.predict_proba(X_test[i].reshape(1, -1))))

    return y_test, y_pred, y_prob
