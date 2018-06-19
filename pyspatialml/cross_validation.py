import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import check_cv


class ThresholdClassifierCV(BaseEstimator, ClassifierMixin):
    """Metaclassifier to perform cutoff threshold optimization

    This implementation is restricted to binary classification problems

    Notes:
    1. The training data are partitioned in k-1, and k sets
    2. The metaclassifier trains the BaseEstimator on the k-1 partitions
    3. The Kth paritions are used to determine the optimal cutoff, taking
       the mean of the thresholds that maximize the scoring metric
    4. The optimal cutoff is applied to all classifier predictions

    Usage:
    If a BaseEstimator is to be used with hyperparameter tuning, the order
    of the operations is recommended to be:
        BaseEstimator -> ThresholdClassifierCV -> Tuning(GridSearchCV)
    a. A set of hyperparameters are passed to ThresholdClassifierCV along with
       a training partition
    b. ThresholdClassifierCV splits that training partition into a train and
       calibration set, and trains the base estimator using those hyperparameters
    c. ThresholdClassifierCV then finds the optimal threshold for those
       hyperparameters on the calibration set.
    d. GridSearchCV then tests how well the hyperparameters and threshold
       perform against the test partition"""

    def __init__(self, estimator, thresholds=np.arange(0.1, 0.9, 0.01),
                 scoring='accuracy', refit=False, cv=3, random_state=None, **params):
        # classifier settings
        self.estimator = estimator
        self.thresholds = thresholds
        self.cv = cv
        self.refit = refit
        self.random_state = random_state
        self.scorers = None
        self.set_params(**params)

        # scoring results
        self.best_threshold = None
        self.threshold_scores = None

        # check scoring and convert to dict of key: scorer pairs
        if scoring is None:
            raise ValueError('No score function is defined')

        elif callable(scoring):
            self.scorers = {'score': scoring}
            self.refit = 'score'

        elif isinstance(scoring, dict):
            self.scorers = scoring

            if self.refit is False:
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to determine which scoring method "
                                 "is used to optimize the classifiers "
                                 "cutoff threshold")

    def find_threshold(self, X, y):
        """Finds the optimal cutoff threshold based on maximizing/minimizing the
        scoring method"""

        estimator = self.estimator
        thresholds = self.thresholds
        scorers = self.scorers
        refit = self.refit

        y_pred = estimator.predict_proba(X)
        scores = dict([(dct, np.zeros((len(thresholds)))) for dct in scorers])

        for i, cutoff in enumerate(thresholds):
            for name, scorer in scorers.items():
                scores[name][i] = self.__scorer_cutoff(y, y_pred, scorer, cutoff)

        top_score = scores[refit][scores[refit].argmax()]
        top_threshold = thresholds[scores[refit].argmax()]
        return top_threshold, top_score, scores

    def score(self, X, y, sample_weight=None):
        """Overloading of classifier score method"""

        _threshold, score, _threshold_scores = self.find_threshold(X, y)
        return score[self.refit]

    @staticmethod
    def __scorer_cutoff(y, y_pred, scorer, cutoff):
        """Helper method to binarize the probability scores based on a cutoff"""

        y_pred = (y_pred[:, 1] > cutoff).astype(int)
        return scorer(y, y_pred)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit method with all sets of parameters

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, shape = [n_samples], optional
            Training vector groups for cross-validation
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        # some checks
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        self.best_threshold = 0
        self.threshold_scores = dict(
            [(k, np.zeros((len(self.thresholds)))) for k in self.scorers])

        for train, cal in cv.split(X, y, groups):
            X_train, y_train = X[train], y[train]
            X_cal, y_cal = X[cal], y[cal]

            if groups is not None:
                groups_train, groups_cal = groups[train], groups[cal]
                estimator.fit(X_train, y_train, groups_train, **fit_params)
            else:
                estimator.fit(X_train, y_train, **fit_params)

            # find cutoff threshold on calibration set
            best_threshold, _score, threshold_scores = self.find_threshold(X_cal, y_cal)

            # sum the scores
            self.best_threshold += best_threshold
            self.threshold_scores = {
                key: (self.threshold_scores[key] + threshold_scores[key]) for key in self.threshold_scores}

        self.best_threshold /= cv.get_n_splits()
        self.threshold_scores = dict([(k,v / cv.get_n_splits()) for (k,v) in self.threshold_scores.items()])
        self.classes_ = self.estimator.classes_

    def predict(self, X):
        y_score = self.estimator.predict_proba(X)
        return np.array(y_score[:,1] >= self.best_threshold)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def set_params(self, **params):
        for param_name in ["estimator", "thresholds", "best_threshold", "threshold_scores", "scorers", "refit"]:
            if param_name in params:
                setattr(self, param_name, params[param_name])
                del params[param_name]

        self.estimator.set_params(**params)
        return self

    def get_params(self, deep=True):
        params={"estimator" :self.estimator,
                "best_threshold": self.best_threshold,
                "threshold_scores": self.threshold_scores,
                "thresholds": self.thresholds,
                "scorers": self.scorers,
                "refit": self.refit}
        params.update(self.estimator.get_params(deep))
        return params


def specificity_score(y_true, y_pred):
    """Calculate specificity score metric for a binary classification

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels
    y_pred: 1d array-like
        Predicted labels, as returned by the classifier

    Returns
    -------
    specificity : float
        Returns the specificity score, or true negative rate, i.e. the
        proportion of the negative label (label=0) samples that are correctly
        classified as the negative label
    """

    cm = confusion_matrix(y_true, y_pred)
    tn = float(cm[0][0])
    fp = float(cm[0][1])

    return tn/(tn+fp)


def spatial_loocv(estimator, X, y, coordinates, size, radius,
                  random_state=None):
    """Spatially buffered leave-One-out cross-validation
    Uses a circular spatial buffer to separate samples that are used to train
    the estimator, from samples that are used to test the prediction accuracy

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    coordinates : 2d-array like
        Spatial coordinates, usually as xy, that correspond to each sample in
        X.
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

    return (y_test, y_pred, y_prob)
