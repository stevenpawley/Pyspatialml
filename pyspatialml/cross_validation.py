import numpy as np
import inspect
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.preprocessing import binarize


class CrossValidateThreshold():
    """Perform cross-validation and calculate scores using a cutoff threshold that
       attains a minimum true positive rate"""

    def __init__(self, estimator, scoring, cv=3, positive=1, n_jobs=1):
        """Initiate a class instance

        Parameters
        ----------
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
        """Run fit method with all sets of parameters

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning

        groups : array-like, shape = [n_samples], optional
            Training vector groups for cross-validation

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator"""
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
        self.test_idx_ = [indexes[1] for indexes in cv.split(X, y, groups)]

    def score(self, tpr_threshold=None, cutoff_threshold=None):
        """Calculates the scoring metrics using a cutoff threshold that attains a true positive rate
        that is equal or greater than the desired tpr_threshold

        Parameters
        ----------
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


class ThresholdClassifierCV(BaseEstimator, ClassifierMixin):
    """Metaclassifier to perform cutoff threshold optimization

    This implementation is restricted to binary classification problems

    Notes
    -----

    - The training data are partitioned in k-1, and k sets
    - The metaclassifier trains the BaseEstimator on the k-1 partitions
    - The Kth paritions are used to determine the optimal cutoff taking
      the mean of the thresholds that maximize the scoring metric
    - The optimal cutoff is applied to all classifier predictions"""

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
        accepts string, callable, list/tuple, or dict"""

        # get dict of arguments from function call
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def find_threshold(self, X, y=None):
        """Finds the optimal cutoff threshold based on maximizing/minimizing the
        scoring method"""

        estimator = self.estimator
        thresholds = self.thresholds
        scorers = self.scorers_
        refit = self.refit

        y_pred = estimator.predict_proba(X)
        scores = dict([(dct, np.zeros((len(thresholds)))) for dct in scorers])

        for i, cutoff in enumerate(thresholds):
            for name, scorer in scorers.items():
                scores[name][i] = self.scorer_cutoff(y, y_pred, scorer, cutoff)

        top_score = scores[refit][scores[refit].argmax()]
        top_threshold = thresholds[scores[refit].argmax()]
        return top_threshold, top_score, scores

    def score(self, X, y):
        """Overloading of classifier score method
        score method is required for compatibility with GridSearch
        The scoring metric should be one that can be maximized (bigger=better)"""

        _threshold, score, _threshold_scores = self.find_threshold(X, y)

        return score

    @staticmethod
    def scorer_cutoff(y, y_pred, scorer, cutoff):
        """Helper method to binarize the probability scores based on a cutoff"""

        y_pred = (y_pred[:, 1] > cutoff).astype(int)
        return scorer(y, y_pred)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit method with all sets of parameters

        Parameters
        ----------
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
        - The response vector, i.e. y, should be initiated with None"""

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
                best_threshold, _score, threshold_scores = self.find_threshold(X_cal, y_cal)

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

    def predict(self, X):
        y_score = self.estimator.predict_proba(X)
        return np.array(y_score[:,1] >= self.best_threshold_)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def specificity_score(y_true, y_pred):
    """Calculate specificity score metric for a binary classification

    Parameters
    ----------

    y_true : 1d array-like
        Ground truth (correct) labels

    y_pred : 1d array-like
        Predicted labels, as returned by the classifier

    Returns
    -------

    specificity : float
        Returns the specificity score, or true negative rate, i.e. the
        proportion of the negative label (label=0) samples that are correctly
        classified as the negative label"""

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
        The object to use to fit the data

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning

    coordinates : 2d-array like
        Spatial coordinates, usually as xy, that correspond to each sample in
        X

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
    https://davidrroberts.wordpress.com/2016/03/11/spatial-leave-one-out-sloo-cross-validation/"""

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
