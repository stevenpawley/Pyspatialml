import math
import numpy as np
import numpy.ma as ma
import rasterio
import sys
import os
from copy import deepcopy
from sklearn import metrics
from numpy.random import RandomState
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


def specificity_score(y_true, y_pred):
    """
    Calculate specificity score

    Args
    ----
    y_true: 1D numpy array of truth values
    y_pred: 1D numpy array of predicted classes

    Returns
    -------
    specificity: specificity score
    
    """
    
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    tn = float(cm[0][0])
    fp = float(cm[0][1])

    return tn/(tn+fp)


def spatial_loocv(estimator, X, y, coordinates, size, radius, sampling=None,
                  random_state=None):
    """
    Spatially Buffered Leave-One-out
    adapted from R code
    https://davidrroberts.wordpress.com/2016/03/11/spatial-leave-one-out-sloo-cross-validation/

    Args
    ----
    size: sample size to process (number of LOO runs)
    radius: radius for the spatial buffer around test point
    
    """

    from numpy.random import RandomState

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

        # balancing
        if sampling is not None:
            X_train, y_train = sampling.fit_sample(X_train, y_train)

        # Build the model
        estimator.fit(X_train, y_train)

        # Predict on test point
        y_pred = np.append(y_pred, estimator.predict(X_test[i].reshape(1, -1)))
        y_prob = np.vstack((
            y_prob, estimator.predict_proba(X_test[i].reshape(1, -1))))

    return (y_test, y_pred, y_prob)


def varimp_permutation(estimator, X, y, n_permutations, scorer,
                       n_jobs, random_state):
    """

    Method to perform permutation-based feature importance during
    cross-validation (cross-validation is applied externally to this
    method)

    Procedure is:
    1. Pass fitted estimator and test partition X y
    2. Assess AUC on the test partition (bestauc)
    3. Permute each variable and assess the difference between bestauc and
       the messed-up variable
    4. Repeat (3) for many random permutations
    5. Average the repeats

    Args
    ----
    estimator (object): estimator that has been fitted to a training partition
    X, y: 2d and 1d numpy arrays of data and labels from a test partition
    n_permutations (integer): number of random permutations to apply
    scorer (object): scikit-learn metric function to use
    n_jobs (integer): integer, number of processing cores
    random_state (float): seed to pass to the numpy random.seed

    Returns
    -------
    scores (2d numpy array): scores for each predictor following permutation

    """

    # calculate score on original variables without permutation
    # determine best metric type for binary/multiclass/regression scenarios
    y_pred = estimator.predict(X)
    best_score = scorer(y, y_pred)

    # repeated permutations and return difference from best score per predictor
    scores = Parallel(n_jobs=n_jobs)(
        delayed(__permute)(
            estimator, X, y, best_score, scorer, random_state)
        for n in range(n_permutations))

    # average the repetitions
    scores = np.asarray(scores)
    scores = scores.mean(axis=0)

    return scores


def __permute(estimator, X, y, best_score, scorer, random_state):
    """

    Permute each predictor and measure difference from best score

    Args
    ----
    estimator (object): scikit learn estimator
    X, y: 2d and 1d numpy arrays data and labels from a test partition
    best_score (float): best scorer obtained on unperturbed data
    scorer (object): scoring method to use to measure importances
    random_state (float): random seed

    Returns
    -------
    scores (2D numpy array): scores for each predictor following permutation

    """

    rstate = RandomState(random_state)

    # permute each predictor variable and assess difference in score
    scores = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        Xscram = np.copy(X)
        Xscram[:, i] = rstate.choice(X[:, i], X.shape[0])

        # fit the model on the training data and predict the test data
        y_pred = estimator.predict(Xscram)
        scores[i] = best_score-scorer(y, y_pred)
        if scores[i] < 0:
            scores[i] = 0

    return scores


def __parallel_fit(estimator, X, y, groups, train_indices, test_indices, sample_weight):
    """

    Fit classifiers/regressors in parallel

    Args
    ----
    estimator (object): scikit learn estimator
    X, y: 2D and 1D numpy arrays of training data and labels
    groups (1D numpy array): of len(y) containing group labels
    train_indices, test_indices: 1D numpy arrays of indices to use for
        training/validation
    sample_weight (1D numpy array): of len(y) containing weights to use during
        fitting

    """
    
    clf = deepcopy(estimator)

    # create training and test folds
    X_train, y_train = X[train_indices], y[train_indices]

    if groups is not None:
        groups_train = groups[train_indices]
    else:
        groups_train = None

    # subset training and test fold sample_weight
    if sample_weight is not None:
        weights = sample_weight[train_indices]

    # specify fit_params for sample_weights if required
    if isinstance(clf, Pipeline) and sample_weight is not None:
        fit_params = {'classifier__sample_weight': weights}
    elif not isinstance(clf, Pipeline) and sample_weight is not None:
        fit_params = {'sample_weight': weights}
    else:
        fit_params = {}

    # fit estimator with/without groups
    if groups is not None and type(clf).__name__ in ['RandomizedSearchCV', 'GridSearchCV']:
        clf.fit(X_train, y_train, groups=groups_train, **fit_params)
    else:
        clf.fit(X_train, y_train, **fit_params)

    return clf


def cross_val_scores(estimator, X, y, groups=None, sample_weight=None, cv=3,
                     scoring='accuracy', feature_importances=False,
                     n_permutations=25, random_state=None, n_jobs=-1):
    """

    Stratified Kfold and GroupFold cross-validation using multiple
    scoring metrics and permutation feature importances

    Args
    ----
    estimator (object): Scikit learn estimator
    X, y: 2D and 1D numpy array of training data and labels
    groups (1D numpy array): group labels
    sample_weight (1D numpy array[n_samples,]): sample weights per sample
    cv (integer or object): Number of cross-validation folds or
        sklearn.model_selection object
    scoring (list): List of performance metrics to use
    feature_importances (boolean): option to perform permutation-based importances
    n_permutations (integer): Number of permutations during feature importance
    random_state (float): Seed to pass to the random number generator

    Returns
    -------
    scores (dict): Containing lists of scores per cross-validation fold
    byclass_scores (dict): Containing scores per class
    fimp (2D numpy array): permutation feature importances per feature
    clf_resamples (list): List of fitted estimators
    predictions (2d numpy array): with y_true, y_pred, fold

    """

    # -------------------------------------------------------------------------
    # create copies of estimator and create cross-validation iterator
    # -------------------------------------------------------------------------

    # deepcopy estimator
    clf = deepcopy(estimator)

    # create model_selection method
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv)

    # -------------------------------------------------------------------------
    # create dictionary of lists to store metrics
    # -------------------------------------------------------------------------
    try:  # py2
        if isinstance(scoring, basestring):
            scoring = [scoring]
    except:  # py3
        if isinstance(scoring, str):
            scoring = [scoring]

    scores = dict.fromkeys(scoring)
    scores = {key: [] for key, value in scores.iteritems()}
    scoring_methods = {'accuracy': metrics.accuracy_score,
                       'balanced_accuracy': metrics.recall_score,
                       'average_precision': metrics.average_precision_score,
                       'brier_loss': metrics.brier_score_loss,
                       'kappa': metrics.cohen_kappa_score,
                       'f1': metrics.f1_score,
                       'fbeta': metrics.fbeta_score,
                       'hamming_loss': metrics.hamming_loss,
                       'jaccard_similarity': metrics.jaccard_similarity_score,
                       'log_loss': metrics.log_loss,
                       'matthews_corrcoef': metrics.matthews_corrcoef,
                       'precision': metrics.precision_score,
                       'recall': metrics.recall_score,
                       'specificity': specificity_score,
                       'roc_auc': metrics.roc_auc_score,
                       'zero_one_loss': metrics.zero_one_loss,
                       'r2': metrics.r2_score,
                       'neg_mean_squared_error': metrics.mean_squared_error}

    byclass_methods = {'f1': metrics.f1_score,
                       'fbeta': metrics.fbeta_score,
                       'precision': metrics.precision_score,
                       'recall': metrics.recall_score}

    # create dict to store byclass metrics results
    n_classes = len(np.unique(y))
    labels = np.unique(y)
    byclass_scores = dict.fromkeys(byclass_methods)
    byclass_scores = {key: np.zeros((0, n_classes)) for key, value in byclass_scores.iteritems()}

    # remove any byclass_scorers that are not in the scoring list
    byclass_scores = {key: value for key, value in byclass_scores.iteritems() if key in scores}

    # check if remaining scorers are valid sklearn metrics
    for i in scores.keys():
        try:
            list(scoring_methods.keys()).index(i)
        except:
            gs.fatal(('Scoring ', i, ' is not a valid scoring method',
                            os.linesep(),
                            'Valid methods are: ', scoring_methods.keys()))

    # set averaging type for global binary or multiclass scores
    if len(np.unique(y)) == 2 and all([0, 1] == np.unique(y)):
        average = 'binary'
    else:
        average = 'macro'

    # create np array to store feature importance scores
    if feature_importances is True:
        fimp = np.zeros((cv.get_n_splits(), X.shape[1]))
        fimp[:] = np.nan
    else:
        fimp = None

    # -------------------------------------------------------------------------
    # extract cross-validation indices
    # -------------------------------------------------------------------------

    if groups is None:
        k_fold = cv.split(X, y)
    else:
        k_fold = cv.split(X, y, groups=groups)

    trains, tests = [], []
    for train_indices, test_indices in k_fold:
        trains.append(train_indices)
        tests.append(test_indices)

    # -------------------------------------------------------------------------
    # Perform multiprocessing fitting of clf on each fold
    # -------------------------------------------------------------------------
    clf_resamples = Parallel(n_jobs=n_jobs)(
        delayed(__parallel_fit)(clf, X, y, groups, train_indices,
                              test_indices, sample_weight)
        for train_indices, test_indices in zip(trains, tests))

    # -------------------------------------------------------------------------
    # loop through each fold and calculate performance metrics
    # -------------------------------------------------------------------------

    # store predictions and indices
    predictions = np.zeros((len(y), 3)) # y_true, y_pred, fold

    fold = 0
    for train_indices, test_indices in zip(trains, tests):

        # create training and test folds
        X_test, y_test = X[test_indices], y[test_indices]

        # prediction of test fold
        y_pred = clf_resamples[fold].predict(X_test)
        predictions[test_indices, 0] = y_test
        predictions[test_indices, 1] = y_pred
        predictions[test_indices, 2] = fold

        # calculate global performance metrics
        for m in scores.keys():
            # metrics that require probabilties
            if m == 'brier_loss' or m == 'roc_auc':
                y_prob = clf_resamples[fold].predict_proba(X_test)[:, 1]
                scores[m] = np.append(
                    scores[m], scoring_methods[m](y_test, y_prob))

            # metrics that have no averaging for multiclass
            elif m == 'kappa' or m == 'specificity' or m == 'accuracy' \
            or m == 'hamming_loss' or m == 'jaccard_similarity' \
            or m == 'log_loss' or m == 'zero_one_loss' \
            or m == 'matthews_corrcoef' or m == 'r2' \
            or m == 'neg_mean_squared_error':
                scores[m] = np.append(
                    scores[m], scoring_methods[m](y_test, y_pred))

            # balanced accuracy
            elif m == 'balanced_accuracy':
                scores[m] = np.append(
                    scores[m], scoring_methods[m](
                        y_test, y_pred, average='macro'))

            # metrics that have averaging for multiclass
            else:
                scores[m] = np.append(
                    scores[m], scoring_methods[m](
                        y_test, y_pred, average=average))

        # calculate per-class performance metrics
        for key in byclass_scores.keys():
            byclass_scores[key] = np.vstack((
                byclass_scores[key], byclass_methods[key](
                    y_test, y_pred, labels=labels, average=None)))

        # feature importances using permutation
        if feature_importances is True:
            fimp[fold, :] = varimp_permutation(
                clf_resamples[fold], X_test, y_test, n_permutations,
                scoring_methods[scoring[0]], n_jobs, random_state)
        fold += 1

    return(scores, byclass_scores, fimp, clf_resamples, predictions)


def predict(estimator, predictor_rasters, output, predict_type='raw', index=None, rowchunk=25):

    """
    Prediction on list of GDAL rasters using a fitted scikit learn model

    Args
    ----
    estimator: Scikit learn estimator object
    predictor_rasters: List of paths to GDAL rasters
    output: Path to a GeoTiff raster for the classification results
    predict_type: 'Raw' for classification/regression, 'prob' for probabilities,
                  'all' for both classification and probabilities
    index: Optional, list of class indices to export
    rowchunk: Integer of raster rows to process at one time

    """

    if isinstance(index, int):
        index = [index]

    # Get the processing region dimensions
    n_features = len(predictor_rasters)
    feature = rasterio.open(predictor_rasters[0])
    rows, cols, crs, transform = (feature.height,
                                  feature.width,
                                  feature.crs,
                                  feature.transform)
                                  
    # create lists of row increments
    windows = []
    for row in range(0, rows, rowchunk):
        if row+rowchunk > rows:
            rowincr = rows - row
        else:
            rowincr = rowchunk
        windows.append((row, row+rowincr, rowincr))


    # Loop through rasters block-by-block
    for block in windows:

        start = block[0]
        end = block[1]
        blocksize = block[2]

        # print progress to screen
        # sys.stdout.write('\r')
        sys.stdout.write('Percentage complete: %0.1f %%' % (start/rows*100))
        sys.stdout.flush()
        
        # create np 3D numpy array
        img_np = ma.empty((blocksize, cols, n_features))

        # loop through each rowblock, and each band and add these values
        # to the 2D array img_np_row
        for f in range(n_features):
            img_np[:, :, f] = (
                rasterio.open(predictor_rasters[f]).read(
                    1, masked=True, window=((start, end), (0, cols))))

        # reshape each image block matrix into a 1D matrix
        # the scikit learn predict function expects a list of pixels
        # not an NxM matrix
        n_samples = blocksize * cols
        flat_pixels = img_np.reshape((n_samples, n_features))
        flat_pixels = ma.masked_invalid(flat_pixels)

        # create mask for NaN values and replace with number
        flat_pixels_mask = np.isnan(flat_pixels)  # True means invalid
        flat_pixels = ma.filled(flat_pixels, fill_value=0)

        # perform the prediction for classification
        if predict_type == 'raw' or predict_type == 'all':
            result_cla = estimator.predict(flat_pixels)
    
            # replace mask
            result_cla = np.ma.masked_array(
                    result_cla, mask=flat_pixels_mask.mask.any(axis=1))
    
            result_cla = ma.filled(result_cla, fill_value=-99999)
    
            # reshape the prediction from a 1D matrix/list
            # back into the original format
            result_cla = result_cla.reshape((blocksize, cols))
    
            # write rowchunks to rasterio raster
            if start == 0:
                clf_output = rasterio.open(
                    output, mode='w', driver='GTiff', height=rows, width=cols,
                    count=1, dtype='float32', crs=crs, transform=transform, nodata=-99999)
    
            clf_output.write(
                result_cla.reshape((blocksize, cols)).astype('float32'),
                window=((start, end), (0, cols)), indexes=1)
                
        # perform the prediction for probabilities
        if predict_type == 'prob' or predict_type == 'all':
            result_proba = estimator.predict_proba(flat_pixels)
            if start == 0:
                if index is None:
                    index = range(result_proba.shape[1])
                    n_bands = len(index)
                else:
                    n_bands = len(np.unique(index))
    
                fname, ext = os.path.basename(output).split(os.path.extsep)
                output_proba = os.path.join(os.path.dirname(output), fname + '_proba' + os.path.extsep + ext)
                proba_output = rasterio.open(
                    output_proba, mode='w', driver='GTiff', height=rows, width=cols,
                    count=n_bands, dtype='float32', crs=crs, transform=transform, nodata=-99999)
    
            for i_class, label in enumerate(index):
                result_proba_i = result_proba[:, label]
    
                # replace mask
                result_proba_i = np.ma.masked_array(
                    result_proba_i, mask=flat_pixels_mask.mask.any(axis=1),
                    fill_value=np.nan)
    
                result_proba_i = ma.filled(
                    result_proba_i, fill_value=-99999)
    
                # reshape the prediction from a 1D matrix/list back into the original format
                result_proba_i = result_proba_i.reshape((blocksize, cols))
    
                # write rowchunks to rasterio raster
                proba_output.write(
                    result_proba_i.astype('float32'), window=((start, end), (0, cols)),
                    indexes=1+i_class)

    # close classification image
    if predict_type == 'raw' or predict_type == 'all':
        clf_output.close()
    if predict_type == 'prob' or predict_type == 'all':
        proba_output.close()
