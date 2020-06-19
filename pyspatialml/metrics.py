from sklearn.metrics import confusion_matrix

def specificity_score(y_true, y_pred):
    """
    Calculate specificity score metric for a binary classification

    Args
    ----
    y_true : 1d array-like
        Ground truth (correct) labels

    y_pred : 1d array-like
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