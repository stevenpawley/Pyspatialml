from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist
import numpy as np
from copy import deepcopy


class GeoDistTransformer(BaseEstimator, TransformerMixin):
    """Transformer to add new features based on geographical distances to 
    reference locations.

    Parameters
    ----------
    ref_xs : list
        A list of x-coordinates of reference locations.
    
    ref_ys : list
        A list of x-coordinates of reference locations.
    
    log : bool (opt), default=False
        Optionally log-transform the distance measures.
    
    Returns
    -------
    X_new : ndarray
        array of shape (n_samples, n_features) with new geodistance features 
        appended to the right-most columns of the array.

    """
    def __init__(
        self,
        ref_xs=None,
        ref_ys=None,
        log=False
    ):
        self.ref_xs = ref_xs
        self.ref_ys = ref_ys
        self.refs_ = None
        self.log = log

    def fit(self, X, y=None):

        self.ref_xs = np.asarray(self.ref_xs)
        self.ref_ys = np.asarray(self.ref_ys)

        if self.ref_xs.ndim == 1:
            self.ref_xs.reshape(-1, 1)
        
        if self.ref_ys.ndim == 1:
            self.ref_ys.reshape(-1, 1)

        self.refs_ = np.column_stack((self.ref_xs, self.ref_ys))
        
        return self

    def transform(self, X, y=None):
        dists = cdist(self.refs_, X).transpose()
        
        if self.log is True:
            dists = np.log(dists)
        
        return np.column_stack((X, dists))
