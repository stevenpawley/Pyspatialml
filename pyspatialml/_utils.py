import multiprocessing
import numpy as np


def get_nodata_value(dtype):
    """Get a nodata value based on the minimum value permissible by dtype

    Parameters
    ----------
    dtype : str or dtype
        dtype to return a nodata value for

    Returns
    -------
    nodata : any number
        A nodata value that is accomodated by the supplied dtype
    """
    try:
        nodata = np.iinfo(dtype).min
    except ValueError:
        nodata = np.finfo(dtype).min

    return nodata


def get_num_workers(n_jobs):
    """Determine cpu count using scikit-learn convention of -1, -2 ...

    Parameters
    ----------
    n_jobs : int
        Number of processing cores including -1 for all cores -1, etc.

    Returns
    -------
    n_jobs : int
        The actual number of processing cores.
    """
    n_cpus = multiprocessing.cpu_count()

    if n_jobs < 0:
        n_jobs = n_cpus + n_jobs + 1

    return n_jobs
