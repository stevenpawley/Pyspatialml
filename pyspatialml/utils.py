import numpy as np
import multiprocessing


def _get_nodata(dtype):
    """Get a nodata value based on the minimum value permissible by dtype.
    """
    try:
        nodata = np.iinfo(dtype).min
    except ValueError:
        nodata = np.finfo(dtype).min

    return nodata


def _get_num_workers(n_jobs):
    n_cpus = multiprocessing.cpu_count()

    if n_jobs < 0:
        n_jobs = n_cpus + n_jobs + 1

    return n_jobs
