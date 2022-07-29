import multiprocessing
import os
import re
from collections import Counter

import numpy as np
import tempfile


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


def _check_alignment(layers):
    """Check that a list of raster datasets are aligned with the same
    pixel dimensions and geotransforms.

    Parameters
    ----------
    layers : list
        List of pyspatialml.RasterLayer objects.

    Returns
    -------
    dict or False
        Dict of metadata if all layers are spatially aligned, otherwise
        returns False.
    """

    src_meta = []
    for layer in layers:
        src_meta.append(layer.ds.meta.copy())

    if not all(i["crs"] == src_meta[0]["crs"] for i in src_meta):
        Warning("crs of all rasters does not match, possible unintended consequences")

    if not all(
        [
            i["height"] == src_meta[0]["height"]
            or i["width"] == src_meta[0]["width"]
            or i["transform"] == src_meta[0]["transform"]
            for i in src_meta
        ]
    ):
        return False

    else:
        return src_meta[0]


def _make_name(name):
    """Converts a file basename to a valid class attribute name.

    Parameters
    ----------
    name : str
        File basename for converting to a valid class attribute name.

    Returns
    -------
    valid_name : str
        Syntactically correct name of layer so that it can form a class
        instance attribute.
    """
    basename = os.path.basename(name)
    sans_ext = os.path.splitext(basename)[0]

    valid_name = sans_ext.replace(" ", "_").replace("-", "_").replace(".", "_")

    if valid_name[0].isdigit():
        valid_name = "x" + valid_name

    valid_name = re.sub(r"[\[\]\(\)\{\}\;]", "", valid_name)
    valid_name = re.sub(r"_+", "_", valid_name)

    return valid_name


def _fix_names(combined_names):
    """Adjusts the names of pyspatialml.RasterLayer objects within the
    Raster when appending new layers.

    This avoids the Raster object containing duplicated names in the
    case that multiple RasterLayers are appended with the same name.

    In the case of duplicated names, the RasterLayer names are appended
    with a `_n` with n = 1, 2, 3 .. n.

    Parameters
    ----------
    combined_names : list
        List of str representing names of RasterLayers. Any duplicates
        will have a suffix appended to them.

    Returns
    -------
    list
        List with adjusted names
    """

    counts = Counter(combined_names)

    for s, num in counts.items():
        if num > 1:
            for suffix in range(1, num + 1):
                if s + "_" + str(suffix) not in combined_names:
                    combined_names[combined_names.index(s)] = s + "_" + str(suffix)
                else:
                    i = 1
                    while s + "_" + str(i) in combined_names:
                        i += 1
                    combined_names[combined_names.index(s)] = s + "_" + str(i)

    return combined_names


class TempRasterLayer:
    """Create a NamedTemporaryFile like object on Windows that has a
    close method

    Workaround used on Windows which cannot open the file a second time
    """

    def __init__(self, tempdir=tempfile.tempdir):
        self.tfile = tempfile.NamedTemporaryFile(dir=tempdir, suffix=".tif").name
        self.name = self.tfile

    def close(self):
        os.unlink(self.tfile)
