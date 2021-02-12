import multiprocessing
import os
import re
from collections import Counter

import numpy as np
import rasterio
from rasterio.windows import Window


class BaseRaster:
    """Base class for Raster objects"""

    def __init__(self):
        self.shape = None
        self.crs = None
        self.transform = None
        self.width = None
        self.height = None
        self.bounds =None
        self.meta = None
        self.count = 0
        self.iloc = None
        self.loc = None
        self._block_shape = (256, 256)

    def _make_name(self, name):
        """Converts a file basename to a valid class attribute name.

        Parameters
        ----------
        name : str
            File basename for convert to a valid class attribute name.

        Returns
        -------
        valid_name : str
            Syntactically correct name of layer so that it can form a class
            instance attribute.
        """
        valid_name = (
            os.path.basename(name).
            split(os.path.extsep)[0].
            replace(" ", "_").
            replace("-", "_")
        )

        if valid_name[0].isdigit():
            valid_name = "x" + valid_name

        valid_name = re.sub(r"[\[\]\(\)\{\}\;]", "", valid_name)
        valid_name = re.sub(r"_+", "_", valid_name)

        if self.names is not None:
            if valid_name in self.names:
                valid_name = "_".join([valid_name, "1"])

        return valid_name

    def _stats(self, max_pixels):
        n_pixels = self.shape[0] * self.shape[1]
        scaling = max_pixels / n_pixels
        out_shape = (round(self.shape[0] * scaling),
                     round(self.shape[1] * scaling))
        arr = self.read(masked=True, out_shape=out_shape)
        return arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))

    def min(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return arr.min(axis=1).data

    def max(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return arr.max(axis=1).data

    def mean(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return arr.mean(axis=1).data

    def median(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return np.median(arr, axis=1).data

    def head(self):
        window = Window(col_off=0, row_off=0, width=20, height=10)
        return self.read(window=window)

    def tail(self):
        window = Window(col_off=self.width - 20, row_off=self.height - 10,
                        width=20, height=10)
        return self.read(window=window)

    @property
    def names(self):
        """Return the names of the RasterLayers in the Raster object

        Returns
        -------
        list
            List of names of RasterLayer objects
        """
        return list(self.loc.keys())

    def close(self):
        """Close all of the RasterLayer objects in the Raster.

        Note that this will cause any rasters based on temporary files to be
        removed. This is intended as a method of clearing temporary files that
        may have accumulated during an analysis session.
        """
        for layer in self.iloc:
            layer.close()

    def block_shapes(self, rows, cols):
        """Generator for windows for optimal reading and writing based on the
        raster format Windows are returns as a tuple with xoff, yoff, width,
        height.

        Parameters
        ----------
        rows : int
            Height of window in rows.

        cols : int
            Width of window in columns.
        """

        for i, col in enumerate(range(0, self.width, cols)):
            if col + cols < self.width:
                num_cols = cols
            else:
                num_cols = self.width - col

            for j, row in enumerate(range(0, self.height, rows)):
                if row + rows < self.height:
                    num_rows = rows
                else:
                    num_rows = self.height - row

                yield Window(col, row, num_cols, num_rows)

    @property
    def block_shape(self):
        """Return the windows size used for raster calculations, specified as
        a tuple (rows, columns).

        Returns
        -------
        tuple
            Block window shape that is currently set for the Raster as a tuple
            in the format of (n_rows, n_columns) in pixels.
        """
        return self._block_shape

    @block_shape.setter
    def block_shape(self, value):
        """Set the windows size used for raster calculations, specified as a
        tuple (rows, columns).

        Parameters
        ----------
        value : tuple
            Tuple of integers for default block shape to read and write data
            from the Raster object for memory-safe calculations. Specified as
            (n_rows,n_columns).
        """
        if not isinstance(value, tuple):
            raise ValueError(
                "block_shape must be set using an integer tuple \
                as (rows, cols)")
        rows, cols = value

        if not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError(
                "tuple must consist of integer values referring \
                to number of rows, cols")
        self._block_shape = (rows, cols)

    def _check_supported_dtype(self, dtype):
        if dtype is None:
            dtype = self.meta["dtype"]
        else:
            if rasterio.dtypes.check_dtype(dtype) is False:
                raise AttributeError(
                    "{dtype} is not a support GDAL dtype".format(dtype=dtype)
                )
        return dtype


def _get_nodata(dtype):
    # Get a nodata value based on the minimum value permissible by dtype
    try:
        nodata = np.iinfo(dtype).min
    except ValueError:
        nodata = np.finfo(dtype).min

    return nodata


def _get_num_workers(n_jobs):
    # Determine cpu count using scikit-learn convention of -1, -2 ...
    n_cpus = multiprocessing.cpu_count()

    if n_jobs < 0:
        n_jobs = n_cpus + n_jobs + 1

    return n_jobs


def _check_alignment(layers):
    """Check that a list of raster datasets are aligned with the same pixel
    dimensions and geotransforms.

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
        Warning("crs of all rasters does not match, " "possible unintended \
                consequences")

    if not all(
            [i["height"] == src_meta[0]["height"] or
             i["width"] == src_meta[0]["width"] or
             i["transform"] == src_meta[0]["transform"] for i in src_meta]):
        return False

    else:
        return src_meta[0]


def _fix_names(combined_names):
    """Adjusts the names of pyspatialml.RasterLayer objects within the Raster
    when appending new layers.

    This avoids the Raster object containing duplicated names in the case that
    multiple RasterLayer's are appended with the same name.

    In the case of duplicated names, the RasterLayer names are appended
    with a `_n` with n = 1, 2, 3 .. n.

    Parameters
    ----------
    combined_names : list
        List of str representing names of RasterLayers. Any duplicates with
        have a suffix appended to them.

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
                    combined_names[combined_names.index(s)] = (
                            s + "_" + str(suffix))
                else:
                    i = 1
                    while s + "_" + str(i) in combined_names:
                        i += 1
                    combined_names[combined_names.index(s)] = s + "_" + str(i)

    return combined_names
