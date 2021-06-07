import numpy as np
from abc import ABC


class RasterStats(ABC):
    def _stats(self, max_pixels):
        rel_width = self.shape[1] / max_pixels

        if rel_width > 1:
            col_scaling = round(max_pixels / rel_width)
            row_scaling = max_pixels - col_scaling
        else:
            col_scaling = round(max_pixels * rel_width)
            row_scaling = max_pixels - col_scaling

        out_shape = (row_scaling, col_scaling)
        arr = self.read(masked=True, out_shape=out_shape)
        return arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))

    def min(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return np.nanmin(arr, axis=1).data

    def max(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return np.nanmax(arr, axis=1).data

    def mean(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return np.nanmean(arr, axis=1).data

    def median(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return np.nanmedian(arr, axis=1).data

    def stddev(self, max_pixels=10000):
        arr = self._stats(max_pixels)
        return np.nanstd(arr, axis=1).data
