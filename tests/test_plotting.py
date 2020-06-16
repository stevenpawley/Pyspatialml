from unittest import TestCase
from pyspatialml import Raster
import pyspatialml.datasets.nc as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class TestPlotting(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_plotting_raster(self):
        stack = Raster(self.predictors)

        # test basic raster matrix plot
        p = stack.plot()
        self.assertIsInstance(p, np.ndarray)

        # test with arguments
        p = stack.plot(
            cmap="plasma",
            norm=mpl.colors.Normalize(vmin=10, vmax=100),
            title_fontsize=10,
            label_fontsize=10,
            names=["band1", "band2", "band3", "band4", "band5", "band7"],
            figsize=(10, 5),
            legend_kwds={"orientation": "horizontal"}
        )
        self.assertIsInstance(p, np.ndarray)

    def test_plotting_rasterlayer(self):
        stack = Raster(self.predictors)
        p = stack.iloc[0].plot(figsize=(10, 10), legend=True)
        self.assertIsInstance(p, mpl.axes.Subplot)

    def test_plotting_single(self):
        stack = Raster(self.predictors[0])
        p = stack.plot(legend_kwds={"orientation": "horizontal", "fraction": 0.04})
        self.assertIsInstance(p, mpl.axes.Subplot)
