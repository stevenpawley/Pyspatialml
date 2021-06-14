from unittest import TestCase

import matplotlib as mpl
import numpy as np

import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestPlotting(TestCase):
    def setUp(self) -> None:
        self.predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]
        self.stack = Raster(self.predictors)
        self.stack_single = Raster(self.predictors[0])

    def tearDown(self) -> None:
        self.stack.close()
        self.stack_single.close()

    def test_plotting_raster(self):

        # test basic raster matrix plot
        p = self.stack.plot()
        self.assertIsInstance(p, np.ndarray)

        # test with arguments
        p = self.stack.plot(
            cmap="plasma",
            norm=mpl.colors.Normalize(vmin=10, vmax=100),
            title_fontsize=10,
            label_fontsize=10,
            names=["band1", "band2", "band3", "band4", "band5", "band7"],
            figsize=(10, 5),
            legend_kwds={"orientation": "horizontal"}
        )
        self.assertIsInstance(p, np.ndarray)

    def test_plotting_single(self):
        p = self.stack_single.plot(
            legend_kwds={"orientation": "horizontal", "fraction": 0.04})
        self.assertIsInstance(p, mpl.axes.Subplot)
