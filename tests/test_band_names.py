import os
from unittest import TestCase

import rasterio
import tempfile
import pyspatialml.datasets.nc as nc
from pyspatialml import Raster


class TestNames(TestCase):
    """Test the initiation of a Raster object when the file raster dataset
    contains band names
    """
    def setUp(self) -> None:
        """Create a temperory file with a raster dataset with band names
        """
        with rasterio.open(nc.multiband) as src:
            self.descriptions = ["band_1", "band_2", "band_3", "band_4", "band_5"]

            self.fp = tempfile.NamedTemporaryFile(suffix=".tif").name

            with rasterio.open(self.fp, "w", **src.meta) as dst:
                for i, band in enumerate(self.descriptions, start=1):
                    dst.write_band(i, src.read(i))
                    dst.set_band_description(i, band)

    def tearDown(self) -> None:
        os.remove(self.fp)

    def test_names_from_file(self) -> None:
        """Test the initiation of a Raster object from a file when the file raster 
        dataset contains band descriptions"""
        r = Raster(self.fp)
        self.assertEqual(list(r.names), self.descriptions)

    def test_names_from_rasterio(self) -> None:
        """Test the initiation of a Raster object from a rasterio.DatasetReader
        object when the file raster dataset has band descriptions
        """
        with rasterio.open(self.fp) as src:
            r = Raster(src)
            self.assertEqual(list(r.names), self.descriptions)

    def test_names_subsetting(self) -> None:
        """Test that the names of the bands are preserved when subsetting a raster
        """
        r = Raster(self.fp)
        subset = r.iloc[[0, 1]]
        self.assertEqual(list(subset.names), self.descriptions[0:2])

        new = r.copy(["band_1", "band_2"])
        new["band_3"] = r["band_3"]
        self.assertEqual(list(new.names), self.descriptions[0:3])
