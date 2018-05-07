# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:36:06 2018

@author: Steven Pawley
"""

import rasterio
import numpy as np
from tempfile import NamedTemporaryFile as tmpfile


class indices(object):
    """Calculate remote sensing spectral indices using Raterio and Numpy
    """
    def __init__(self, src, blue=None, green=None, red=None, nir=None,
                 swir1=None, swir2=None, swir3=None):
        """Define new spectral indices class

        Parameters
        ----------
        src : rasterio._io.RasterReader, list of rasterio._io.RasterReader,
        path to GDAL file with stacked imagery bands, or list of paths to GDAL
        imagery bands

        blue : int
            Index of blue band.
        green : int
            Index of green band.
        red : int
            Index of red band.
        nir : int
            Index of near-infrared band (700-1100nm).
        swir1 : int
            Index of short-wave-infrared band (other, not currently used).
        swir2 : int
            Index of short-wave-infrared band (1400-1800nm).
        swir3 : int
            Index of short-wave-infrared band (2000-2500nm).
        """

        # read image data
        if isinstance(src, rasterio.io.DatasetReader):
            # bands contained within a single file opened as RasterReader
            src_arr = read_memmap(src)
            self.meta = src.meta.copy()

        elif isinstance(src, str):
            # path to stacked bands in single file
            src = rasterio.open(src)
            self.meta = src.meta.copy()
            src_arr = read_memmap(src)
            src.close()

        elif isinstance(src, list) and isinstance(src[1:], src[:-1]) \
                and isinstance(src[0], rasterio._io.RasterReader):
            # bands as separate files opened as RasterReader objects
            src_arr = np.memmap(
                filename=tmpfile(),
                shape=(len(src), src[0].shape[0], src[0].shape[1]),
                dtype='float32')

            self.meta = src[0].meta.copy()
            for i, band in enumerate(src):
                src_arr[i, :, :] = band.read(i)

        elif isinstance(src, list) and isinstance(src[1:], src[:-1]) \
                and isinstance(src[0], str):
            # paths to separate bands
            srcs = []
            for path in src:
                srcs.append(rasterio.open(path))

            src_arr = np.memmap(
                filename=tmpfile(),
                shape=(len(src), srcs[0].shape[0], srcs[0].shape[1]),
                dtype='float32')

            for i, src in enumerate(srcs):
                src_arr[i, :, :] = src.read(i)
            self.meta = srcs[0].meta.copy()
            [i.close() for i in srcs]

        # assign bands
        if blue:
            self.blue = src_arr[blue, :, :]
        if green:
            self.green = src_arr[green, :, :]
        if red:
            self.red = src_arr[red, :, :]
        if nir:
            self.nir = src_arr[nir, :, :]
        if swir1:
            self.swir1 = src_arr[swir1, :, :]
        if swir2:
            self.swir2 = src_arr[swir2, :, :]
        if swir3:
            self.swir3 = src_arr[swir3, :, :]

    def dvi(self):
        """Difference vegetation index (DVI)
        Richardson, A. J., and Weigand, C. (1977) Distinguishing vegetation
        from soil background information. Photogrammetric Engineering and
        Remote Sensing, p. 43.

        DVI = nir - red"""
        return (self.s * self.nir - self.red)

    def ctvi(self):
        """Corrected transformed vegetation index (CTVI)
        Perry, C. R., Jr. and Lautenschlager, L.F. (1984) Functional
        equivalence of spectral vegetation indices. Remote Sens of Environ
        v.14, 169-182.

        ctvi = ndvi+0.5 / sqrt(abs(ndvi+0.5))"""
        pl = (self.nir-self.red)/(self.nir+self.red) + 0.5
        return pl / np.sqrt(np.abs(pl))

    def ndvi(self):
        """Normalized difference vegetation index (NDVI)
        Kriegler, F.J., Malila, W.A., Nalepka, R.F., and Richardson, W. (1969)
        Preprocessing transformations and their effects on multispectral
        recognition. Proceedings of the Sixth International Symposium on
        Remote Sensing of Environment, p. 97-131.

        ndvi = (nir-red) / (nir+red)"""
        return (self.nir-self.red)/(self.nir+self.red)

    def evi(self, G, C1, C2, L):
        """Enhanced vegetation index (EVI)
        A. Huete, K. Didan, T. Miura, E. P. Rodriguez, X. Gao, L. G. Ferreira.
        (2002) Overview of the radiometric and biophysical performance of the
        MODIS vegetation indices. Remote Sensing of Environment v.83 95-213.

        Parameters
        ----------
        G : float
            gain factor
        C1, C2 : float
            coefficients of the aerosol resistance term, which uses the blue
            band to correct for aerosol influences in the red band. The
            coefficients adopted in the MODIS-EVI algorithm are; L=1, C1 = 6,
            C2 = 7.5, and G (gain factor) = 2.5
        L : float
            canopy background adjustment that addresses non-linear,
            differential NIR and red radiant transfer through a canopy

        Returns
        -------
        evi : 2D array-like"""
        return (self.G *
                ((self.nir - self.red) /
                 (self.nir + self.C1 * self.red -
                  self.C2 * self.blue + self.L)))

    def pvi(self, b, theta):
        """Perpendicular Vegetation Index (PVI)
        Richardson, A. J., and Weigand, C. (1977) Distinguishing vegetation
        from soil background information. Photogrammetric Engineering and
        Remote Sensing, p. 43.

        pvi = np.sqrt(nir - b) * np.cos(a) - red * np.sin(a)

        Parameters
        ----------
        b : float
            Intercept of soil line against NIR reflectivity
        theta : float
            Angle of soil line between the horizontal axis of red reflectivity,
            and the vertical axis of NIR reflectivity

        Returns
        -------
        pvi : 2D array-like
            2D numpy array of pvi

        Notes
        -----
        The spectral signature of multiple soil samples (same soil type) tends
        to show a linear relationship between red and nir bands, i.e. the soil
        line made from soil_line = a b*x.
        PVI calculates the orthogonal (perpendicular) distance of the
        vegetation signature from the soil line, i.e. it attempts to remove the
        influence of soil from the vegetation index. A limitation is that this
        method assumes the same soil type to underlie the study region."""
        return np.sqrt(self.nir - b) * np.cos(theta) - self.red * np.sin(theta)

    def evi2(self, G):
        """Two-band Enhanced vegetation index (EVI2)
        Jiang, Z., Huete, A.R., Didan, K., and Miura, T. (2008)
        Development of a two-band enhanced vegetation index without a blue band
        Remote Sensing of Environment v.112, 3833–3845.

        evi2 = G * (nir-red) / (nir + (2.4 * red)

        Notes
        -----
        The Enhanced vegetation index (EVI) minimizes soil and atmospheric
        influences, but requires a blue band. EVI2 represents a modification to
        EVI for sensors that do not contain a blue band, but attempts to
        produce results that are similar to EVI.

        Parameters
        ----------
        G : float
            Gain factor, usually 2.5 adapted from MODIS

        Returns
        -------
        evi2 : 2D array-like"""
        return self.G * ((self.nir - self.red) / (self.nir + 2.4 * self.red))

    def gemi(self):
        """Global environmental monitoring index (GEMI)
        Pinty, B., and Verstraete, M.M. (1992) GEMI: a non-linear index to
        monitor global vegetation from satellites. Vegetatio v.101 15-20.

        Notes
        -----
        TODO"""

        return ((((np.power(self.nir, 2) - np.power(self.red, 2)) * 2.0 +
                 (self.nir * 1.5) + (self.red * 0.5)) /
                (self.nir + self.red + 0.5)) *
                (1 - ((((np.power(self.nir, 2) - np.power(self.red, 2)) * 2 +
                        (self.nir * 1.5) + (self.red * 0.5)) /
                      (self.nir + self.red + 0.5)) * 0.25)) -
                ((self.red - 0.125) / (1 - self.red)))

    def gndvi(self):
        """Green Normalized diff vegetation index (GNDVI)
        Gitelson, A., and Merzlyak, M. (1998) Remote Sensing of Chlorophyll
        Concentration in Higher Plant Leaves. Advances in Space Research 22,
        689-692.

        Notes
        -----
        TODO"""
        return (self.nir - self.green)/(self.nir + self.green)

    def mndwi(self):
        """Modified Normalised Difference Water Index (MNDWI)
        Hu, H. (2006) Modification of normalised difference water index (NDWI)
        to enhance open water features in remotely sensed imagery.
        International Journal of Remote Sensing 27(14), 3025-3033 .

        mndwi = (green-swir2) / (green+swir2)

        Notes
        -----
        Maximizes the reflectivity of water by using the green band instead of
        NIR as in the original index."""
        return (self.green-self.swir2) / (self.green+self.swir2)

    def msavi(self):
        """Modified soil adjusted vegetation index (MSAVI)"""
        return self.nir + 0.5 - (0.5 * np.sqrt(
            np.power(2.0 * self.nir + 1.0, 2) - 8.0 *
            (self.nir - (2.0 * self.red))))

    def msavi2(self):
        """Modified soil adjusted vegetation index 2"""
        return (2.0 * self.nir + 1.0 -
                np.sqrt(np.power(2.0 * self.nir + 1.0, 2) -
                        8.0 * (self.nir - self.red))) / 2.0

    def ndvic(self, swir2ccc, swir2cdiff):
        """Normalized difference vegetation index"""
        return ((self.nir - self.red) / (self.nir + self.red) *
                (1 - (self.swir2 - self.swir2ccc)/self.swir2cdiff))

    def nbri(self):
        """normalized burn ratio index
        """
        return (self.swir2-self.nir)/(self.swir2+self.nir)

    def ndwi(self):
        """normalized difference water index
        McFeeters 1996
        """
        return (self.green - self.nir)/(self.green + self.nir)

    def ndwi2(self):
        """Gao 1996, Chen 2005
        """
        return (self.nir - self.swir2)/(self.nir + self.swir2)

    def rvi(self):
        """Ratio Vegetation Index
        Jordan, C.F. (1969) Derivation of leaf-area index from quality of light
        on the forest floor. Ecology, v. 50(4), 663–666.

        Based on the principle that leaves absorb more red than infrared light

        rvi = red / nir
        """
        return self.red / self.nir

    def satvi(self, L):
        """Soil adjusted total vegetation index
        """
        return (((self.swir2 - self.red) / (self.swir2 + self.red + self.L)) *
                (1.0 + self.L) - (self.swir3 / 2.0))

    def savi(self, L):
        """Soil adjusted vegetation index
        Huete1988
        """
        return ((self.nir - self.red) * (1.0 + self.L) /
                (self.nir + self.red + self.L))

    def slavi(self):
        return self.nir / (self.red + self.swir2)

    def sri(self):
        """Simple ratio index
        """
        return self.nir / self.red

    def tvi(self):
        """Transformed Vegetation Index
        Deering 1975
        """
        return np.sqrt((self.nir-self.red)/(self.nir+self.red) + 0.5)

    def ttvi(self):
        """Thiams Transformed Vegetation Index
        Thiam 1997"""
        return np.sqrt(np.abs((self.nir-self.red)/(self.nir+self.red) + 0.5))

    def wdvi(self):
        return self.nir - self.s * self.red

