import math

import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map.

    Source:
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    Parameters
    ----------
    N : int
        The number of colors in the colormap

    base_cmap : str
        The name of the matplotlib cmap to convert into a discrete map.

    Returns
    -------
    matplotlib.cmap
        The cmap converted to a discrete map.
    """

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)


class RasterPlotMixin:
    def plot(
        self,
        cmap=None,
        norm=None,
        figsize=None,
        out_shape=(500, 500),
        title_fontsize=8,
        label_fontsize=6,
        legend_fontsize=6,
        names=None,
        fig_kwds=None,
        legend_kwds=None,
        subplots_kwds=None,
    ):
        """Plot a Raster object as a raster matrix

        Parameters
        ----------
        cmap : str (opt), default=None
            Specify a single cmap to apply to all of the RasterLayers.
            This overides the cmap attribute of each RasterLayer.

        norm :  matplotlib.colors.Normalize (opt), default=None
            A matplotlib.colors.Normalize to apply to all of the
            RasterLayers. This overides the norm attribute of each
            RasterLayer.

        figsize : tuple (opt), default=None
            Size of the resulting matplotlib.figure.Figure.

        out_shape : tuple, default=(500, 500)
            Number of rows, cols to read from the raster datasets for
            plotting.

        title_fontsize : any number, default=8
            Size in pts of titles.

        label_fontsize : any number, default=6
            Size in pts of axis ticklabels.

        legend_fontsize : any number, default=6
            Size in pts of legend ticklabels.

        names : list (opt), default=None
            Optionally supply a list of names for each RasterLayer to
            override the default layer names for the titles.

        fig_kwds : dict (opt), default=None
            Additional arguments to pass to the
            matplotlib.pyplot.figure call when creating the figure
            object.

        legend_kwds : dict (opt), default=None
            Additional arguments to pass to the
            matplotlib.pyplot.colorbar call when creating the colorbar
            object.

        subplots_kwds : dict (opt), default=None
            Additional arguments to pass to the
            matplotlib.pyplot.subplots_adjust function. These are used to
            control the spacing and position of each subplot, and can
            include{left=None, bottom=None, right=None, top=None,
            wspace=None, hspace=None}.

        Returns
        -------
        axs : numpy.ndarray
            array of matplotlib.axes._subplots.AxesSubplot or a single
            matplotlib.axes._subplots.AxesSubplot if Raster object
            contains only a single layer.
        """

        # some checks
        if norm:
            if not isinstance(norm, mpl.colors.Normalize):
                raise AttributeError(
                    "norm argument should be a matplotlib.colors.Normalize object"
                )

        if cmap:
            cmaps = [cmap for i in self.iloc]
        else:
            cmaps = [i.cmap for i in self.iloc]

        if norm:
            norms = [norm for i in self.iloc]
        else:
            norms = [i.norm for i in self.iloc]

        if names is None:
            names = self.names
        else:
            if len(names) != self.count:
                raise AttributeError(
                    "arguments 'names' needs to be the same length as the number of "
                    "RasterLayer objects "
                )

        if fig_kwds is None:
            fig_kwds = {}

        if legend_kwds is None:
            legend_kwds = {}

        if subplots_kwds is None:
            subplots_kwds = {}

        if figsize:
            fig_kwds["figsize"] = figsize

        # estimate required number of rows and columns in figure
        rows = int(np.sqrt(self.count))
        cols = int(math.ceil(np.sqrt(self.count)))

        if rows * cols < self.count:
            rows += 1

        fig, axs = plt.subplots(rows, cols, **fig_kwds)

        # axs.flat is an iterator over the row-order flattened axs array
        if isinstance(axs, np.ndarray):
            for ax, n, cmap, norm, name in zip(
                axs.flat, range(self.count), cmaps, norms, names
            ):

                arr = self.iloc[n].read(masked=True, out_shape=out_shape)
                ax.set_title(name, fontsize=title_fontsize, y=1.00)

                im = ax.imshow(
                    arr,
                    extent=[
                        self.bounds.left,
                        self.bounds.right,
                        self.bounds.bottom,
                        self.bounds.top,
                    ],
                    cmap=cmap,
                    norm=norm,
                )

                divider = make_axes_locatable(ax)

                if "orientation" not in legend_kwds.keys():
                    legend_kwds["orientation"] = "vertical"

                if legend_kwds["orientation"] == "vertical":
                    legend_pos = "right"

                elif legend_kwds["orientation"] == "horizontal":
                    legend_pos = "bottom"

                cax = divider.append_axes(legend_pos, size="10%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax, **legend_kwds)
                cbar.ax.tick_params(labelsize=legend_fontsize)

                # hide tick labels by default when multiple rows or cols
                ax.axes.get_xaxis().set_ticklabels([])
                ax.axes.get_yaxis().set_ticklabels([])

                # show y-axis tick labels on first subplot
                if n == 0 and rows > 1:
                    ticks_loc = ax.get_yticks().tolist()
                    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_yticklabels(
                        ax.yaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize,
                    )

                if n == 0 and rows == 1:
                    ticks_loc = ax.get_xticks().tolist()
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_xticklabels(
                        ax.xaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize,
                    )

                    ticks_loc = ax.get_yticks().tolist()
                    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_yticklabels(
                        ax.yaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize,
                    )

                if rows > 1 and n == (rows * cols) - cols:
                    ticks_loc = ax.get_xticks().tolist()
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_xticklabels(
                        ax.xaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize,
                    )

            for ax in axs.flat[axs.size - 1 : self.count - 1 : -1]:
                ax.set_visible(False)

            plt.subplots_adjust(**subplots_kwds)

        else:
            arr = self.iloc[0].read(masked=True, out_shape=out_shape)
            cmap = cmaps[0]
            norm = norms[0]
            axs.set_title(list(names)[0], fontsize=title_fontsize, y=1.00)
            im = axs.imshow(
                arr,
                extent=[
                    self.bounds.left,
                    self.bounds.right,
                    self.bounds.bottom,
                    self.bounds.top,
                ],
                cmap=cmap,
                norm=norm,
            )

            divider = make_axes_locatable(axs)

            if "orientation" not in legend_kwds.keys():
                legend_kwds["orientation"] = "vertical"

            if legend_kwds["orientation"] == "vertical":
                legend_pos = "right"

            elif legend_kwds["orientation"] == "horizontal":
                legend_pos = "bottom"

            cax = divider.append_axes(legend_pos, size="10%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax, **legend_kwds)
            cbar.ax.tick_params(labelsize=legend_fontsize)

        return axs


class RasterLayerPlotMixin:
    def plot(
        self,
        cmap=None,
        norm=None,
        ax=None,
        cax=None,
        figsize=None,
        out_shape=(500, 500),
        categorical=None,
        legend=False,
        vmin=None,
        vmax=None,
        fig_kwds=None,
        legend_kwds=None,
    ):
        """Plot a RasterLayer using matplotlib.pyplot.imshow

        Parameters
        ----------
        cmap : str (default None)
            The name of a colormap recognized by matplotlib.
            Overrides the cmap attribute of the RasterLayer.

        norm : matplotlib.colors.Normalize (opt)
            A matplotlib.colors.Normalize to apply to the RasterLayer.
            This overrides the norm attribute of the RasterLayer.

        ax : matplotlib.pyplot.Artist (optional, default None)
            axes instance on which to draw to plot.

        cax : matplotlib.pyplot.Artist (optional, default None)
            axes on which to draw the legend.

        figsize : tuple of integers (optional, default None)
            Size of the matplotlib.figure.Figure. If the ax argument is
            given explicitly, figsize is ignored.

        out_shape : tuple, default=(500, 500)
            Number of rows, cols to read from the raster datasets for
            plotting.

        categorical : bool (optional, default False)
            if True then the raster values will be considered to
            represent discrete values, otherwise they are considered to
            represent continuous values. This overrides the
            RasterLayer 'categorical' attribute. Setting the argument
            categorical to True is ignored if the
            RasterLayer.categorical is already True.

        legend : bool (optional, default False)
            Whether to plot the legend.

        vmin, xmax : scale (optional, default None)
            vmin and vmax define the data range that the colormap
            covers. By default, the colormap covers the complete value
            range of the supplied data. vmin, vmax are ignored if the
            norm parameter is used.

        fig_kwds : dict (optional, default None)
            Additional arguments to pass to the
            matplotlib.pyplot.figure call when creating the figure
            object. Ignored if ax is passed to the plot function.

        legend_kwds : dict (optional, default None)
            Keyword arguments to pass to matplotlib.pyplot.colorbar().

        Returns
        -------
        ax : matplotlib axes instance
        """

        # some checks
        if fig_kwds is None:
            fig_kwds = {}

        if ax is None:
            if cax is not None:
                raise ValueError("'ax' can not be None if 'cax' is not.")
            fig, ax = plt.subplots(figsize=figsize, **fig_kwds)

        ax.set_aspect("equal")

        if norm:
            if not isinstance(norm, mpl.colors.Normalize):
                raise AttributeError(
                    "norm argument should be a " "matplotlib.colors.Normalize object"
                )

        if cmap is None:
            cmap = self.cmap

        if norm is None:
            norm = self.norm

        if legend_kwds is None:
            legend_kwds = {}

        arr = self.read(masked=True, out_shape=out_shape)

        if categorical is True:
            if self.categorical is False:
                N = np.bincount(arr)
                cmap = discrete_cmap(N, base_cmap=cmap)
            vmin, vmax = None, None

        im = ax.imshow(
            X=arr,
            extent=rasterio.plot.plotting_extent(self.ds),
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )

        if legend is True:
            plt.colorbar(im, cax=cax, ax=ax, **legend_kwds)

        return ax
