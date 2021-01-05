import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def rasterio_normalize(arr, axis=None):
    """Scales an array using min-max scaling.

    Parameters
    ----------
    arr : ndarray
        A numpy array containing the image data.

    axis : int (opt)
        The axis to perform the normalization along.

    Returns
    -------
    numpy.ndarray
        The normalized array
    """
    v_max = np.nanmax(arr, axis)
    v_min = np.nanmin(arr, axis)
    norm = (arr - v_min) / (v_max - v_min)
    return norm


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


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """Function to offset the "center" of a colormap. Useful for data with a negative
    min and positive max and you want the middle of the colormap's dynamic range to be
    at zero.

    Source:
    http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Parameters
    ----------
    cmap : str
        The matplotlib colormap to be altered

    start :  any number
        Offset from lowest point in the colormap's range. Defaults to 0.0 (no lower
        offset). Should be between 0.0 and `midpoint`.
    midpoint :  any number between 0.0 and 1.0
        The new center of the colormap. Defaults to 0.5 (no shift). In general, this
        should be  1 - vmax/(vmax + abs(vmin)). For example if your data range from
        -15.0 to +5.0 and you want the center of the colormap at 0.0, `midpoint` should
        be set to  1 - 5/(5 + 15)) or 0.75.
    stop :  any number between `midpoint` and 1.0
        Offset from highets point in the colormap's range. Defaults to 1.0 (no upper
        offset).

    Returns
    -------
    matplotlib.cmap
        The colormap with its centre shifted to the midpoint value.
    """

    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


class RasterPlot(object):
    def plot(
        self,
        cmap=None,
        norm=None,
        figsize=None,
        out_shape=(100, 100),
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
            A matplotlib.colors.Normalize to apply to all of the RasterLayers.
            This overides the norm attribute of each RasterLayer.

        figsize : tuple (opt), default=None
            Size of the resulting matplotlib.figure.Figure.

        out_shape : tuple, default=(100, 100)
            Number of rows, cols to read from the raster datasets for plotting.

        title_fontsize : any number, default=8
            Size in pts of titles.

        label_fontsize : any number, default=6
            Size in pts of axis ticklabels.

        legend_fontsize : any number, default=6
            Size in pts of legend ticklabels.

        names : list (opt), default=None
            Optionally supply a list of names for each RasterLayer to override the
            default layer names for the titles.

        fig_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.figure call when
            creating the figure object.

        legend_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.colorbar call when
            creating the colorbar object.

        subplots_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.subplots_adjust
            function. These are used to control the spacing and position of each
            subplot, and can include
            {left=None, bottom=None, right=None, top=None, wspace=None, hspace=None}.

        Returns
        -------
        axs : numpy.ndarray
            array of matplotlib.axes._subplots.AxesSubplot or a single
            matplotlib.axes._subplots.AxesSubplot if Raster object contains only a
            single layer.
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
                    "arguments 'names' needs to be the same length as the number of RasterLayer objects"
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
                    ax.set_yticklabels(
                        ax.yaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                    )
                if n == 0 and rows == 1:
                    ax.set_xticklabels(
                        ax.xaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                    )
                    ax.set_yticklabels(
                        ax.yaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                    )
                if rows > 1 and n == (rows * cols) - cols:
                    ax.set_xticklabels(
                        ax.xaxis.get_majorticklocs().astype("int"), fontsize=label_fontsize
                    )

            for ax in axs.flat[axs.size - 1 : self.count - 1 : -1]:
                ax.set_visible(False)

            plt.subplots_adjust(**subplots_kwds)
        
        else:
            arr = self.iloc[0].read(masked=True, out_shape=out_shape)
            cmap = cmaps[0]
            norm = norms[0]
            axs.set_title(names[0], fontsize=title_fontsize, y=1.00)            
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
