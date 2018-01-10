#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def discrete_cmap(N, base_cmap=None):
    # https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)


def raster_plot(x, extent, shapes=None, names=None, stretches=None, smin=None,
                smax=None, cmaps=None, cmap_type=None, aspect='auto',
                width=12, height=12, wspace=0.4, hspace=0.4,
                title_fontsize=12, label_fontsize=10, out=None):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from skimage import exposure
    import math

    # determine number of rasters; band,rows,cols
    nmaps = x.shape[0]

    # error checking and default plotting settings
    if names is not None:
        assert len(names) == nmaps, 'map names not equal to rasters'
    else:
        names = ['Raster ' + str(i) for i in range(nmaps)]

    if stretches is not None:
        assert len(stretches) == nmaps, 'stretches not equal to n_rasters'
    else:
        stretches = [False] * nmaps

    if smin is not None:
        assert len(smin) == nmaps, 'smin not equal to n_rasters'
    else:
        smin = [0] * nmaps

    if smax is not None:
        assert len(smax) == nmaps, 'smax not equal to n_rasters'
    else:
        smax = [100] * nmaps

    if cmaps is not None:
        assert len(cmaps) == nmaps, 'cmaps not equal to n_rasters'
    else:
        cmaps = ['jet'] * nmaps

    # continuous or discrete color maps
    if cmap_type is not None:
        for i, clr in enumerate(cmap_type):
            if clr != 'c':
                cmaps[i] = discrete_cmap(N=int(clr), base_cmap=cmaps[i])

    # estimate required number of rows and columns in figure
    rows = int(np.sqrt(nmaps))
    cols = int(math.ceil(np.sqrt(nmaps)))
    if rows*cols < nmaps:
        rows += 1

    fig, axs = plt.subplots(rows, cols, figsize=(width, height))

    # axs.flat is an iterator over the row-order flattened axs array
    for ax, grid_n, name, stretch, ssmin, ssmax, ramp in zip(
            axs.flat, range(nmaps), names, stretches, smin, smax, cmaps):

        rio_np = x[grid_n, :, :]

        # perform histogram stretching
        if stretch is True:
            v_min, v_max = np.percentile(rio_np[~rio_np.mask], (smin, smax))

            out_min, out_max = \
                rio_np[~rio_np.mask].min(), rio_np[~rio_np.mask].max()

            rio_np = exposure.rescale_intensity(
                rio_np, in_range=(v_min, v_max), out_range=(out_min, out_max))

        ax.set_title(name, fontsize=title_fontsize, y=1.00)
        im = ax.imshow(
            rio_np, cmap=ramp,
            extent=[extent.left, extent.right, extent.bottom, extent.top])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=label_fontsize)

        # hide tick labels by default when multiple rows or cols
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])

        # show y-axis tick labels on first subplot
        if grid_n == 0 and rows > 1:
            ax.set_yticklabels(
                    ax.yaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)
        if grid_n == 0 and rows == 1:
            ax.set_xticklabels(
                    ax.xaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)
            ax.set_yticklabels(
                    ax.yaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)
        if rows > 1 and grid_n == (rows*cols)-cols:
            ax.set_xticklabels(
                    ax.xaxis.get_majorticklocs().astype('int'),
                    fontsize=label_fontsize)

        if shapes is not None:
            shapes.plot(ax=im.axes, color='gray')

    # To hide the last plot that isn't showing, do this:
    # axs.flat[-1].set_visible(False)
    # or more generally to hide empty plots
    for ax in axs.flat[axs.size - 1:nmaps - 1:-1]:
        ax.set_visible(False)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    if out is not None:
        plt.savefig(out, dpi=300)
    plt.show()


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1

    # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.

    http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
