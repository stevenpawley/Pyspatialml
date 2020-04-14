import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
