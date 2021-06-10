import random

from scipy.cluster.hierarchy import cut_tree, linkage
from shapely.geometry import Point


def filter_points(gdf, min_dist=0, remove="first"):
    """Filter points in geodataframe using a minimum distance buffer.

    Args
    ----
    gdf : Geopandas GeoDataFrame
        Containing point geometries.

    min_dist : int or float, optional (default=0)
        Minimum distance by which to filter out closely spaced points.

    remove : str, optional (default='first')
        Optionally choose to remove 'first' occurrences or 'last'
        occurrences.

    Returns
    -------
    xy : 2d array-like
        Numpy array filtered coordinates
    """
    xy = gdf.geometry.bounds.iloc[:, 0:2]

    Z = linkage(xy, "complete")
    tree_thres = cut_tree(Z, height=min_dist)
    gdf["tree_thres"] = tree_thres

    if remove == "first":
        gdf = gdf.groupby(by="tree_thres").first()

    elif remove == "last":
        gdf = gdf.groupby(by="tree_thres").last()

    return gdf


def get_random_point_in_polygon(poly):
    """Generates random shapely Point geometry objects within a single
    shapely Polygon object.

    Args
    ----
    poly : Shapely Polygon object

    Returns
    -------
    p : Shapely Point object
    """

    (minx, miny, maxx, maxy) = poly.bounds

    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))

        if poly.contains(p):
            return p
