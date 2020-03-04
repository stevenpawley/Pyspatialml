from collections.abc import Mapping
from collections import OrderedDict

class _LocIndexer(Mapping):
    """
    Access raster maps by using a label

    Represents a structure similar to a dict, but can return values using a
    list of keys (not just a single key)

    Parameters
    ----------
    parent : pyspatialml.Raster
        Requires to parent Raster object in order to setattr when
        changes in the dict, reflecting changes in the RasterLayers occur.
    """

    def __init__(self, parent, *args, **kw):
        self.parent = parent
        self._dict = OrderedDict(*args, **kw)

    def __getitem__(self, keys):
        """
        Index a _LocIndexer instance using keys (labels)

        Parameters
        ----------
        keys : str, or list
            A label or list of labels used for selection.
        
        Returns
        -------
        pyspatialml.RasterLayer, or list of RasterLayers.
        """

        if isinstance(keys, str):
            selected = self._dict[keys]
        else:
            selected = [self._dict[i] for i in keys]    
        return selected

    def __str__(self):
        """
        Returns a string of the OrderedDict
        """
        return str(self._dict)

    def __setitem__(self, key, value):
        """
        Assign a new value to an element in the _LocIndexer instance using a key

        Currently items can be set using a single key only

        Parameters
        ----------
        key : str
            Name of index.

        value : pyspatialml.RasterLayer
            New value to assign to the index.
        """
        self._dict[key] = value
        setattr(self.parent, key, value)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def pop(self, key):
        pop = self._dict.pop(key)
        delattr(self.parent, key)
        return pop
    
    def rename(self, old, new):
        """
        Renames a key while preserving the order

        Parameters
        ----------
        old : str
            Name of label to rename.
        
        new : str
            New name to be assigned to index.
        """
        self._dict = OrderedDict([(new, v) if k == old else (k, v) 
                                 for k, v in self._dict.items()])
        delattr(self.parent, old)
        setattr(self.parent, new, self._dict[new])
