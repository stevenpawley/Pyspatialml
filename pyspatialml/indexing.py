from collections.abc import Mapping
from collections import OrderedDict


class ExtendedDict(Mapping):
    """
    Dict that can return based on multiple keys

    Args
    ---
    parent : Raster object to store RasterLayer indexing
        Requires to parent Raster object in order to setattr when
        changes in the dict, reflecting changes in the RasterLayers occur
    """

    def __init__(self, parent, *args, **kw):
        self.parent = parent
        self._dict = OrderedDict(*args, **kw)

    def __getitem__(self, keys):
        if isinstance(keys, str):
            return self._dict[keys]
        return [self._dict[i] for i in keys]

    def __str__(self):
        return str(self._dict)

    def __setitem__(self, key, value):
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


class LinkedList(object):
    """
    Provides integer-based indexing of a ExtendedDict

    Args
    ---
    parent : Raster object to store RasterLayer indexing
        Requires to parent Raster object in order to setattr when
        changes in the dict, reflecting changes in the RasterLayers occur
    """

    def __init__(self, parent, extended_dict):
        self.parent = parent
        self._index = extended_dict

    def __setitem__(self, index, value):

        if isinstance(index, int):
            key = list(self._index.keys())[index]
            self._index[key] = value
            setattr(self.parent, key, value)

        if isinstance(index, slice):
            index = list(range(index.start, index.stop))

        if isinstance(index, (list, tuple)):
            for i, idx in enumerate(index):
                key = list(self._index.keys())[idx]
                self._index[key] = value[i]
                setattr(self.parent, key, value[i])

    def __getitem__(self, index):
        key = list(self._index.keys())[index]
        return self._index[key]
