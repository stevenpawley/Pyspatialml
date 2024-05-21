import pandas as pd
from collections.abc import MutableMapping
from . import raster, rasterlayer


class _LocIndexer(MutableMapping):
    """Access pyspatialml.RasterLayer objects by using a key.

    Represents a structure similar to a dict but allows access using a
    list of keys (not just a single key).
    """

    def __init__(self, *args, **kw):
        self.__dict__.update(*args, **kw)

    def __getitem__(self, key):
        """Defines the subset method for the _LocIndexer. Allows the
        contained RasterLayer objects to be subset using a either
        single, or multiple labels corresponding to the names of each
        RasterLayer.

        Parameters
        ----------
        key : a single str, or a list of str

        Returns
        -------
        Returns a RasterLayer if only a single item is subset, or a
        Raster if multiple items are subset.

        """
        if isinstance(key, str):
            new = self.__dict__[key]
        else:
            selected = []
            for i in key:
                if i in self.names is False:
                    raise KeyError("key not present in Raster object")
                else:
                    selected.append(self.__dict__[i])
            new = raster.Raster(selected)
        return new

    def __setitem__(self, key, value):
        """Allows a RasterLayer object to be assigned to a name within
        a Raster object. This automatically updates the indexer with
        the layer, and adds the RasterLayer's name as an attribute in
        the Raster.

        Parameters
        ----------
        key : str
            The key to use for the assignment:

        value : pyspatialml.RasterLayer
            A single RasterLayer object to assign to the key.
        """
        if isinstance(value, rasterlayer.RasterLayer):
            self.__dict__[key] = value
        else:
            raise ValueError("value is not a RasterLayer object")

    def __iter__(self):
        """Iterates through keys"""
        return iter(self._keys)

    def __len__(self):
        """Number of layers in the indexer"""
        return len(self.__dict__) - len(self._internal)

    def __delitem__(self, key):
        """Delete a key:value pair"""
        self.__dict__.pop(key)

    def __repr__(self):
        print("Raster Object Containing {n} Layers".format(n=self.count))
        meta = pd.DataFrame(
            {
                "attribute": ["names", "files", "rows", "cols", "res", "nodatavals"],
                "values": [
                    list(self.names),
                    self.files,
                    self.shape[0],
                    self.shape[1],
                    self.res,
                    self.nodatavals,
                ],
            }
        )
        print(meta)

        return ""

    @property
    def _keys(self):
        d = {k: v for (k, v) in self.__dict__.items() if k not in self._internal}
        return d.keys()

    def _rename_inplace(self, old, new):
        """Rename a RasterLayer from `old` to `new. This method renames
        the layer in the indexer and renames the equivalent attribute
        in the parent Raster object.

        Parameters
        ----------
        old : str
            Name of the existing key.

        new : str
            Name to use to rename the existing key.
        """
        # rename the index by rebuilding the dict
        original_keys = list(self.__dict__.keys())
        new_keys = [new if i == old else i for i in original_keys]
        new_dict = dict(zip(new_keys, self.__dict__.values()))
        self.__dict__ = new_dict

        # update the internal name of a RasterLayer
        self.__dict__[new].name = new

    @property
    def loc(self):
        """Alias for the getter method of the indexer"""
        return self

    @loc.setter
    def loc(self, key, value):
        """Alias for the setter method if the indexer"""
        self.__dict__[key] = value

    @property
    def iloc(self):
        """Reference to an integer-based indexer to access the layers
        by integer position rather than label"""
        return _iLocIndexer(self)

    @property
    def names(self):
        return self._keys

    @names.setter
    def names(self, value):
        if isinstance(value, str):
            value = [value]

        if len(value) != self.count:
            raise ValueError(
                "Length of new names has to equal the number of layers in the Raster"
            )

        renamer = {old: new for (old, new) in zip(self.names, value)}
        self.rename(renamer, in_place=True)


class _iLocIndexer(object):
    """Access pyspatialml.RasterLayer objects using an index position

    A wrapper around _LocIndexer to enable integer-based indexing of
    the items in the OrderedDict. Setting and getting items can occur
    using a single index position, a list or tuple of positions, or a
    slice of positions.

    Methods
    -------
    __getitem__ : index
        Subset RasterLayers using an integer index, a slice of indexes,
        or a list/tuple of indexes. Returns a RasterLayer is a single
        item is subset, or a Raster if multiple layers are subset.

    __setitem__ : index, value
        Assign a RasterLayer to a index position within the indexer.
        The index can be a single integer position, a slice of
        positions, or a list/tuple of positions. This method also
        updates the parent Raster object's attributes with the names
        of the new RasterLayers that were passed as the value.
    """

    def __init__(self, loc_indexer):
        """Initiate a _iLocIndexer

        Parameters
        ----------
        loc_indexer : pyspatialml.raster._LocIndexer
            An instance of a _LocIndexer.
        """
        self._index = loc_indexer

    def __setitem__(self, index, value):
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            self._index[key] = value

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if start is None:
                start = 0
            if stop is None:
                stop = self.count
            if step is None:
                step = 1

            index = list(range(start, stop, step))

        if isinstance(index, (list, tuple)):
            for i, v in zip(index, value):
                key = list(self._index.keys())[i]
                self._index[key] = v

    def __getitem__(self, index):
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            selected = self._index[key]

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if start is None:
                start = 0

            if stop is None:
                stop = self.count

            if step is None:
                step = 1

            index = list(range(start, stop, step))

        if isinstance(index, (list, tuple)):
            key = []
            for i in index:
                key.append(list(self._index.keys())[i])
            selected = raster.Raster([self._index[k] for k in key])

        return selected
