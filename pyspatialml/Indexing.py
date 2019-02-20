#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:25:03 2019

@author: steven
"""

from collections.abc import Mapping

class ExtendedDict(Mapping):
    """
    Dict that can return based on multiple keys
    """
    def __init__(self, *args, **kw):
        self._dict = dict(*args, **kw)
    
    def __getitem__(self, keys):
        if isinstance(keys, str):
            return self._dict[keys]
        return [self._dict[i] for i in keys]
    
    def __str__(self):
        return str(self._dict)
    
    def __setitem__(self, key, value):
        self._dict[key] = value
        
    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self):
        return len(self._dict)
    
    def pop(self, key):
        return self._dict.pop(key)


class LinkedList:
    """
    Provides integer-based indexing of a ExtendedDict
    """
    def __init__(self, d):
        self._index = d
    
    def __setitem__(self, index, value):
        
        if isinstance(index, int):
            key = list(self._index.keys())[index]
            self._index[key] = value
        
        if isinstance(index, slice):
            index = list(range(index.start, index.stop))
        
        if isinstance(index, (list, tuple)):
            for i, idx in enumerate(index):
                key = list(self._index.keys())[idx]
                self._index[key] = value[i]
    
    def __getitem__(self, index):
        key = list(self._index.keys())[index]
        return self._index[key]


class Raster:
    def __init__(self, *args, **kw):
        self.loc = ExtendedDict(*args, **kw)
        self.iloc = LinkedList(self.loc)
        
mydict = Raster(one=1)
mydict.loc['two'] = 2
mydict.loc['one']
mydict.loc[('one', 'two')]
mydict.iloc[0]
mydict.iloc[0] = 100
mydict.loc['one']
mydict.loc['two']
mydict.iloc[0:2] = [500, 500]
mydict.loc[('one', 'two')]
mydict.loc.pop('one')