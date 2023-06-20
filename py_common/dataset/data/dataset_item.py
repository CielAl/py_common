from collections import OrderedDict
from typing import Mapping, Union, Iterable, Dict

import numpy as np


class DatasetItem(Mapping):
    """
    Must Inherit Mapping inorder to be accepted by Pytorch Collator Functions
    Order of dict is defined only in OrderedDict, or dict in python3.7 and the
    Cpython implementation of python3.6
    """

    @property
    def data_dict(self) -> OrderedDict:
        return self._data_dict

    @data_dict.setter
    def data_dict(self, new_value: OrderedDict):
        if not isinstance(new_value, OrderedDict):
            raise TypeError(f"Expect OrderedDict, got {type(new_value)}")
        self._data_dict = new_value

    def __init__(self, data_dict: Union[OrderedDict, Iterable] = None):
        if data_dict is None:
            data_dict = OrderedDict()
        if not isinstance(data_dict, OrderedDict):
            data_dict = OrderedDict(data_dict)
        super().__init__()
        self._data_dict = data_dict

    @classmethod
    def build(cls, obj):
        if isinstance(obj, DatasetItem):
            return obj
        elif not isinstance(obj, OrderedDict):
            obj = OrderedDict(obj)
        return cls(obj)

    def get(self, k):
        return self.data_dict.get(k)

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def set(self, key, value):
        self.__setitem__(key, value)
        return self

    def __len__(self):
        return len(self.data_dict)

    def keys(self):
        return self.data_dict.keys()

    def values(self):
        return self.data_dict.values()

    def items(self):
        return self.data_dict.items()

    def __contains__(self, item):
        return self.data_dict.__contains__(item)

    def __eq__(self, other):
        return self.data_dict.__eq__(other)

    def __ne__(self, other):
        return self.data_dict.__ne__(other)

    '''
    # ### It will break the assumptions in pytorch that the __iter__ of mapping uses its keys.
    def __iter__(self):
        """
        Unpacking Pattern using values instead of keys.
        For the compatibility of old codes that treat dataset items as tuples.
        Returns:
        """
        return iter(self.data_dict.items())
    '''

    def __iter__(self):
        return self.data_dict.__iter__()

    def __repr__(self):
        return f"{type(self).__name__} enclosing: {self.data_dict.__repr__()}"

    def _validate_type_order(self, type_order: np.ndarray):
        type_order = np.atleast_1d(type_order).ravel()
        assert len(self.keys()) == len(type_order), \
            f"length of type order mismatch. Expect {list(self.keys())}. Got {type_order}"

    def _re_order_helper(self, type_order):
        self._validate_type_order(type_order)
        for key in type_order:
            item = self.data_dict.pop(key)
            self.data_dict[key] = item

    def re_order(self, type_order, inplace_dict: bool = True):
        """

        Args:
            type_order (): If None, do no-op.
            inplace_dict (bool): if True, re-sort the OrderedDict. If False, create a new OrderedDict.
                Inplace op may be 40% faster.

        Returns:

        """
        if type_order is None:
            return self
        self._validate_type_order(type_order)
        if not inplace_dict:
            self._data_dict = OrderedDict([(key, self.data_dict[key]) for key in type_order])
        else:
            self._re_order_helper(type_order)
        return self


class DataItemUnpackByVal(object):
    def __init__(self, item: Union[DatasetItem, Dict]):
        if isinstance(item, dict):
            item = DatasetItem.build(item)
        assert isinstance(item, DatasetItem), f"Expect {DatasetItem.__name__}. Got {type(item)}"
        self._item = item

    def __getitem__(self, item):
        return self._item[item]

    def __setitem__(self, key, value):
        self._item[key] = value

    def __len__(self):
        return len(self._item)

    def keys(self):
        return self._item.keys()

    def values(self):
        return self._item.values()

    def items(self):
        return self._item.items()

    def __contains__(self, item):
        return self._item.__contains__(item)

    def __iter__(self):
        return iter(self._item.items())
