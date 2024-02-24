"""
TODO...
"""
from abc import ABC, abstractmethod
from typing import Sequence
import os


class DataParser(ABC):
    __slice_patch_idx_list: Sequence[str]

    def _init_slice_patch_idx_list(self, slice_list):
        self.__slice_patch_idx_list = [os.path.splitext(os.path.basename(x))[0] for x in slice_list]

    @property
    def slice_patch_idx_list(self):
        return self.__slice_patch_idx_list

    @abstractmethod
    def data_point(self, *args, **kwargs):
        ...

    @abstractmethod
    def slice_from_idx(self, *args, **kwargs):
        ...

    @abstractmethod
    def feat_from_idx(self, *args, **kwargs):
        ...

