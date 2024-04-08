import numpy as np

from .data_class import ModelInput
from typing import List, Dict, Optional
from .base import AbstractDataset
from .cached import CachedDataset
from torch.utils.data import Subset


class TypedSubset(AbstractDataset, CachedDataset):
    """
    Wrapper dataset to add on-the-fly augmentation.
    """
    _dataset: AbstractDataset
    _indices: List | np.ndarray
    subset: AbstractDataset | Subset

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self.subset)

    def new_cache(self):
        return self._dataset.new_cache()

    def init_cache(self, cache: Optional[Dict] = None):
        self._cache = self.dataset.init_cache(cache)
        self.dataset._cache = self._cache

    def __init__(self, dataset: AbstractDataset, indices: List | np.ndarray):
        """

        Args:
            dataset: associated dataset.
            indices: indices of the subset
        """
        super().__init__()
        self._dataset = dataset
        self._indices = indices
        self.subset = Subset(self._dataset, self._indices)

    def fetch(self, index) -> ModelInput:
        data: ModelInput = self.subset[index]
        assert isinstance(data, Dict) and set(data.keys()).issubset(ModelInput.__annotations__.keys())
        return data

    @classmethod
    def build(cls, dataset: AbstractDataset, indices: List | np.ndarray):
        return cls(dataset=dataset, indices=indices)
