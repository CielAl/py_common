from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from .data_class import ModelInput
from typing import Dict, Set


SET_DATA_KEYS: Set = set(ModelInput.__annotations__.keys())


class AbstractDataset(Dataset, ABC):

    @staticmethod
    def _validate_type(data: ModelInput) -> ModelInput:
        assert isinstance(data, Dict)
        key_set = set(data.keys())
        assert key_set.issuperset(SET_DATA_KEYS)
        return data

    @abstractmethod
    def __len__(self):
        return NotImplemented

    @abstractmethod
    def fetch(self, index) -> ModelInput:
        raise NotImplementedError

    def __getitem__(self, index) -> ModelInput:
        data = self.fetch(index)
        return AbstractDataset._validate_type(data)
