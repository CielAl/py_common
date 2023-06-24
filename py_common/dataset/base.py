from torch.utils.data import Dataset, get_worker_info, DataLoader
from abc import ABC, abstractmethod
from .data_class import ModelInput
from typing import Dict, Set, Any


SET_DATA_KEYS: Set = set(ModelInput.__annotations__.keys())


class AbstractDataset(Dataset, ABC):

    _cache: Dict
    CACHE_NAME = '_cache'

    @abstractmethod
    def new_cache(self):
        return NotImplemented

    def init_cache(self):
        self._cache = self.new_cache()

    def is_cached(self, key: str):
        return hasattr(self, AbstractDataset.CACHE_NAME) and isinstance(self._cache, Dict)\
            and key in self._cache

    @staticmethod
    def worker_init_func(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        dataset.init_cache()

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

    @classmethod
    def new_dataloader(cls, dataset, num_workers: int = 0, **kwargs) -> DataLoader:
        assert isinstance(num_workers, int) and num_workers >= 0
        init_func = cls.worker_init_func if num_workers > 0 else None
        if num_workers == 0:
            dataset.init_cache()
        return DataLoader(dataset, worker_init_fn=init_func, num_workers=num_workers, **kwargs)

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        raise NotImplementedError
