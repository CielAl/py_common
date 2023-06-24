"""
Base abstract class - Dataset that returns ModelInput (TypedDict) in __getitem__
Optionally have cache for unpickleable objects supported for multiprocessing by setting it in worker_init_fn
of dataloader (directly initiate it in single processing mode)
The cache is supposed to be a dict that can be looked up from a key to return the data.

"""
from torch.utils.data import Dataset, get_worker_info, DataLoader
from abc import ABC, abstractmethod
from .data_class import ModelInput
from typing import Dict, Set


SET_DATA_KEYS: Set = set(ModelInput.__annotations__.keys())


class AbstractDataset(Dataset, ABC):

    _cache: Dict
    CACHE_NAME = '_cache'

    DEFAULT_VALUE: int = 0

    @abstractmethod
    def new_cache(self) -> Dict:
        """
        Override to create the cache object in subclasses if necessary.

        Returns:
            Cache objects (Dict)
        """
        return NotImplemented

    def init_cache(self):
        """
        Invoke to initialize the _cache field in worker_init_fn or in factory methods depending on whether there
        are multiple workers.

        Returns:

        """
        self._cache = self.new_cache()

    def is_cached(self, key: str):
        """
        Check if cache is available by examining if the _cache attribute is initialized, whether it's a dict type,
        and whether the value to query is preloaded
        Args:
            key: key to query the preloaded values.

        Returns:

        """
        return hasattr(self, AbstractDataset.CACHE_NAME) and isinstance(self._cache, Dict)\
            and key in self._cache

    @staticmethod
    def worker_init_func(worker_id):
        """
        Function to initializing workers in each iteration. Initialize the cache under multiprocessing.
        Otherwise cache containing Unpickleable objects may not work in multiprocessing.
        Args:
            worker_id:

        Returns:

        """
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        dataset.init_cache()

    @staticmethod
    def _validate_type(data: ModelInput) -> ModelInput:
        """
        Validate if the data is a ModelInput type by enforcing its keys. The data must contain all necessary keys
        defined in ModelInput. Its key set should be a superset of that of ModelInput.
        Args:
            data: data to examine.

        Returns:
            data (ModelInput)

        Raises:
            AssertionError
        """
        assert isinstance(data, Dict)
        key_set = set(data.keys())
        assert key_set.issuperset(SET_DATA_KEYS)
        return data

    @abstractmethod
    def __len__(self):
        return NotImplemented

    @abstractmethod
    def fetch(self, index) -> ModelInput:
        """
        Override to fetch data from index. It's invoked in __getitem__
        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def __getitem__(self, index) -> ModelInput:
        """
        Use self.fetch to obtain the data by index but always guarantee that the data is a ModelInput
        Args:
            index: index of data (e.g., corresponding index in a list)

        Returns:

        """
        data = self.fetch(index)
        return AbstractDataset._validate_type(data)

    @classmethod
    def new_dataloader(cls, dataset: 'AbstractDataset', num_workers: int = 0, **kwargs) -> DataLoader:
        """
        A generic factory method to build a dataloader from the dataset. Initializing the cache either directly or
        via worker_init_fn depending on how many workers are there. num_workers > 1 indicates multiprocessing and
        the cache will be initialized by worker_init_fn.
        Args:
            dataset: Dataset of the dataloader
            num_workers: number of workers for multiprocessing. If greater than 0 than the cache will be initialized by
                worker_init_fn at the beginning of each full traverse.
            **kwargs: Other keyword args for pytorch's DataLoader.

        Returns:

        """
        assert isinstance(num_workers, int) and num_workers >= 0
        init_func = cls.worker_init_func if num_workers > 0 else None
        if num_workers == 0:
            dataset.init_cache()
        return DataLoader(dataset, worker_init_fn=init_func, num_workers=num_workers, **kwargs)

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        """
        A factory method to build the dataset.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError
