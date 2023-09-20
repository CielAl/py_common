from abc import abstractmethod
from typing import runtime_checkable, Protocol, Dict
from torch.utils.data import Dataset, get_worker_info, DataLoader
from py_common.loggers.global_logger import GlobalLoggers
import logging
logger = GlobalLoggers.instance().get_logger(__name__, logging.DEBUG)


@runtime_checkable
class Cached(Protocol):
    _cache: Dict
    CACHE_NAME = '_cache'

    def new_cache(self) -> Dict:
        """
        Override to create the cache object in subclasses if necessary.

        Returns:
            Cache objects (Dict)
        """
        pass


class CacheManager(dict):
    # not working
    @staticmethod
    def validated_key_ref(key):
        assert isinstance(key, Cached)
        ref_idx = id(key)
        return ref_idx

    def __contains__(self, other):
        other_id = id(other)
        return super().__contains__(other_id)

    def __setitem__(self, key: Cached, value):
        ref_idx = CacheManager.validated_key_ref(key)
        super().__setitem__(ref_idx, value)

    def get(self, key: Cached, default=None):
        ref_idx = CacheManager.validated_key_ref(key)
        return super().get(ref_idx, default)

    def __getitem__(self, key: Cached):
        ref_idx = CacheManager.validated_key_ref(key)
        return super().__getitem__(ref_idx)

    def clear_cache(self, key: Cached):
        ref_idx = CacheManager.validated_key_ref(key)
        super().pop(ref_idx, None)

    def __init__(self):
        super().__init__()


CACHE_MANAGER = CacheManager()


class CachedDataset(Dataset, Cached):

    def __del__(self):
        global CACHE_MANAGER
        CACHE_MANAGER.clear_cache(self)

    @abstractmethod
    def new_cache(self) -> Dict:
        """
        Override to create the cache object in subclasses if necessary.

        Returns:
            Cache objects (Dict)
        """
        return NotImplemented

    def __init_cache_helper(self):
        global CACHE_MANAGER
        # new_value = CACHE_MANAGER.get(self, None)
        # cache = new_value if new_value is not None else self.new_cache()
        # assert isinstance(cache, Dict)
        if self not in CACHE_MANAGER:
            cache = self.new_cache()
            CACHE_MANAGER[self] = cache
        else:
            logger.debug(f"Cache Hit")

        assert CACHE_MANAGER[self] is not None and isinstance(CACHE_MANAGER[self], Dict)
        return CACHE_MANAGER[self]

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
        return hasattr(self, Cached.CACHE_NAME) and isinstance(self._cache, Dict)\
            and key in self._cache

    # noinspection PyUnusedLocal
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
        # print(dataset)
        dataset.init_cache()

        # dataset._cache = cache

    @classmethod
    def new_dataloader(cls, dataset: 'CachedDataset', num_workers: int = 0, **kwargs) -> DataLoader:
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
