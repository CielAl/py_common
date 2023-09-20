"""
Decouple the key function of memory pools for tiles here from the dataset implementation.

This is because caching of tiles should be optional and not coupled to our dataset baseclass.
For any sublcass of BaseDataset need pooling, extend them from PooledDataset as well.

"""
from typing import Protocol, runtime_checkable, Dict, Tuple, Union, Optional, Any
from torch.utils.data import Dataset
import numpy as np
import multiprocessing as mp


@runtime_checkable
class Buffer(Protocol):
    _tile_buffer: Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]
    _buffer_size: int
    manager: mp.Manager

    def clear_buffer(self):
        """Empty the buffer.

        Clear the internal dict.

        Returns:

        """
        if hasattr(self, '_tile_buffer'):
            self._tile_buffer.clear()

    def new_buffer(self, buffer_size: int):
        """

        New pool. Use the multiprocessing's share memory to cache the pools.

        Args:
            buffer_size: maximum number of item can be stored.

        Returns:

        """
        self.manager = mp.Manager()
        self._tile_buffer = self.manager.dict()
        self._buffer_size = buffer_size

    def current_size(self) -> int:
        return len(self._tile_buffer)

    def key_cached(self, key) -> bool:
        """Query whether the key is already in buffer.

        Args:
            key:

        Returns:

        """
        return key in self._tile_buffer


class BufferedDataset(Dataset, Buffer):

    def _query_buffer_by_key(self, key) -> Optional[Any]:
        """Interface of tile query by a unique key.

        Args:
            key: The unique key (must be hashable) to query the data.

        Returns:
            query results. None if not cached.
        """
        return self._tile_buffer.get(key, None)

    def _add_into_buffer_by_key(self, key, data):
        """Add tile to the buffer if the tile is not cached and if the size limit is not exceeded.

        Args:
            key: key to query the tile. Depends on implementation of subclasses
            data: data to query.

        Returns:

        """
        if self.current_size() < self._buffer_size and not self.key_cached(key):
            self._tile_buffer[key] = data