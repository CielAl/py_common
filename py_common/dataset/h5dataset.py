"""
Wrapper import from existing HDF5 code --> need to revise for adaption of BaseDataset wrapper
"""
# temp wrapper --> may need to migrate later
# note --> need to regenerate the h5files
# from h5data.dataset import H5Dataset as LegacyH5Dataset
# from torch.utils.data import DataLoader
from py_common.h5py_helper.reader import H5Reader
from py_common.h5py_helper.parser import H5ParserCore
from .base import AbstractDataset
from .data_class import ModelInput
from .image_buffer import BufferedDataset
from .cached import CachedDataset
from typing import List, Dict, Sequence, Optional
import os


class CachedH5Dataset(AbstractDataset, CachedDataset, BufferedDataset):
    """H5Dataset implementation with caching the file resource handle and tiles.

    Note that for shuffled dataloader, larger chunk size will only throttle the IO more.
    HDF5 is far faster than reading from WSI using bbox directly only in sequential reading.
    Random access is still throttled.

    This is because, random access (shuffle) of dataloader leads to partially read chunks: after a chunk is read,
    the whole chunk of consecutive data section will be loaded in the HDF5 cache (managed by the file handle). However,
    since the next index to read is random, and it may not be in the cached chunk. This repeats until the cache is full
    and previously-cached chunks are discarded (probably never used).

    Therefore, for random access, user should carefully benchmark the chunk size and cache size of HDF5 handle
    (rdcc_nbytes) for better IO efficiency, as well as the relationship of batch size.

    Another thing to do is that, you may create your dataloader sampler (see torch.util.data.DataLoader) to
    partially shuffle the data (i.e., shuffle the dataset into different segments while each segment is used
    as a batch) to reduce the partially read chunk or chunk discarded from cache without being reused.


    """

    _h5reader: H5Reader
    _uri: str
    worker_cache: Dict
    _rdcc_nbytes: Optional[int]

    @property
    def h5reader(self):
        return self._h5reader

    def __len__(self):
        return len(self.h5reader)

    def new_cache(self):
        """Cache - use the HDF5's internal cache

        Returns:

        """
        # {self._uri: self.h5reader.new_h5()}
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        return {self._uri: self.h5reader.new_h5(self._rdcc_nbytes)}

    def init_cache(self):
        """
        Invoke to initialize the _cache field in worker_init_fn or in factory methods depending on whether there
        are multiple workers.

        Returns:

        """

        self._cache = self.new_cache()

    def __init__(self, uri: str, reader: H5Reader, rdcc_nbytes: Optional[int] = None, buffer_ratio: float = 1):
        """Init the instance.

        Args:
            uri: HDF5 file location.
            reader: The H5Reader object to parse HDF5 dataset
            rdcc_nbytes: rdcc_nbytes to define the cache size. See `h5py`. If None then use `H5Reader.CACHE_SIZE`
            buffer_ratio: ratio (fractional number in [0, 1]) of tiles to be buffered.
        """
        # super().__init__()
        self._uri = uri
        self._rdcc_nbytes = rdcc_nbytes
        self._h5reader = reader
        pool_size = int(buffer_ratio * len(self._h5reader))
        self.new_buffer(pool_size)
        # todo - somehow the eviction mechanism of cache prevent loading multiple chunked datasets
        # todo refactor later but for now we force to cache the label and bbox
        # data['label'].item()
        # with self.h5reader.get_h5root() as root:
        #     self._label_cache = root[H5ParserCore.CONST_LABEL][:]
        #     self._bbox_cache = root[H5ParserCore.CONST_BBOX][:]

    @classmethod
    def build(cls, uri: str,
              array_fields: Sequence = (H5ParserCore.CONST_TILE, H5ParserCore.CONST_BBOX, H5ParserCore.CONST_LABEL,
                                        H5ParserCore.CONST_URI),
              buffer_ratio: float = 1, **kwargs):
        """
        Factory method. Build from file name

        Args:
            uri: uri of h5 file
            array_fields: which fields to read from hdf5 data
            buffer_ratio: ratio (from 0.0 to 1.0) of tiles to be cached in memory.

        Returns:

        """
        array_fields_read: List[str] = [H5ParserCore.CONST_TILE, H5ParserCore.CONST_BBOX, H5ParserCore.CONST_LABEL,
                                        H5ParserCore.CONST_URI]
        # , H5ParserCore.CONST_BBOX, H5ParserCore.CONST_LABEL
        primary_field: str = H5ParserCore.CONST_TILE
        reader = H5Reader(uri=uri, array_fields_read=array_fields_read, primary_field=primary_field)
        return cls(uri=uri, reader=reader, buffer_ratio=buffer_ratio)

    def get_item_from_reader(self, index: int):
        """
        Read data using the associated reader.

        Args:
            index:

        Returns:

        """
        # no effect here since we do not enforce caching of file handles.
        # but leave it here in case retaining of file handle actually helps and my observation is wrong.
        if self.is_cached(self._uri):
            return self.h5reader.get_item_helper(self.worker_cache[self._uri], index)
        #
        with self.h5reader.get_h5root() as root:
            return self.h5reader.get_item_helper(root, index)

    def fetch(self, index: int) -> ModelInput:
        """Helper function.

        Reading tiles. Read from memory directly if cached in the pool. Otherwise read it using the associated h5reader
        and cache it into the pool if not exceeding the pool ratio.

        Args:
            index: dataset index. i-th item to read

        Returns:
            NetInput
        """
        data = self._query_buffer_by_key(index)
        if data is None:
            data: Dict = self.get_item_from_reader(index)
        img = data[H5ParserCore.CONST_TILE]
        label = data[H5ParserCore.CONST_LABEL].item()
        # label = self._label_cache[index].item()
        bbox = data[H5ParserCore.CONST_BBOX]
        # bbox = self._bbox_cache[index]
        self._add_into_buffer_by_key(index, data)

        # todo store actual slide names later
        # filename = self.h5reader.get_h5root()
        filename = data[H5ParserCore.CONST_URI]
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        # todo to update
        net_input = ModelInput(data=img, ground_truth=label, original=0, filename=filename, meta=bbox)
        return net_input

    def clear_cache(self):
        """close the handle to release the cache memory.

        Returns:

        """
        if hasattr(self, 'worker_cache'):
            for k, v in self.worker_cache.items():
                v.close()
