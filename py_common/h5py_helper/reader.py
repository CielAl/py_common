import h5py
from contextlib import contextmanager
from .parser import H5ParserCore
from typing import List, Dict, Optional


class H5Reader:
    """Core functionality to read from HDF5

    """
    __uri: str
    __primary_field: str
    CACHE_SIZE: int = 1024 * 1024 * 1
    ERROR_FIELD_NOT_CREATED: str = "The dataset is not created:"

    @property
    def primary_field(self) -> str:
        """The primary data field which determines the length of the HDF5 dataset.

        Returns:
            name of the primary field
        """
        return self.__primary_field

    @primary_field.setter
    def primary_field(self, x: str):
        """Setter.

        Args:
            x: new field name

        Returns:

        """
        self.__primary_field = x

    @property
    def uri(self, ) -> str:
        """uri of HDF5 file.

        Returns:

        """
        return self.__uri

    def new_h5(self, rdcc_nbytes: int = None) -> h5py.File:
        """Instantiate a new h5py.File.

         Can be used in multiprocessing so each persistent worker can get a file handle and their cache instead of
         discarding them after each reading operation.

        Args:
            rdcc_nbytes: number of bytes for chunk cache. See `h5py` documentation.

        Returns:

        """
        # hdf5 supports one writer + multiple readers a time.
        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        if rdcc_nbytes is None:
            rdcc_nbytes = H5Reader.CACHE_SIZE
        return h5py.File(self.uri, mode='r', rdcc_nbytes=rdcc_nbytes, rdcc_w0=0.25)

    @contextmanager
    def get_h5root(self, rdcc_nbytes: Optional[int] = None) -> h5py.File:
        """Context manager version of getting h5py.File handle. Release the handle upon exit.

        This is useful when your use case need better management of file handles: e.g., only use the reader but without
        using dl_pipeline.dataset.h5dataset.CachedH5Dataset

        Args:
            rdcc_nbytes: number of bytes for chunk cache. See `h5py` documentation.

        Yields:
            h5py.File: a new file handle that will be closed upon exit.
        """
        root = self.new_h5(rdcc_nbytes=rdcc_nbytes)
        yield root
        root.close()

    def __init__(self, uri: str, array_fields_read: List[str], primary_field: str = H5ParserCore.CONST_TILE):
        """Generic reader of h5 data.

        Args:
            uri: uri of HDF5 file
            array_fields_read: list of array fields to be read and packed in __getitem__
            primary_field: primary field name that determine the dataset size. By default, it's the tile.
        """
        self.__uri = uri
        self.primary_field = primary_field
        self.array_fields_read = array_fields_read

    def __len__(self) -> int:
        """length of the primary data

        Returns:
            int: length
        """
        with self.get_h5root() as h5root:
            return len(h5root[self.primary_field])

    def get_item_helper(self, h5root: h5py.File, index) -> Dict:
        """Read i-th elements of all fields specified in array_fields_read

        Can be used in HDF5 dataset.

        Args:
            h5root: the h5py.File handle
            index: index loc of all preset "array_fields_read". Note if index exceeds the boundary of any dataset,
                an exception will be raised.

        Returns:
            Dict of i-th elements from each field: Dict[field, i-th element of the field]
        """
        field_list = self.array_fields_read
        # with self.get_h5root() as h5root:
        # not_a_dataset = [x for x in field_list if x not in h5root]
        # assert len(not_a_dataset) == 0, f"{H5Reader.ERROR_FIELD_NOT_CREATED}: {not_a_dataset}"
        o_dict = dict()
        for name in field_list:
            o_dict[name] = h5root[name][index]

        return o_dict
