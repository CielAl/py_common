import h5py
import numpy as np
from typing import Sequence, Union, Dict, Tuple
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def to_numpy(input_data: Union[Sequence, np.ndarray]) -> np.ndarray:
    """
    Helper function to convert any squence into numpy array
    Args:
        input_data: data to collate.

    Returns:
        numpy.ndarray of the sequence of data.
    """
    return np.asarray(input_data)


class H5Util:

    @staticmethod
    def add_array_dataset_helper(h5root: h5py.File, name: str, shape: Tuple, max_size: int = None,
                                 chunk_size=None,
                                 dtype=None):
        """
        Helper function to add an array dataset into the hdf5 root.
        Args:
            h5root: Root of the hdf5 file.
            name: Name of the added dataset
            shape: Shape of a single element in the array
            max_size: max size of the array (e.g., total num of elements)
            chunk_size: hdf5 chunk size of the array
            dtype: dtype of individual elements

        Returns:
            None
        """
        max_shape = (max_size,) + shape
        shape_all = (1,) + shape
        chunks = (chunk_size,) + shape
        h5root.create_dataset(name=name, shape=shape_all, maxshape=max_shape, chunks=chunks, dtype=dtype)

    @staticmethod
    def add_str_dataset_helper(h5root: h5py.File, name: str, str_data_in: Union[Sequence, np.ndarray]):
        """
        Helper function to add string-based dataset.
        Args:
            h5root: Root of the hdf5 file.
            name: Name of the dataset.
            str_data_in: A list of string or string np.ndarrays.

        Returns:
            None
        """
        str_data = to_numpy(str_data_in).astype(object)
        dtype_str = h5py.special_dtype(vlen=str)
        h5root.create_dataset(name=name, data=str_data, dtype=dtype_str)

    def add_array_dataset(self, name: str, shape: Tuple, max_size: int = None, chunk_size=None, dtype=None) -> "H5Util":
        """
        Fluent interface to add array dataset into the  self.h5root
        Args:
            name: Name of the added dataset
            shape: Shape of a single element in the array
            max_size: max size of the array (e.g., total num of elements)
            chunk_size: hdf5 chunk size of the array
            dtype: dtype of individual elements

        Returns:
            self
        """
        H5Util.add_array_dataset_helper(self.h5root, name=name, shape=shape, max_size=max_size, chunk_size=chunk_size,
                                        dtype=dtype)
        return self

    def add_str_dataset(self, name: str, str_data_in: Union[Sequence, np.ndarray]):
        """
        Fluent interface to add string dataset.
        Args:
            name: Name of the dataset to add
            str_data_in: A list or np.array of strings to add.

        Returns:

        """
        H5Util.add_str_dataset_helper(self.h5root, name, str_data_in)
        return self

    @staticmethod
    def validate_insert_idx(old_size, insert_idx):
        """
        Validate whether the insertion index is either in range or immediately after the end of the dataset (append).
        Args:
            old_size: previous size of the dataset
            insert_idx: 0-based index of insertion

        Returns:
            insert_idx itself if it is validated
        """
        right_most_idx = old_size - 1
        immediately_after = old_size
        assert (0 <= insert_idx <= right_most_idx) or (insert_idx == immediately_after)
        return insert_idx

    @staticmethod
    def in_current_boundary(old_size, insert_idx, insert_data_size: int):
        """
        Test if the target insert id (insert_idx) and the corresponding size of data to be inserted is within the
            current array boundary.
        Args:
            old_size: Size of the dataset 0 ~ len - 1
            insert_idx: where to insert
            insert_data_size: length of the array to insert

        Returns:
            bool
        """
        # insert_idx itself is inboundary
        assert insert_data_size > 0, f"Empty Data Encountered"

        # [insert_idx .... right_ind_included]
        right_ind_included = insert_idx + insert_data_size - 1

        right_most_idx = old_size - 1
        left_in_boundary = (0 <= insert_idx <= right_most_idx)
        right_in_boundary = (insert_idx <= right_ind_included <= right_most_idx)
        return left_in_boundary and right_in_boundary

    @staticmethod
    def update_array_size(h5root: h5py.File, dataset_name: str, data_shape: Tuple, insert_idx: int):
        """
        Extend hdf5 array and update the size if the insert_idx is beyond the current boundary given by the shape
        of data.
            Note: Since H5py dataset start with at least one element
            (no empty array support), "appending" operation
            does not work as intended because it will skip index 0. (Hyp5 create an all-zero array by default).
            Thus, for the very first item --> it must be added via insertion operation rather than appending.

        Args:
            h5root: location of the hdf5 root file.
            dataset_name: Name of dataset under the hdf5 root.
            data_shape: shape of the data in convention of (N ....) wherein the leading dimension is the number of
                instances.
            insert_idx: where to insert

        Returns:

        """
        insert_dim = len(data_shape)
        old_dataset_dim = len(h5root[dataset_name].shape)

        assert insert_dim == old_dataset_dim, f"Dim mismatch Insert: {insert_dim} vs. Dataset {old_dataset_dim}"
        insert_data_size = data_shape[0]
        old_size = h5root[dataset_name].shape[0]

        H5Util.validate_insert_idx(old_size=old_size, insert_idx=insert_idx)
        is_in_boundary = H5Util.in_current_boundary(old_size=old_size,
                                                    insert_idx=insert_idx,
                                                    insert_data_size=insert_data_size)
        # resize only if data has been appended.
        # the h5py dataset starts with at least one all-zero array
        if is_in_boundary:
            return
        # might overlap
        # new_size = right_most + 1 = (insert_idx + insert_data_size - 1) + 1
        new_size = insert_idx + insert_data_size
        h5root[dataset_name].resize(new_size, axis=0)

    def add_data_to_array(self, dataset_name: str, data: np.ndarray, insert_idx: int):
        """
        Add data (N * ...) to the h5py dataset. (Append if the insert_idx is immediately after the end of the dataset)
        A leading singleton dim must be added in prior
        Args:
            dataset_name: target dataset name
            data: data to insert
            insert_idx: insert index

        Returns:

        """
        insert_size = data.shape[0]
        H5Util.update_array_size(self.h5root, dataset_name=dataset_name, data_shape=data.shape, insert_idx=insert_idx)
        self.h5root[dataset_name][-insert_size:] = data
        return self

    @classmethod
    def build(cls, filename: str, mode: str = 'w-'):
        """
        Factory builder.
        Args:
            filename: output filename.
            mode: IO mode. Default w- (write new file. fail if exists --> avoid overwriting)

        Returns:

        """
        h5root = h5py.File(filename, mode)
        return cls(h5root)

    def __init__(self, h5root: h5py.File):
        self.h5root = h5root

    def __enter__(self):
        # self.h5root.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Automatically close the file handle after exit.
        Args:
            exc_type:
            exc_val:
            exc_tb:

        Returns:

        """
        self.h5root.__exit__(exc_type, exc_val, exc_tb)
