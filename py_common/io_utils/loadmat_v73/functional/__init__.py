import h5py
from typing import Sequence, List, Dict
from ..path_tree.tree import PathTree
import numpy as np


class HDF5Mat:
    """A wrapper class to contain all relevant functions to parse the V7.3 Mat which is a HDF5 file.

    """
    @staticmethod
    def should_exclude(path: str | Sequence[str], level_excludes: PathTree):
        """Check whether a given path is excluded (registered in level_excludes)

        The path must be in the path tree and also be a leaf node for exclusion.

        Args:
            path: a fullpath str, or a list of components split by a delimiter.
            level_excludes: A PathTree registered all paths to exclude

        Returns:
            bool
        """
        return path in level_excludes and level_excludes.is_leaf(path)

    @staticmethod
    def parse_dataset(root: h5py.Group, dataset: h5py.Dataset, encoding: str = 'utf-16') -> List[str] | np.ndarray:
        """Helper function to parse the HDF5 dataset.

         Assume if the dtype is object, it is a dataset of strings follows the given encoding.

        Args:
            root: root of the HDF5 file
            dataset: h5py.Dataset
            encoding: encoding of chars.

        Returns:
            np.ndarray for numerical data and list for str.
        """
        if dataset.dtype.kind == 'O':
            return HDF5Mat.parse_hdf5_string_dataset(root, dataset=dataset, encoding=encoding)
        else:
            return np.array(dataset)

    @staticmethod
    def parse_hdf5_string_dataset(root: h5py.Group, *, dataset: h5py.Dataset, encoding: str = 'utf-16') -> List[str]:
        """Helper function to parse the string dataset, e.g., a cell array of chars in matlab.

        Args:
            root: root group
            dataset: h5py.Dataset
            encoding: encoding of char

        Returns:
            list of str
        """
        string_list = []
        for ref in dataset:
            obj = root[ref[0]]
            string_data = obj[()].tobytes().decode(encoding)
            string_list.append(string_data)
        return string_list

    @staticmethod
    def parse_group(root: h5py.Group, group: h5py.Group,
                    path_stack: Sequence[str], exclude_path_tree: PathTree, encoding: str = 'utf-16') -> Dict:
        """Parse the given h5py.Group using the encoding.

        Args:
            root:
            group:
            path_stack:
            exclude_path_tree:
            encoding:

        Returns:

        """
        data = dict()
        for key, item in group.items():
            current_stack = tuple(path_stack) + (key,)
            if HDF5Mat.should_exclude(current_stack, exclude_path_tree):
                continue
            if isinstance(item, h5py.Dataset):
                data[key] = HDF5Mat.parse_dataset(root, item, encoding)
            elif isinstance(item, h5py.Group):
                path_stack = current_stack
                group_data = HDF5Mat.parse_group(root, item, path_stack, exclude_path_tree, encoding)
                if group_data:
                    data[key] = group_data
        return data

    @staticmethod
    def apply_squeeze(data: np.ndarray | Dict) -> np.ndarray | Dict:
        """A helper to squeeze the np.ndarray

        Args:
            data: np.ndarray or a dict that may contain np.ndarray. If the type is dict then a recursive search will be
                performed to squeeze or underlying numpy arrays.

        Returns:
            squeezed data.
        """
        if isinstance(data, np.ndarray):
            return np.squeeze(data)
        elif isinstance(data, dict):
            return {k: HDF5Mat.apply_squeeze(v) for k, v in data.items()}
        return data

    @staticmethod
    def loadmat_v7_3(file_path: str,
                     squeeze: bool = False,
                     exclude_list: List[str] = None,
                     encoding: str = 'utf-16') -> Dict:
        """Interface to parse v7.3 mat.

        Args:
            file_path: path of the mat file
            squeeze: whether to squeeze the numerical data
            exclude_list: variables that are excluded from reading. To exclude specific fields from a mat struct,
                follow the convention of struct/field_1/child_of_field_1, similar to the fashion of a path.
                Fields can be nested.
            encoding: encoding for str data.

        Returns:
            data read from the mat as a dict, following the convention of varname->data.
            Struct are treated as a dict which can be nested.
        """
        # Convert exclude_list to a hierarchical structure
        exclude_path_tree = PathTree.build(exclude_list)

        with h5py.File(file_path, 'r') as file:
            path_stack = tuple()
            data = HDF5Mat.parse_group(file, file, path_stack, exclude_path_tree, encoding)
            return HDF5Mat.apply_squeeze(data) if squeeze else data
