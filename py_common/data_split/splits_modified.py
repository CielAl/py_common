from numpy._typing import _64Bit
from py_common.data_split.base import Splitter, SplitOut, SPLIT_DATA_KEYS
from py_common.io_utils.factory import write_data, load_data, TYPE_SUPPORTED
import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from numpy import int64, ndarray, dtype, signedinteger
from .base import TYPE_SPLITS

_SET_OUT_KEYS = SplitOut.__annotations__.keys()


def parse_split(split_dict: SplitOut, split_idx) -> Tuple[Tuple[List, List], Tuple[List[int], List[int]]]:
    train_idx, val_idx = split_dict['split_indices'][split_idx]
    train_samples = [split_dict['file_list'][x] for x in train_idx]
    val_samples = [split_dict['file_list'][x] for x in val_idx]
    return (train_samples, val_samples), (train_idx, val_idx)


class DataSplit:
    __splitter: Splitter

    _export_dir: str
    _cohort_name: str
    _name_prefix: str
    _name_suffix: str
    _io_method: TYPE_SUPPORTED
    _ext_val: bool

    TRAIN_IDX: int = 0
    TEST_IDX: int = 1

    DEFAULT_READER: TYPE_SUPPORTED = 'json'

    @property
    def ext_val(self):
        return self._ext_val

    @property
    def splitter(self):
        return self.__splitter

    @property
    def io_method(self) -> TYPE_SUPPORTED:
        """json or pickle, etc.

        Returns:

        """
        return self._io_method

    def __init__(self, splitter: Splitter, export_dir: str, name_prefix: str, cohort_name: str,
                 name_suffix: str = None, ext_val: bool = 1,
                 io_method: TYPE_SUPPORTED = DEFAULT_READER):
        self.__splitter = splitter
        self._ext_val = ext_val
        self._export_dir = export_dir
        self._cohort_name = cohort_name
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix
        self._io_method = io_method
    #
    # def split_ext(self, x, y, group=None):
    #
    #
    #     return

    def _split_list(self, x, y, group=None) -> List[TYPE_SPLITS]:
        """Obtain the list of splits (train idx, test idx) using the splitter assigned.

        Args:
            x:
            y:
            group:

        Returns:

        """

        return list(self.splitter.split(x, y, group))

    def _split_list_ext(self, x, y, group=None) -> list[ndarray[Any, dtype[Any]]]:
        """Obtain the list of splits (train idx, test idx) using the splitter assigned.

        Args:
            x:
            y:
            group:

        Returns:

        """
        train_idx = []
        test_idx = []
        for i in range(len(x)):
            if 'TCGA' in x[i][0]:
                test_idx.append(i)
            else:
                train_idx.append(i)
        global tra_idx
        tra_idx = np.arange(0,11082, dtype=int64)
        global tes_idx
        tes_idx = np.arange(11082,18670, dtype=int64)
        return [tra_idx, tes_idx]

    @staticmethod
    def _filename_helper(cohort_name: str, name_suffix: str = None, ext='.json'):
        return f"{cohort_name}{name_suffix}{ext}"

    @staticmethod
    def _savepath_helper(export_dir: str, name_prefix: str):
        """Helper function to join the export_dir and any prefix (e.g., folder names) as the output directory.

        Args:
            export_dir: root dir of exportation
            name_prefix: prefix. Can be one or more subdirectories along the path

        Returns:

        """
        return os.path.join(export_dir, name_prefix)

    def get_export_path(self, create_dir: bool = True, ext='.json'):
        """

        Args:
            create_dir:
            ext:

        Returns:

        """
        save_folder = DataSplit._savepath_helper(self._export_dir, self._name_prefix)
        if create_dir:
            os.makedirs(save_folder, exist_ok=True)
        dest_name = os.path.join(save_folder, DataSplit._filename_helper(self._cohort_name, self._name_suffix, ext))
        return dest_name

    def get_split_data(self, x, y, group=None) -> SplitOut:
        split_data = self._split_list(x, y, group)
        return SplitOut(file_list=x, label_list=y, group_list=group,
                        split_indices=split_data, cohort=self._cohort_name)

    def get_split_data_ext(self, x, y, group=None) -> SplitOut:
        split_data = self._split_list_ext(x, y, group)
        return SplitOut(file_list=x, label_list=y, group_list=group,
                        split_indices=split_data, cohort=self._cohort_name)
    @staticmethod
    def write_to_file(fname, method: TYPE_SUPPORTED, split_data: SplitOut, **kwargs):
        write_data(fname, method, split_data, **kwargs)  # cls=NumpyEncoder

    def export_split(self, X, y, group, ext='.json', **export_kwargs):
        dest_name = self.get_export_path(create_dir=True, ext=ext)
        split_data = self.get_split_data(X, y, group)
        DataSplit.write_to_file(dest_name, self.io_method, split_data, **export_kwargs)
        return split_data

    @staticmethod
    def read_split(filename: str, method: TYPE_SUPPORTED = DEFAULT_READER) -> SplitOut:
        """Read the split file into the typedict

        Args:
            filename:
            method:

        Returns:

        """
        data_dict = load_data(filename, method)

        assert isinstance(data_dict, Dict)
        assert set(data_dict.keys()).issuperset(_SET_OUT_KEYS)
        return data_dict

    @staticmethod
    def _shuffled_data(data: SplitOut, field: SPLIT_DATA_KEYS, is_train: bool, which_split: int,
                       dtype=object) -> List:
        target_field = data[field]
        split_tuple = data['split_indices'][which_split]
        print(f'field:{field}')
        print(f'split_tuple.shape{split_tuple.shape}')
        print(f'which_split:{which_split}')
        print(f'tra_idx.shape:{tra_idx.shape}')
        print(f'tes_idx.shape:{tes_idx.shape}')
        split_indices = split_tuple[tra_idx] if is_train else split_tuple[tes_idx]
        # print(f'split_indices:{split_indices}')
        split_field = np.asarray(target_field, dtype=dtype)[split_indices]
        # print(f'split_field: {split_field}SPLIT_DATA_KEYS{SPLIT_DATA_KEYS}')
        return split_field.tolist()

    @staticmethod
    def shuffled_files(data: SplitOut, is_train: bool, which_split: int, dtype=object) -> List:
        """Read the files of a specified split from filelist usingthe split idx

        Args:
            data:
            is_train:
            which_split:

        Returns:

        """
        return DataSplit._shuffled_data(data, 'file_list', is_train=is_train, which_split=which_split,
                                        dtype=dtype)

    @staticmethod
    def shuffled_labels(data: SplitOut, is_train: bool, which_split: int) -> List:
        """Read the files of a specified split from label_list usingthe split idx

        Args:
            data:
            is_train:
            which_split:

        Returns:

        """
        return DataSplit._shuffled_data(data, 'label_list', is_train=is_train, which_split=which_split)

    @staticmethod
    def shuffled_group(data: SplitOut, is_train: bool, which_split: int) -> List:
        """Read the files of a specified split from group_list using the split idx

        Args:
            data:
            is_train:
            which_split:

        Returns:

        """
        return DataSplit._shuffled_data(data, 'group_list', is_train=is_train, which_split=which_split)

    def nested_split(self, x, y, g, inner_split: Optional['DataSplit'] = None) -> SplitOut:
        """ Get a nested split (outer = train + held out test, inner = internal train/validation) from outer train.

        Args:
            x:
            y:
            g:
            inner_split:

        Returns:

        """
        # get internal train/test first
        if not self.ext_val:
            train_test_split_data = self.get_split_data(x, y, g)
        else:
            train_test_split_data = self.get_split_data_ext(x, y, g)
        # from training data --> get new x
        # new_x = DataSplit.shuffled_files(train_test_split_data, is_train=True, which_split=0)
        # new_y = DataSplit.shuffled_labels(train_test_split_data, is_train=True, which_split=0)
        # new_g = DataSplit.shuffled_group(train_test_split_data, is_train=True, which_split=0)
        # # within the train above, further split the train/validation
        # new_data_split = inner_split if inner_split is not None else self
        # internal_split_data = new_data_split.get_split_data(new_x, new_y, new_g)
        # train_test_split_data['inner_split'] = internal_split_data
        # return train_test_split_data
        # print(f'self.get_split_data(x, y, g){self.get_split_data(x, y, g)}')
        # print(f'self.get_split_data_ext(x, y, g){self.get_split_data_ext(x, y, g)}')
        return self.new_nested_split(train_test_split_data, inner_split)

    def new_nested_split(self, outer_split, inner_split):
        new_x = DataSplit.shuffled_files(outer_split, is_train=True, which_split=0)
        # print(f'new_x:{new_x}')
        new_y = DataSplit.shuffled_labels(outer_split, is_train=True, which_split=0)
        new_g = DataSplit.shuffled_group(outer_split, is_train=True, which_split=0)
        new_data_split = inner_split if inner_split is not None else self
        internal_split_data = new_data_split.get_split_data(new_x, new_y, new_g)
        outer_split['inner_split'] = internal_split_data
        return outer_split

    def export_nested_split(self, X, y, group, *, inner_split: Optional['DataSplit'] = None,
                            ext='.json', **export_kwargs):
        dest_name = self.get_export_path(create_dir=True, ext=ext)
        nested_out = self.nested_split(X, y, group, inner_split=inner_split)
        DataSplit.write_to_file(dest_name, self.io_method, nested_out, **export_kwargs)
        return nested_out

    @staticmethod
    def write_custom_split(fp: str, write_mode, *, cohort: str,
                           file_list: List, label_list: Optional[List], group_list: Optional[List],
                           split_indices: List[TYPE_SPLITS], **export_kwargs):
        o_dict = SplitOut(cohort=cohort, file_list=file_list,
                          label_list=label_list, group_list=group_list, split_indices=split_indices)
        write_data(fp, write_mode, o_dict, **export_kwargs)

    @staticmethod
    def match_index(outer_split: SplitOut, inner_split: SplitOut):
        train_val_set = set(np.asarray(outer_split['split_indices'][0]))
        test_set = set(np.asarray(outer_split['split_indices'][1]))
        # print(inner_split['split_indices'])
        train_set = set(np.asarray(inner_split['split_indices'][0][0]))
        val_set = set(np.asarray(inner_split['split_indices'][0][1]))
        assert train_set.isdisjoint(val_set)
        assert max(train_set.union(val_set)) + 1 <= len(train_val_set)
        assert train_val_set.isdisjoint(test_set)
        # match indices
        inner_train_indices = np.asarray(inner_split['split_indices'][0][0])
        inner_val_indices = np.asarray(inner_split['split_indices'][0][1])

        outer_train_val_indices = np.asarray(outer_split['split_indices'][0])
        outer_test_indices = np.asarray(outer_split['split_indices'][1])

        absolute_train_indices = outer_train_val_indices[inner_train_indices]
        absolute_val_indices = outer_train_val_indices[inner_val_indices]
        absolute_test_indices = outer_test_indices
        set_all = set(absolute_train_indices).union(set(absolute_val_indices)).union(set(absolute_test_indices))
        assert set_all == train_val_set.union(test_set)
        return absolute_train_indices, absolute_val_indices, absolute_test_indices