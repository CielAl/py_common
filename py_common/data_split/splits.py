import json
from py_common.io_utils.json import write_json, load_json
import os
from typing import Protocol, runtime_checkable, Iterable, TypedDict, List, Tuple, Union, Dict
import numpy as np

TYPE_INDICES_LIST = Union[np.ndarray, List[int]]
TYPE_SPLITS = Tuple[TYPE_INDICES_LIST, TYPE_INDICES_LIST]


class SplitOut(TypedDict):
    cohort: str

    file_list: List
    label_list: Union[None, List]
    group_list: Union[None, List]
    split_indices: List[TYPE_SPLITS]


_SET_OUT_KEYS = SplitOut.__annotations__.keys()


class Splitter(Protocol):

    @runtime_checkable
    def split(self, X, y, group) -> Iterable:
        pass

    @runtime_checkable
    def get_n_splits(self, X, y, group) -> Iterable:
        pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DataSplit:
    __splitter: Splitter

    _export_dir: str
    _cohort_name: str
    _name_prefix: str
    _name_suffix: str

    TRAIN_IDX: int = 0
    TEST_IDX: int = 1

    @property
    def splitter(self):
        return self.__splitter

    def __init__(self, splitter: Splitter, export_dir: str, name_prefix: str, cohort_name: str,
                 name_suffix: str = None):
        self.__splitter = splitter

        self._export_dir = export_dir
        self._cohort_name = cohort_name
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix

    def _split_list(self, X, y, group=None):
        return list(self.splitter.split(X, y, group))

    @staticmethod
    def _filename_helper(cohort_name: str, name_suffix: str = None):
        return f"{cohort_name}{name_suffix}.json"

    @staticmethod
    def _savepath_helper(export_dir: str, name_prefix: str):
        return os.path.join(export_dir, name_prefix)

    def get_export_path(self, create_dir: bool = True):
        save_folder = DataSplit._savepath_helper(self._export_dir, self._name_prefix)
        if create_dir:
            os.makedirs(save_folder, exist_ok=True)
        dest_name = os.path.join(save_folder, DataSplit._filename_helper(self._cohort_name, self._name_suffix))
        return dest_name

    def get_split_data(self, X, y, group=None) -> SplitOut:
        split_data = self._split_list(X, y, group)
        return SplitOut(file_list=X, label_list=y, group_list=group, split_indices=split_data, cohort=self._cohort_name)

    @staticmethod
    def _write_to_json(fname, split_data: SplitOut):
        write_json(fname, split_data, indent=4, cls=NumpyEncoder)

    def export_split(self, X, y, group, ):
        dest_name = self.get_export_path(create_dir=True)
        split_data = self.get_split_data(X, y, group)
        DataSplit._write_to_json(dest_name, split_data)
        return split_data

    @staticmethod
    def read_split(filename: str) -> SplitOut:
        data_dict = load_json(filename)
        assert isinstance(data_dict, Dict)
        assert set(data_dict.keys()).issuperset(_SET_OUT_KEYS)
        return data_dict

    @staticmethod
    def shuffled_files(data: SplitOut, is_train: bool, which_split: int) -> List:
        file_list = data['file_list']
        split_tuple = data['split_indices'][which_split]
        split_indices = split_tuple[DataSplit.TRAIN_IDX] if is_train else split_tuple[DataSplit.TEST_IDX]

        file_arr = np.asarray(file_list)
        out = file_arr[split_indices]
        return out.tolist()
