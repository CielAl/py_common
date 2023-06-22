from .data_class import ModelInput
from typing import Callable, Union
from .base import AbstractDataset
from copy import deepcopy


class TransformSet(AbstractDataset):
    _dataset: AbstractDataset
    _transforms: Union[Callable, None]
    _copy_flag: bool
    _keep_original: bool
    DEFAULT: int = 0

    def __len__(self):
        return len(self._dataset)

    def __init__(self, dataset: AbstractDataset, transforms: Callable,
                 keep_original: bool = False,
                 copy_flag: bool = False):
        super().__init__()
        self._dataset = dataset
        self._transforms = transforms
        self._copy_flag = copy_flag
        self._keep_original = keep_original

    def fetch(self, index) -> ModelInput:
        data: ModelInput = self._dataset[index]
        if self._keep_original:
            data['original'] = data['data']
            if self._copy_flag:
                data['original'] = deepcopy(data['data'])
        else:
            data['original'] = TransformSet.DEFAULT

        if self._transforms is not None:
            data['data'] = self._transforms(data['data'])
        return data

