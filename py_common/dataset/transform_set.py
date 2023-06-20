from .data_class import ModelInput
from typing import Callable, Union
from .base import AbstractDataset


class TransformSet(AbstractDataset):
    _dataset: AbstractDataset
    _transforms: Union[Callable, None]

    def __len__(self):
        return len(self._dataset)

    def __init__(self, dataset: AbstractDataset, transforms: Callable):
        super().__init__()
        self._dataset = dataset
        self._transforms = transforms

    def fetch(self, index) -> ModelInput:
        data: ModelInput = self._dataset[index]
        data['original'] = data['data']
        if self._transforms is not None:
            data['data'] = self._transforms(data['data'])
        return data

