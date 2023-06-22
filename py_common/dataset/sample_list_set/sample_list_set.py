from ..base import AbstractDataset
from ..data_class import ModelInput
from typing import List, Generic, TypeVar, Callable
TYPE_SAMPLE = TypeVar('TYPE_SAMPLE')


class SampleListSet(AbstractDataset, Generic[TYPE_SAMPLE]):
    _sample_list: List[TYPE_SAMPLE]
    _parse_func: Callable[[TYPE_SAMPLE], ModelInput]

    def __init__(self, sample_list: List[TYPE_SAMPLE], parse_func: Callable[[TYPE_SAMPLE], ModelInput]):
        self._sample_list = sample_list
        self._parse_func = parse_func

    def __len__(self):
        return len(self._sample_list)

    def fetch(self, index) -> ModelInput:
        sample = self._sample_list[index]
        data = self._parse_func(sample)
        return data
