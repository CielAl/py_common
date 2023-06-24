from ..base import AbstractDataset
from ..data_class import ModelInput
from typing import List, Generic, TypeVar, Callable
from abc import abstractmethod
TYPE_SAMPLE = TypeVar('TYPE_SAMPLE')


class SampleListSet(AbstractDataset, Generic[TYPE_SAMPLE]):
    _sample_list: List[TYPE_SAMPLE]
    _parse_func: Callable[[TYPE_SAMPLE], ModelInput]

    @property
    def sample_list(self):
        return self._sample_list

    def __init__(self, sample_list: List[TYPE_SAMPLE]):
        #  parse_func: Callable[[TYPE_SAMPLE], ModelInput]
        # self._parse_func = parse_func
        self._sample_list = sample_list

    def __len__(self):
        return len(self._sample_list)

    @abstractmethod
    def parse_func(self, sample: TYPE_SAMPLE) -> ModelInput:
        raise NotImplementedError

    def fetch(self, index) -> ModelInput:
        sample = self._sample_list[index]
        data = self.parse_func(sample)
        return data
