"""
Generic dataset class that read from a list of samples - each sample corresponding the key information to of the
data point, e.g., URIs and labels.
"""
from ..base import AbstractDataset
from ..cached import CachedDataset
from ..data_class import ModelInput
from typing import List, Generic, TypeVar, Callable
from abc import abstractmethod

# Generic type of the individual sample
TYPE_SAMPLE = TypeVar('TYPE_SAMPLE')


class SampleListSet(AbstractDataset, CachedDataset, Generic[TYPE_SAMPLE]):
    """
    Override the parse_func to parse the sample and return the data.
    """
    _sample_list: List[TYPE_SAMPLE]
    _parse_func: Callable[[TYPE_SAMPLE], ModelInput]

    @property
    def sample_list(self) -> List[TYPE_SAMPLE]:
        """

        Returns:
            list of samples
        """
        return self._sample_list

    def __init__(self, sample_list: List[TYPE_SAMPLE]):
        """
        Input
        Args:
            sample_list:
        """
        #  parse_func: Callable[[TYPE_SAMPLE], ModelInput]
        # self._parse_func = parse_func
        self._sample_list = sample_list

    def __len__(self):
        return len(self._sample_list)

    @abstractmethod
    def parse_func(self, sample: TYPE_SAMPLE) -> ModelInput:
        """
        Read the corresponding sample and generate the ModelInput.
        Args:
            sample:

        Returns:
            Corresponding ModelInput.
        """
        raise NotImplementedError

    def fetch(self, index) -> ModelInput:
        sample = self._sample_list[index]
        data = self.parse_func(sample)
        return data
