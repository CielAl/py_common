from .data_class import ModelInput
from typing import Callable, Union, Optional
from .base import AbstractDataset
from .cached import CachedDataset
from copy import deepcopy


class TransformSet(AbstractDataset, CachedDataset):
    """
    Wrapper dataset to add on-the-fly augmentation.
    """
    _dataset: AbstractDataset
    _transforms: Union[Callable, None]
    _copy_flag: bool
    _keep_original: bool

    def __len__(self):
        return len(self._dataset)

    def new_cache(self):
        return self._dataset.new_cache()

    def __init__(self, dataset: AbstractDataset, transforms: Callable,
                 keep_original: bool = False,
                 copy_flag: bool = False,
                 target_transforms: Optional[Callable] = None):
        """

        Args:
            dataset: associated dataset.
            transforms: augmentation functions. If no transforms then set it to None.
            keep_original: Whether to keep the copy of data into "original" field of ModelInput
            copy_flag: Whether to perform deepcopy of the data. Otherwise just keep the reference, e.g., if the
                transformation itself is not in-place and create the copy then it's not necessary to copy again.
            target_transforms: transformation over ground truth - e.g., add ToTensor or PILToTensor
                to gauge the channel order.

            Warning:
                Since transforms and target_transforms are performed in parallel, it is not recommended to
                use two augmentation with random parameters in most of cases.
        """
        super().__init__()
        self._dataset = dataset
        self._transforms = transforms
        self._copy_flag = copy_flag
        self._keep_original = keep_original

        self._target_transforms = target_transforms

    def fetch(self, index) -> ModelInput:
        data: ModelInput = self._dataset[index]
        if self._keep_original:
            data['original'] = data['data'] if not self._copy_flag else deepcopy(data['data'])
        else:
            data['original'] = AbstractDataset.DEFAULT_VALUE

        if self._transforms is not None:
            data['data'] = self._transforms(data['data']) if data['data'] is not None else None

        if self._target_transforms is not None:
            data['ground_truth'] = self._target_transforms(data['ground_truth']) \
                if data['ground_truth'] is not None else None
        return data

    @classmethod
    def build(cls, dataset: AbstractDataset, transforms: Callable,
              keep_original: bool = False,
              copy_flag: bool = False,
              target_transforms: Optional[Callable] = None):
        return cls(dataset=dataset, transforms=transforms, keep_original=keep_original, copy_flag=copy_flag,
                   target_transforms=target_transforms)
