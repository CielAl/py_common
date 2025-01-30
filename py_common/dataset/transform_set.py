from .data_class import ModelInput, TYPE_MODEL_INPUT
from typing import Callable, Union, Optional, Dict, Mapping
from .base import AbstractDataset
from .cached import CachedDataset
from copy import deepcopy
from warnings import warn

TYPE_TRANSFORMS_DICT = Dict[TYPE_MODEL_INPUT, Callable]


class TransformSet(AbstractDataset, CachedDataset):
    """
    Wrapper dataset to add on-the-fly augmentation.
    """
    _dataset: AbstractDataset
    _transforms: Union[Callable, None]
    _copy_flag: bool
    _keep_original: bool
    _transforms_dict: Optional[Dict[TYPE_MODEL_INPUT, Callable]]

    @property
    def dataset(self):
        return self._dataset
    
    def __len__(self):
        return len(self._dataset)

    def new_cache(self):
        return self._dataset.new_cache()

    def init_cache(self, cache: Optional[Dict] = None):
        """
        Invoke to initialize the _cache field in worker_init_fn or in factory methods depending on whether there
        are multiple workers.

        Returns:

        """

        self._cache = self.new_cache() if cache is None else cache
        self._dataset.init_cache(self._cache)

    def __init__(self, dataset: AbstractDataset, transforms: Optional[Callable] = None,
                 keep_original: bool = False,
                 copy_flag: bool = False,
                 target_transforms: Optional[Callable] = None,
                 transforms_dict: Optional[Dict[TYPE_MODEL_INPUT, Callable]] = None):
        """

        Args:
            dataset: associated dataset.
            transforms: augmentation functions. If no transforms then set it to None.
            keep_original: Whether to keep the copy of data into "original" field of ModelInput
            copy_flag: Whether to perform deepcopy of the data. Otherwise just keep the reference, e.g., if the
                transformation itself is not in-place and create the copy then it's not necessary to copy again.
            target_transforms: transformation over ground truth - e.g., add ToTensor or PILToTensor
                to gauge the channel order.
            transforms_dict: key of data to transform --> transform function for other fields

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
        
        self._transforms_dict = self.__class__.curate_transforms(transforms_dict)

    @staticmethod
    def curate_transforms(misc_transforms_dict: Optional[Dict[TYPE_MODEL_INPUT, Callable]],
                          *exclude_keys: str):
        if misc_transforms_dict is None:
            return misc_transforms_dict
        assert isinstance(misc_transforms_dict, Mapping), f"{type(misc_transforms_dict)}"
        for exclude in exclude_keys:
            assert exclude not in misc_transforms_dict, f"{exclude}"
        return misc_transforms_dict
    
    @staticmethod
    def transforms_model_input(data: ModelInput, key: TYPE_MODEL_INPUT, transform_func: Callable):
        data[key] = transform_func(data[key])
        return data

    @classmethod
    def transforms_by_func_dict(cls, data: ModelInput,
                                transforms_dict: Optional[Dict[TYPE_MODEL_INPUT, Callable]]):
        if transforms_dict is None:
            return data
        for key, func in transforms_dict.items():
            data = cls.transforms_model_input(data, key, func)
        return data

    def fetch(self, index) -> ModelInput:
        data: ModelInput = self._dataset[index]
        if self._keep_original:
            data['original'] = data['data'] if not self._copy_flag else deepcopy(data['data'])
        else:
            data['original'] = AbstractDataset.DEFAULT_VALUE

        if self._transforms is not None:
            warn("transforms deprecated", DeprecationWarning, stacklevel=2)
            data['data'] = self._transforms(data['data']) if data['data'] is not None else None

        if self._target_transforms is not None:
            warn("target transforms deprecated", DeprecationWarning, stacklevel=2)
            data['ground_truth'] = self._target_transforms(data['ground_truth']) \
                if data['ground_truth'] is not None else None
        data = self.__class__.transforms_by_func_dict(data, self._transforms_dict)
        return data

    @classmethod
    def build(cls, dataset: AbstractDataset, transforms: Optional[Callable] = None,
              keep_original: bool = False,
              copy_flag: bool = False,
              target_transforms: Optional[Callable] = None,
              transforms_dict: Optional[Dict[TYPE_MODEL_INPUT, Callable]] = None):
        return cls(dataset=dataset, transforms=transforms, keep_original=keep_original, copy_flag=copy_flag,
                   target_transforms=target_transforms, transforms_dict=transforms_dict)
