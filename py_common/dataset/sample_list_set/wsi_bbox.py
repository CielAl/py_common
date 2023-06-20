from .sample_list_set import SampleListSet
from ..data_class import ModelInput
from typing import Tuple, Union, List, Callable, TypeVar
import torch
import numpy as np
from py_common._import_openslide import openslide
from functools import partial

# left/top/right/bottom half open on right and bottom
TYPE_WSI_BBOX = Tuple[int, int, int, int]
TYPE_LABEL = Union[int, torch.Tensor, np.ndarray]
TYPE_WSI_SAMPLE = Tuple[str, TYPE_WSI_BBOX, TYPE_LABEL]


class SlideBBoxSet(SampleListSet[TYPE_WSI_SAMPLE]):

    # default 0
    __level: int

    @property
    def level(self):
        if not hasattr(self, '__level'):
            self.__level = 0
        return self.__level

    @level.setter
    def level(self, new_val: int):
        self.__level = new_val

    @staticmethod
    def bbox_parse(bbox: TYPE_WSI_BBOX) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        [left, top, right, bottom] --> [left, top], [width, height]
        Open on right and bottom
        Args:
            bbox:

        Returns:

        """
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        assert width > 0
        assert height > 0
        return (left, top), (width, height)

    @staticmethod
    def parse_func_level(sample: TYPE_WSI_SAMPLE, level: int):
        filename, bbox, label, *rest = sample
        location, size = SlideBBoxSet.bbox_parse(bbox)

        with openslide.OpenSlide(filename) as osh:
            pil = osh.read_region(location, level, size).convert("RGB")
        output = ModelInput(data=pil, ground_truth=label, original=None, filename=filename, meta=bbox)
        return output

    # def __init__(self, sample_list: List[TYPE_WSI_SAMPLE],
    #              parse_func: Callable[[TYPE_WSI_SAMPLE], ModelInput], level: int):
    #     super().__init__(sample_list, parse_func)

    @classmethod
    def build(cls, sample_list: List[TYPE_WSI_SAMPLE],
              level: int):
        parse_func = partial(SlideBBoxSet.parse_func_level, level=level)
        obj = cls(sample_list, parse_func)
        obj.level = level
        return obj



