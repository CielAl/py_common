from .sample_list_set import SampleListSet
from ..data_class import ModelInput
from typing import Tuple, Union, List, Dict
import torch
from torch.utils.data import get_worker_info, DataLoader
import numpy as np
from py_common.import_openslide import openslide
from functools import partial
from py_common.loggers.global_logger import GlobalLoggers

logger = GlobalLoggers.instance().get_logger(__name__)


# left/top/right/bottom half open on right and bottom
TYPE_WSI_BBOX = Tuple[int, int, int, int]
TYPE_LABEL = Union[int, torch.Tensor, np.ndarray]

# openslide does not have public fname attribute (protected only)
TYPE_OSH_COMPOUND = Tuple[str, openslide.OpenSlide]
TYPE_WSI_SAMPLE_STR = Tuple[str, TYPE_WSI_BBOX, TYPE_LABEL]
TYPE_SAMPLE_OSH = Tuple[TYPE_OSH_COMPOUND, TYPE_WSI_BBOX, TYPE_LABEL]
TYPE_SAMPLE_OSH_OR_STR = Union[TYPE_WSI_SAMPLE_STR, TYPE_SAMPLE_OSH]


class SlideBBoxSet(SampleListSet[TYPE_WSI_SAMPLE_STR]):

    # default 0
    __level: int

    @staticmethod
    def size_verify_helper(sample: TYPE_WSI_SAMPLE_STR, new_size: int):
        assert isinstance(new_size, int)
        uri, bbox, label, *rest = sample
        left, top, right, bottom = bbox
        bbox_new = left, top, left + new_size, top + new_size
        return uri, bbox_new, label

    @staticmethod
    def sample_verify_tile_size(sample_list: List[TYPE_WSI_SAMPLE_STR], new_size: Union[int, None]):
        if new_size is None:
            return sample_list
        return [SlideBBoxSet.size_verify_helper(sample, new_size) for sample in sample_list]

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
    def parse_helper(osh, filename, bbox, label, level):
        location, size = SlideBBoxSet.bbox_parse(bbox)
        pil = osh.read_region(location, level, size).convert("RGB")
        pil = np.array(pil, copy=True)
        output = ModelInput(data=pil, ground_truth=label, original=0, filename=filename, meta=bbox)
        return output

    @staticmethod
    def slide_parse_func(sample: TYPE_WSI_SAMPLE_STR, level: int):
        filename, bbox, label, *rest = sample
        with openslide.OpenSlide(filename) as osh:
            return SlideBBoxSet.parse_helper(osh, filename, bbox, label, level)

    def parse_func(self, sample: TYPE_WSI_SAMPLE_STR) -> ModelInput:
        return SlideBBoxSet.slide_parse_func(sample, self.level)

    @classmethod
    def build(cls, sample_list: List[TYPE_WSI_SAMPLE_STR],
              level: int,
              new_size: int = None):
        sample_list = SlideBBoxSet.sample_verify_tile_size(sample_list, new_size)
        obj = cls(sample_list)
        obj.level = level
        return obj


class CachedSlideBBoxSet(SlideBBoxSet):

    # only set by worker init func
    _cache: Dict[str, openslide.OpenSlide]

    def new_cache(self) -> Dict[str, openslide.OpenSlide]:
        return CachedSlideBBoxSet.new_osh_dict(self.sample_list)

    @staticmethod
    def slide_parse_func(sample: TYPE_SAMPLE_OSH, level: int):
        uri, bbox, label, *rest = sample
        assert isinstance(uri, Tuple)
        filename, osh,  = uri
        assert isinstance(filename, str)
        assert isinstance(osh, openslide.OpenSlide)
        return CachedSlideBBoxSet.parse_helper(osh, filename, bbox, label, level)

    def sample_from_cache(self, sample: TYPE_WSI_SAMPLE_STR):
        filename, bbox, label, *rest = sample
        if not self.is_cached(filename):
            return sample
        # read from cached
        osh = self._cache[filename]
        return (filename, osh), bbox, label

    def parse_func(self, sample: TYPE_SAMPLE_OSH_OR_STR) -> ModelInput:
        sample = self.sample_from_cache(sample)
        uri, bbox, label, *rest = sample
        if isinstance(uri, str):
            return super().parse_func(sample)
        return CachedSlideBBoxSet.slide_parse_func(sample, self.level)

    @staticmethod
    def new_osh_dict(sample_list: List[TYPE_WSI_SAMPLE_STR]):
        wsi_set = set(x[0] for x in sample_list)
        return {x: openslide.OpenSlide(x) for x in wsi_set}

    @classmethod
    def build_loader(cls, sample_list: List[TYPE_WSI_SAMPLE_STR], level,
                     new_size: int = None,
                     num_workers: int = 0,
                     **kwargs) -> DataLoader:
        dataset = cls.build(sample_list, level, new_size)
        return cls.new_dataloader(dataset, num_workers, **kwargs)

