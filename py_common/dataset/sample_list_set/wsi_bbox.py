"""
SampleListSet that read list of (slide_uri, tile bbox, label) to generate tile data.
"""
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
    """
    Type of sample is Tuple[str, TYPE_WSI_BBOX, TYPE_LABEL] --> A tuple of
    (1) wsi filename.
    (2) 4-Tuple of [left, top, right, bottom] (half open on right and bottom) of tile location.
    (3) scalar label value
    """

    # default 0
    __level: int

    def new_cache(self):
        """
        Not supported in this class

        Returns:
            NotImplemented
        """
        return NotImplemented

    @staticmethod
    def size_curate_helper(sample: TYPE_WSI_SAMPLE_STR, new_size: int):
        """
        Verify and curate the size of the bbox if `new_size` is set in case the generated bboxes do not have
        homogeneous size for batch procedures.

        Args:
            sample: input sample.
            new_size: new_size to curate. Identity mapping if set to None.

        Returns:
            Curated sample if new_size is not None.
        """
        assert isinstance(new_size, int) or new_size is None
        if new_size is None:
            return sample
        uri, bbox, label, *rest = sample
        left, top, right, bottom = bbox
        bbox_new = left, top, left + new_size, top + new_size
        return uri, bbox_new, label

    @staticmethod
    def sample_curated_tile_size(sample_list: List[TYPE_WSI_SAMPLE_STR], new_size: Union[int, None]):
        """
        Verify and curate the size of the bbox if `new_size` is set in case the generated bboxes do not have
        homogeneous size for batch procedures.

        Args:
            sample_list: list of samples
            new_size: new_size to curate. Identity mapping if set to None.

        Returns:
            List of curated sample if new_size is not None.
        """
        if new_size is None:
            return sample_list
        return [SlideBBoxSet.size_curate_helper(sample, new_size) for sample in sample_list]

    @property
    def level(self):
        if not hasattr(self, '__level'):
            self.__level = 0
        return self.__level

    @level.setter
    def level(self, new_val: int):
        """
        Level of openslide.read_region
        Args:
            new_val:

        Returns:

        """
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
    def parse_helper(osh, filename, bbox, label, level) -> ModelInput:
        """
        Read the tile from openslide handles using bbox and assign the field values of ModelInput using the rest
        of args. Tile will be converted to numpy array so it will be easier to collate in dataloader.
        Args:
            osh: openslide handle
            filename: filename to write into ModelInput since osh does not have public attribute of filenames
            bbox: bbox coordinate of tile
            label: scala label value
            level: which level to read from WSI.

        Returns:

        """
        location, size = SlideBBoxSet.bbox_parse(bbox)
        pil = osh.read_region(location, level, size).convert("RGB")
        pil = np.array(pil, copy=True)
        return ModelInput(data=pil, ground_truth=label, original=0, filename=filename, meta=bbox)

    @staticmethod
    def slide_parse_func(sample: TYPE_WSI_SAMPLE_STR, level: int):
        """
        Helper function of the parse_func.
        Args:
            sample:
            level:

        Returns:

        """
        filename, bbox, label, *rest = sample
        with openslide.OpenSlide(filename) as osh:
            return SlideBBoxSet.parse_helper(osh, filename, bbox, label, level)

    def parse_func(self, sample: TYPE_WSI_SAMPLE_STR) -> ModelInput:
        """
        Use openslide.OpenSlide's `read_region` and corresponding bbox location to access tile data.
        Args:
            sample:

        Returns:

        """
        return SlideBBoxSet.slide_parse_func(sample, self.level)

    @classmethod
    def build(cls, sample_list: List[TYPE_WSI_SAMPLE_STR],
              level: int,
              new_size: int = None):
        sample_list = SlideBBoxSet.sample_curated_tile_size(sample_list, new_size)
        obj = cls(sample_list)
        obj.level = level
        return obj


class CachedSlideBBoxSet(SlideBBoxSet):
    """
    SlideBBoxSet that cached the openslide.OpenSlide handles (osh) to accelerate tile reading. Each openslide.OpenSlide
    handle internally caches the data for image decoding and reading and therefore it's significantly faster to keep
    the openslide.OpenSlide handles. Handles are stored in the _cache as dict of <filename, handle>. In multiprocessing
    the cache is initialized in the beginning each epoch of full traverse of dataloader using the
    worker_init_fn. For single-processing the _cache is initiated directly.
    Cache are only initialized when new_dataloader or build_loader is invoked to create the dataloader.

    If cache is enabled, samples will be curated to Tuple[Tuple[str, osh], TYPE_WSI_BBOX, TYPE_LABEL] and parsed
    accordingly: Tuple[Tuple[filename, osh], bbox, label]
    """

    # only set by worker init func
    _cache: Dict[str, openslide.OpenSlide]

    def new_cache(self) -> Dict[str, openslide.OpenSlide]:
        """
        Override the abstract method to pre-load all osh.

        Returns:
            Dict that map filenames to osh.
        """
        return CachedSlideBBoxSet.new_osh_dict(self.sample_list)

    @staticmethod
    def slide_parse_func(sample: TYPE_SAMPLE_OSH, level: int):
        """
        Helper function to parse curated sample: Tuple[Tuple[filename, osh], bbox, label].
        Args:
            sample: Tuple[Tuple[str, osh], TYPE_WSI_BBOX, TYPE_LABEL]. Osh is read from the cache.
            level: level to read from WSIs.

        Returns:

        """
        uri, bbox, label, *rest = sample
        assert isinstance(uri, Tuple)
        filename, osh,  = uri
        assert isinstance(filename, str)
        assert isinstance(osh, openslide.OpenSlide)
        return CachedSlideBBoxSet.parse_helper(osh, filename, bbox, label, level)

    def sample_from_cache(self, sample: TYPE_WSI_SAMPLE_STR):
        """
        Curate the sample to read the osh from cache if cache is initialized.

        Args:
            sample: original sample before curation as tuple of (filename, bbox, label)

        Returns:
            Curated sample if cache is initialized and osh are cached, or original sample if otherwise.
        """
        filename, bbox, label, *rest = sample
        if not self.is_cached(filename):
            return sample
        # read from cached
        osh = self._cache[filename]
        return (filename, osh), bbox, label

    def parse_func(self, sample: TYPE_WSI_SAMPLE_STR) -> ModelInput:
        """
        If osh is cached then curate the sample to read the tile from cached osh. Otherwise, parse using the
        super class's parse_func to read from the original sample.
        Args:
            sample: original sample

        Returns:
            ModelInput
        """
        sample = self.sample_from_cache(sample)
        uri, bbox, label, *rest = sample
        if isinstance(uri, str):
            return super().parse_func(sample)
        return CachedSlideBBoxSet.slide_parse_func(sample, self.level)

    @staticmethod
    def new_osh_dict(sample_list: List[TYPE_WSI_SAMPLE_STR]):
        """
        Helper function to generate all osh for sample list.
        Args:
            sample_list:

        Returns:

        """
        wsi_set = set(x[0] for x in sample_list)
        return {x: openslide.OpenSlide(x) for x in wsi_set}

    @classmethod
    def build_loader(cls, sample_list: List[TYPE_WSI_SAMPLE_STR], level,
                     new_size: int = None,
                     num_workers: int = 0,
                     **kwargs) -> DataLoader:
        dataset = cls.build(sample_list, level, new_size)
        return cls.new_dataloader(dataset, num_workers, **kwargs)
