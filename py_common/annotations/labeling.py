from .annot_collection import AnnotCollection
from typing import Tuple, List, Union, Dict, Literal, get_args, Any
from .annotation.base import TYPE_RAW_LABEL, Region
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from itertools import product
from lazy_property import LazyProperty
TYPE_BBOX = Tuple[int, int, int, int]

UNIQUE_ALL = Literal['all']
UNIQUE_DISCARD = Literal['discard']
UNIQUE_FIRST = Literal['first']
UNIQUE_FLAG = Literal[UNIQUE_ALL, UNIQUE_DISCARD, UNIQUE_FIRST]

_unique_set = set(get_args(UNIQUE_FLAG))

LIST_OPERAND_ADD = Literal['concat']
LIST_OPERAND_APPEND = Literal['append']
LIST_OPERANDS = Literal[LIST_OPERAND_ADD, LIST_OPERAND_APPEND]


class Labeler:
    _bbox_list: List[TYPE_BBOX]
    _tiles: List[Polygon]
    _annot_collection: AnnotCollection
    _tile_tree: STRtree

    BG_LABEL: TYPE_RAW_LABEL = 0
    __query_cache: Dict

    @staticmethod
    def _bbox_sanitized(bbox: TYPE_BBOX):
        assert len(bbox) == 4
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        assert left * top * right * bottom >= 0, f"{bbox}"
        assert width * height > 0, f"{width, height}"
        return bbox

    @property
    def tile_tree(self):
        return self._tile_tree

    @property
    def tiles(self):
        return self.tiles

    @property
    def annot_collection(self):
        return self._annot_collection

    @staticmethod
    def valid_box(bbox: TYPE_BBOX) -> Polygon:
        bbox_polygon = shapely_box(*bbox)
        if bbox_polygon.is_valid:
            return bbox_polygon
        return bbox_polygon.buffer(0)

    def __init__(self, bbox_list: List[TYPE_BBOX], annot_collection: AnnotCollection):
        self._bbox_list = [Labeler._bbox_sanitized(bbox) for bbox in bbox_list]
        self._tiles = [Labeler.valid_box(*bbox) for bbox in bbox_list]
        self._annot_collection = annot_collection
        self._tile_tree = STRtree(self._tiles)
        self.__query_cache = dict()

    @staticmethod
    def _index_threshold(tile_list: List[Polygon], region: Region, raw_indices: List[int],
                         threshold: Union[float, None]) -> List[int]:
        region_polygon = region['polygon']
        if threshold is None or threshold <= 0:
            return [x for x in raw_indices]
        # for tile in tile_list:
        #     intersect: Polygon = tile.intersection(region_polygon)
        return [index for index in raw_indices if tile_list[index].intersection(region_polygon).area > 0]

    @staticmethod
    def query_tiles_in_region(str_tree: STRtree, region: Region,
                              threshold: float = None,
                              predicate='overlaps', distance=None) -> Tuple[List[int], List[int]]:
        # label = region['label']
        polygon = region['polygon']
        raw_index = str_tree.query(polygon, predicate=predicate, distance=distance)
        all_index_in_annotation = [x for x in raw_index]
        index_threshed = Labeler._index_threshold(str_tree.geometries, region, raw_index, threshold)
        return index_threshed, all_index_in_annotation

    @staticmethod
    def query_tiles_by_label(str_tree: STRtree, annot_collection: AnnotCollection, label: TYPE_RAW_LABEL,
                             threshold: float = None,
                             predicate='overlaps', distance=None) -> Tuple[List[int], List[int]]:
        region_list: List[Region] = annot_collection.label_to_regions_map[label]
        index_threshed = []
        all_index_in_annotation = []
        for region in region_list:
            curr_threshed, curr_all_in_region = Labeler.query_tiles_in_region(str_tree, region, threshold,
                                                                              predicate, distance)
            index_threshed += curr_threshed
            all_index_in_annotation += curr_all_in_region
        return index_threshed, all_index_in_annotation

    # @staticmethod
    # def _new_dict_of_list(keys: List[TYPE_RAW_LABEL]) -> Dict:
    #     o_dict = dict()
    #     for k in keys:
    #         o_dict[k] = []
    #     return o_dict

    @staticmethod
    def _dict_value_accumulate(data_dict: Dict[Any, List], key: Any, value: Any, operand: LIST_OPERANDS):
        assert operand in set(get_args(LIST_OPERANDS))
        data_dict[key] = data_dict.get(key, [])
        # data_dict[key].append(value)
        if operand == get_args(LIST_OPERAND_ADD)[0]:
            data_dict[key] += value
        elif operand == get_args(LIST_OPERAND_APPEND)[0]:
            data_dict[key].append(value)
        else:
            pass

    def query_tiles_helper(self, threshold: float = None,
                           predicate='overlaps', distance=None) -> Dict[TYPE_RAW_LABEL, List[int]]:
        # all_keys = [Labeler.BG_LABEL] + list(self.annot_collection.label_to_regions_map.keys())
        index_threshed: Dict[TYPE_RAW_LABEL, List[int]] = dict()  # Labeler._new_dict_of_list(all_keys)
        all_index_in_annotation = []
        for label in self.annot_collection.label_to_regions_map:
            index_threshed_label, in_region_label = Labeler.query_tiles_by_label(self._tile_tree,
                                                                                 self.annot_collection, label,
                                                                                 threshold, predicate, distance)
            # index_threshed[label] = index_threshed.get(label, [])
            # index_threshed[label] += index_threshed_label
            Labeler._dict_value_accumulate(index_threshed, key=label, value=index_threshed_label, operand='concat')
            all_index_in_annotation += in_region_label
        # index_threshed[Labeler.BG_LABEL] = index_threshed.get(Labeler.BG_LABEL, [])
        index_threshed[Labeler.BG_LABEL] += all_index_in_annotation
        return index_threshed

    def query_tiles(self, threshold: float = None,
                    predicate='overlaps', distance=None) -> Dict[TYPE_RAW_LABEL, List[int]]:
        key = f"{threshold}_{predicate}_{distance}"
        if key not in self.__query_cache:
            self.__query_cache[key] = self.query_tiles_helper(threshold=threshold, predicate=predicate,
                                                              distance=distance)
        return self.__query_cache[key]

    @LazyProperty
    def _all_indices(self) -> List[int]:
        return list(range(len(self._bbox_list)))

    @staticmethod
    def unique_index_helper(index_to_label: Dict[int, List[TYPE_RAW_LABEL]], unique_flag: UNIQUE_FLAG):
        match unique_flag:
            case "all":
                return index_to_label
            case "discard":
                return {index: label_list for index, label_list in index_to_label.items() if len(label_list) == 1}
            case "first":
                return {index: label_list[:1] for index, label_list in index_to_label.items() if len(label_list) > 0}

    def get_tile_labels(self, unique_flag: UNIQUE_FLAG, threshold: float = None,
                        predicate='overlaps', distance=None) -> List[Tuple[TYPE_BBOX, TYPE_RAW_LABEL]]:

        assert unique_flag in _unique_set

        index_threshed: Dict[TYPE_RAW_LABEL, List[int]] = self.query_tiles(threshold, predicate, distance)
        index_to_label: Dict[int, List[TYPE_RAW_LABEL]] = dict()  # Labeler._new_dict_of_list(keys=self._all_indices)
        for label, index_list in index_threshed.items():
            for index in index_list:
                # index_to_label[index]
                Labeler._dict_value_accumulate(index_to_label, key=index, value=label, operand='append')

        index_to_label = Labeler.unique_index_helper(index_to_label, unique_flag)
        out_list: List[Tuple[TYPE_BBOX, TYPE_RAW_LABEL]] = []
        for index, label_list in index_to_label.items():
            bbox: TYPE_BBOX = self._bbox_list[index]
            pairs: List[Tuple[TYPE_BBOX, TYPE_RAW_LABEL]] = list(product([bbox], label_list))
            out_list += pairs
        return out_list
