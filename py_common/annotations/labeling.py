from .annot_collection import AnnotCollection
from typing import Tuple, List, Union, Dict
from .annotation.base import TYPE_RAW_LABEL, Region
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon
from shapely.strtree import STRtree
TYPE_BBOX = Tuple[int, int, int, int]


class Labeler:
    _bbox_list: List[TYPE_BBOX]
    _tiles: List[Polygon]
    _annot_collection: AnnotCollection
    _tile_tree: STRtree

    BG_LABEL: TYPE_RAW_LABEL = 0

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
                              threshold: float = None, predicate='overlaps', distance=None):
        # label = region['label']
        polygon = region['polygon']
        raw_index = str_tree.query(polygon, predicate=predicate, distance=distance)
        all_index_in_annotation = [x for x in raw_index]
        index_threshed = Labeler._index_threshold(str_tree.geometries, region, raw_index, threshold)
        return index_threshed, all_index_in_annotation

    @staticmethod
    def query_tiles_by_label(str_tree: STRtree, annot_collection: AnnotCollection, label: TYPE_RAW_LABEL,
                             threshold: float = None, predicate='overlaps', distance=None):
        region_list: List[Region] = annot_collection.label_to_regions_map[label]
        index_threshed = []
        all_index_in_annotation = []
        for region in region_list:
            curr_threshed, curr_all_in_region = Labeler.query_tiles_in_region(str_tree, region, threshold,
                                                                              predicate, distance)
            index_threshed += curr_threshed
            all_index_in_annotation += curr_all_in_region
        return index_threshed, all_index_in_annotation

    @staticmethod
    def _new_index_dict(keys: List[TYPE_RAW_LABEL]) -> Dict:
        o_dict = dict()
        for k in keys:
            o_dict[k] = []
        return o_dict

    def query_tiles(self, threshold: float = None, predicate='overlaps', distance=None):
        all_keys = [Labeler.BG_LABEL] + list(self.annot_collection.label_to_regions_map.keys())
        index_threshed: Dict = Labeler._new_index_dict(all_keys)
        all_index_in_annotation = []
        for label in self.annot_collection.label_to_regions_map:
            index_threshed_label, in_region_label = Labeler.query_tiles_by_label(self._tile_tree,
                                                                                 self.annot_collection, label,
                                                                                 threshold, predicate, distance)
            # index_threshed[label] = index_threshed.get(label, [])
            index_threshed[label].append()
            all_index_in_annotation += in_region_label
        # index_threshed[Labeler.BG_LABEL] = index_threshed.get(Labeler.BG_LABEL, [])
        index_threshed[Labeler.BG_LABEL] += all_index_in_annotation
        return index_threshed
