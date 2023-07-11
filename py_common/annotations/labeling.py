from .annot_collection import AnnotCollection
from typing import Tuple, List
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon
from shapely.strtree import STRtree
TYPE_BBOX = Tuple[int, int, int, int]


class Labeler:
    _bbox_list: List[TYPE_BBOX]
    _tiles: List[Polygon]
    _annot_collection: AnnotCollection
    _tile_tree: STRtree

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

    def __init__(self, bbox_list: List[TYPE_BBOX], annot_collection: AnnotCollection):
        self._bbox_list = [Labeler._bbox_sanitized(bbox) for bbox in bbox_list]
        self._tiles = [shapely_box(*bbox) for bbox in bbox_list]
        self._annot_collection = annot_collection
        self._tile_tree = STRtree(self._tiles)
