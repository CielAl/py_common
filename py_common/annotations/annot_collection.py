from typing import List, Dict, Union, Type, Literal, get_args, Tuple, Mapping
from types import MappingProxyType
# from shapely.strtree import STRtree
# from shapely.geometry import box as shapely_box
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import numpy as np
from shapely import affinity
from lazy_property import LazyProperty
from .annotation.base import Annotation, Region, TYPE_RAW_LABEL
from .annotation.imagescope import ImageScopeAnnotation
from .annotation.geojson import GEOJsonAnnotation

TYPE_BBOX = Tuple[int, int, int, int]

TYPE_GEO = Literal["geojson"]
TYPE_IMAGESCOPE = Literal["imagescope"]
TYPE_SUPPORTED_PARSER = Literal[TYPE_GEO, TYPE_IMAGESCOPE]

PARSER_BUILDER_MAP: Dict[str, Type[Annotation]] = {
    get_args(TYPE_GEO)[0]: GEOJsonAnnotation,
    get_args(TYPE_IMAGESCOPE)[0]: ImageScopeAnnotation
}


class AnnotCollection:
    _annotation_list: List[Annotation]
    _label_to_regions_map: Mapping[TYPE_RAW_LABEL, List[Region]]

    @LazyProperty
    def all_regions(self) -> List[Region]:
        region_list = []
        for annotation in self._annotation_list:
            region_list += annotation.regions
        return region_list

    # @LazyProperty
    # def multipolygons(self) -> MultiPolygon:
    #     for annotation in self._annotation_list:

    def __init__(self, annotation_list: List[Annotation]):
        self._annotation_list = annotation_list
        self._label_to_regions_map = self._new_label_to_regions_map()

    @classmethod
    def build(cls, parser_type: TYPE_SUPPORTED_PARSER, uri: str, label_map: Union[Dict[Union[str, int], int], None]):
        construct = PARSER_BUILDER_MAP[parser_type]
        annotation_list = construct.build_from_uri(uri=uri, label_map=label_map)
        return cls(annotation_list)

    def _new_label_to_regions_map(self) -> Mapping[TYPE_RAW_LABEL, List[Region]]:
        out_dict: Dict[TYPE_RAW_LABEL, List[Region]] = dict()
        for region in self.all_regions:
            region: Region
            label: TYPE_RAW_LABEL = region['label']
            out_dict[label] = out_dict.get(label, [])
            out_dict[label].append(region)
        return MappingProxyType(out_dict)

    @property
    def label_to_regions_map(self) -> Mapping[TYPE_RAW_LABEL, List[Region]]:
        return self._label_to_regions_map

    @staticmethod
    def rescale_by_img_bbox(polygon: Polygon, offset_xy: Tuple[float, float], resize_factor: float) -> Polygon:
        if isinstance(offset_xy, float):
            offset_xy = (offset_xy, offset_xy)
        x_off, y_off = offset_xy
        polygon = affinity.translate(polygon, xoff=x_off, yoff=y_off)
        polygon = affinity.scale(polygon, xfact=resize_factor, yfact=resize_factor, origin=(0, 0))
        return polygon

    @staticmethod
    def polygon_filled(draw_pil: ImageDraw, polygon: Polygon, offset_xy: Tuple[float, float], resize_factor: float):
        polygon = AnnotCollection.rescale_by_img_bbox(polygon, offset_xy, resize_factor)
        # outer
        exterior_coords = list(polygon.exterior.coords)
        draw_pil.polygon(exterior_coords, fill=1, outline=1, width=0)
        for component in polygon.interiors:
            interior_coord = list(component.coords)
            draw_pil.polygon(interior_coord, fill=0, outline=0, width=0)
        return draw_pil

    def annotation_to_mask(self, width: int, height: int, offset_xy: Tuple[float, float],
                           resize_factor: float):
        # binary
        mask = Image.new(mode="1", size=(width, height))
        draw_pil = ImageDraw.Draw(mask)
        all_regions: List[Region] = self.all_regions
        for region in all_regions:
            polygon: Polygon = region['polygon']
            if polygon.is_empty or (not polygon.is_valid):
                continue
            draw_pil = AnnotCollection.polygon_filled(draw_pil, polygon, offset_xy, resize_factor)
        # noinspection PyTypeChecker
        return np.array(mask)