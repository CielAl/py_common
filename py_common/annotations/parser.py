from typing import List, Dict, Union, Type, Literal, get_args
from .annotation.base import Annotation, Region
from .annotation.imagescope import ImageScopeAnnotation
from .annotation.geojson import GEOJsonAnnotation

TYPE_GEO = Literal["geojson"]
TYPE_IMAGESCOPE = Literal["imagescope"]
TYPE_SUPPORTED_PARSER = Literal[TYPE_GEO, TYPE_IMAGESCOPE]

PARSER_BUILDER_MAP: Dict[str, Type[Annotation]] = {
    get_args(TYPE_GEO)[0]: GEOJsonAnnotation,
    get_args(TYPE_IMAGESCOPE)[0]: ImageScopeAnnotation
}


class AnnotationParser:
    _annotation_list: List[Annotation]

    def all_regions(self) -> List[Region]:
        region_list = []
        for annotation in self._annotation_list:
            region_list += annotation.regions
        return region_list

    def __init__(self, annotation_list: List[Annotation]):
        self._annotation_list = annotation_list

    @classmethod
    def build(cls, parser_type: TYPE_SUPPORTED_PARSER, uri: str, label_map: Dict[Union[str, int], int]):
        construct = PARSER_BUILDER_MAP[parser_type]
        return construct.build_from_uri(uri=uri, label_map=label_map)