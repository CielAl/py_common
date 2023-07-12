# List[List[List[Tuple]]]
from typing import Tuple, List, NamedTuple, Union
from py_common.io_utils.json import load_json
TYPE_JSON_BBOX = Union[List[int], Tuple[int, int, int, int]]


class BBox(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


def curated_bbox(bbox: Union[BBox, TYPE_JSON_BBOX], curate_size: Union[int, None]) -> BBox:
    left, top, right, bottom = bbox
    if curate_size is None:
        return BBox(left=left, top=top, right=right, bottom=bottom)
    assert curate_size > 0
    # open on right and bottom
    right = left + curate_size
    bottom = top + curate_size
    return BBox(left=left, top=top, right=right, bottom=bottom)


def qc_read(uri: str, curate_size: Union[int, None]) -> List[BBox]:
    nested_bbox_list: List[List[TYPE_JSON_BBOX]] = load_json(uri)
    bbox_list_flatten_raw: List[TYPE_JSON_BBOX] = sum(nested_bbox_list, [])
    return [curated_bbox(x, curate_size) for x in bbox_list_flatten_raw]
