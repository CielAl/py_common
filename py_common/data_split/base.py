from typing import runtime_checkable, Protocol, Iterable, List, Optional, Literal, Tuple, Union  # TypedDict
import numpy as np
from typing_extensions import TypedDict, NotRequired

TYPE_INDICES_LIST = Union[np.ndarray, List[int]]
TYPE_SPLITS = Tuple[TYPE_INDICES_LIST, TYPE_INDICES_LIST]

@runtime_checkable
class Splitter(Protocol):

    def split(self, X, y, group) -> Iterable:
        pass

    def get_n_splits(self, X, y, group) -> int:
        pass


SPLIT_DATA_KEYS = Literal['file_list', 'label_list', 'group_list']


class SplitOut(TypedDict):
    cohort: str

    file_list: List  # in some cases labels or extra info can be grouped with uri as a tuple.
    label_list: Optional[List]  #
    group_list: Optional[List]
    split_indices: List[TYPE_SPLITS]
    inner_split: NotRequired[Optional['SplitOut']]

