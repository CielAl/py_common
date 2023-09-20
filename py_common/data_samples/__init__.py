from typing import TypedDict, Literal, Union
# import numpy as np
TYPE_SPLIT = Union[Literal['train'], Literal['val'], Literal['test']]


class BaseDataPoint(TypedDict):
    uri: str
    label: int
    split: TYPE_SPLIT
    mask: str
    time_to_event: float
    event: int
    pid: str

