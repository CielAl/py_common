from typing import TypedDict, Literal, Union
# import numpy as np
import pandas as pd
TYPE_SPLIT = Union[Literal['train'], Literal['val'], Literal['test']]


class BaseDataPoint(TypedDict):
    uri: str
    label: int
    split: TYPE_SPLIT

