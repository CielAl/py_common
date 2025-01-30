from .json import write_json, load_json
from .pickle import write_pickle, load_pickle
from .yaml import load_yaml
from typing import Literal, Callable

TYPE_JSON = Literal['json']
TYPE_PICKLE = Literal['pickle']
TYPE_YAML = Literal['yaml']

TYPE_SUPPORTED = Literal[TYPE_PICKLE, TYPE_JSON, TYPE_YAML]


def _loader(method: TYPE_SUPPORTED) -> Callable:
    match method:
        case 'json':
            return load_json
        case 'pickle':
            return load_pickle
        case 'yaml':
            return load_yaml
        case _:
            raise ValueError(f"Unsupported {method}")


def _writer(method: TYPE_SUPPORTED) -> Callable:
    match method:
        case 'json':
            return write_json
        case 'pickle':
            return write_pickle
        case _:
            raise ValueError(f"Unsupported {method}")


def write_data(fname: str, method: TYPE_SUPPORTED, data, **kwargs):
    writer_func = _writer(method)
    return writer_func(fname, data, **kwargs)


def load_data(fname: str, method: TYPE_SUPPORTED, **kwargs):
    loader_func = _loader(method)
    return loader_func(fname, **kwargs)
