from enum import Enum
from typing import Union


class BaseEnum(Enum):

    @classmethod
    def from_str(cls, value: str):
        assert isinstance(value, str)
        value = value.upper()
        assert value in cls.__members__, f"{value} not in {cls.__members__}"
        return cls.__members__[value]

    @classmethod
    def from_int(cls, value: int):
        assert isinstance(value, int)
        return cls._value2member_map_[value]

    @classmethod
    def build(cls, value: Union[str, int]):
        assert isinstance(value, (int, str, cls))
        match value:
            case cls():
                return value
            case int():
                return cls.from_int(value)
            case str():
                return cls.from_str(value)
            case _:
                raise NotImplementedError(f"{value} not implemented")
