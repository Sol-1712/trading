from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

def serialize(obj):
    """
    Recursively convert an object into standard Python types suitable for
    YAML/JSON serialization.
    """

    if is_dataclass(obj):
        return {
            field.name: serialize(
                getattr(obj, field.name)
            )
            for field in fields(obj)
        }

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            key: serialize(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            serialize(item)
            for item in obj
        ]

    if isinstance(obj, type):
        return obj.__name__

    return obj


def deserialize(data):
    """
    Recursively convert a serialized object back into its original type.
    """
    pass

