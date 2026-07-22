from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

def serialize(obj):
    """
    Recursively convert an object into YAML/JSON-safe Python types.

    Handles dataclasses, Enums, Paths, dicts, lists/tuples, and types.
    Other values are returned unchanged.

    Parameters
    ----------
    obj : Any
        Object to serialize (nested structures supported).

    Returns
    -------
    Any
        Structure of built-in types suitable for yaml.safe_dump / json.dump.
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
