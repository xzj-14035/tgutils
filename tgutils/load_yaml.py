"""
Load data from YAML files.
"""

from typing import Any
from typing import Dict
from typing import Optional

import yaml


def load_dictionary(path: str, data: Any = None, *,
                    allowed_keys: Optional[Dict[str, type]] = None,
                    required_keys: Optional[Dict[str, type]] = None,
                    key_type: type = str,
                    value_type: Optional[type] = None) -> Dict[Any, Any]:
    """
    Load a dictionary with string keys a YAML or JSON file.

    Parameters
    ----------
    path
        The path of the YAML/JSON file.
    data
        Optional data loaded from the file.
        If this is ``None``, the file is loaded instead.
    allowed_keys
        An optional dictionary of allowed keys,
        where the value is the expected type of the loaded value.
        If not ``None``, other keys are rejected (unless listed in `required_keys`).
    required_keys
        An optional dictionary of required_keys,
        where the value is the expected type of the loaded value.
        If not ``None``, specified keys that are missing from the loaded data are an error.
    key_type
        The expected type of the keys, ``str`` by default.
    value_type
        An optional type.
        If not ``None``
    Returns
    -------
    Dict[str, Any]
        The loaded dictionary.
    """
    if data is None:
        with open(path, 'r') as file:
            data = yaml.safe_load(file.read())

    if not isinstance(data, dict):
        raise RuntimeError('The file: %s '
                           'does not contain a top-level mapping'
                           % path)

    if required_keys is None:
        required_keys = {}

    for key, value in data.items():
        if not isinstance(key, key_type):
            raise RuntimeError('A key is a: %s '
                               'instead of: %s '
                               'in the file: %s'
                               % (key.__class__, key_type, path))
        if allowed_keys is not None and key not in allowed_keys and key not in required_keys:
            raise RuntimeError('Unexpected key: %s '
                               'in the file: %s'
                               % (key, path))

    for key in required_keys:
        if key not in data:
            raise RuntimeError('Missing key: %s '
                               'in the file: %s'
                               % (key, path))

    if allowed_keys is None:
        allowed_keys = {}

    for key, value in data.items():
        verify_type(path, "key", key, value, allowed_keys.get(key, None))
        verify_type(path, "key", key, value, required_keys.get(key, None))
        verify_type(path, "key", key, value, value_type)

    return data


def verify_type(path: str, element_kind: str, element_identifier: str,
                value: Any, expected_type: Optional[type]) -> None:
    """
    Verify the type of an element loaded from a YAML/JSON file.

    If the value has an unexpected type, throws a ``RuntimeError``.

    Parameters
    ----------
    path
        The path of the loaded YAML/JSON file.
    element_kind
        The kind of element this is (for the error message).
    element_identifier
        The identifier of the element (unique within its kind).
    value
        The loaded value of the element.
    expected_type
        The expected Python class the value should be an instance of.
    """
    if expected_type is not None and not isinstance(value, expected_type):
        raise RuntimeError('The %s: %s '
                           'has a value of the type: %s '
                           'instead of the expected type: %s '
                           'in the file: %s'
                           % (element_kind, element_identifier,
                              value.__class__,
                              expected_type,
                              path))
