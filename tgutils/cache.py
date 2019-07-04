"""
Thread-safe cache.
"""

from multiprocessing import Lock
from typing import Callable
from typing import Dict
from typing import Generic
from typing import TypeVar

Key = TypeVar('Key')
Value = TypeVar('Value')


class Cache(Generic[Key, Value]):  # pylint: disable=too-few-public-methods
    """
    Thread-safe cache.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: Dict[Key, Value] = {}

    def lookup(self, key: Key, compute_value: Callable[[], Value]) -> Value:
        """
        Lookup a value by its key, computing it only if this is the first lookup.
        """
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                return value
            value = compute_value()
            self._data[key] = value
            return value

    def __contains__(self, name: str) -> bool:
        return name in self._data
