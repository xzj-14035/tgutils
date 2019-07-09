"""
Thread-safe cache.
"""

from multiprocessing import Lock
from typing import Callable
from typing import Dict
from typing import Generic
from typing import TypeVar

import weakref

Key = TypeVar('Key')
Value = TypeVar('Value')


class Cache(Generic[Key, Value]):  # pylint: disable=too-few-public-methods
    """
    Thread-safe cache.
    """

    _reset_lock = Lock()
    _reset_caches: weakref.WeakSet = weakref.WeakSet()

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: Dict[Key, Value] = {}
        with Cache._reset_lock:
            Cache._reset_caches.add(self)

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

    def __contains__(self, key: Key) -> bool:
        return key in self._data

    @staticmethod
    def reset() -> None:
        """
        Clear all the caches (for tests).
        """
        with Cache._reset_lock:
            for cache in Cache._reset_caches:
                cache._data.clear()  # pylint: disable=protected-access
