"""
Stubs for ``mypy`` for ``numpy.random``.
"""

from numpy import ndarray
from typing import Any
from typing import Optional


def choice(*args: Any, **kwargs: Any) -> ndarray: ...


def normal(*args: Any, **kwargs: Any) -> Any: ...


def rand(*args: Any, **kwargs: Any) -> ndarray: ...


def seed(s: Optional[int]) -> None: ...
