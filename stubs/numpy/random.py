"""
Stubs for ``mypy`` for ``numpy.random``.
"""

from typing import Any
from typing import Optional
from numpy import ndarray


def choice(*args: Any, **kwargs: Any) -> ndarray: ...


def normal(*args: Any, **kwargs: Any) -> Any: ...


def rand(*args: Any, **kwargs: Any) -> ndarray: ...


def seed(s: Optional[int]) -> None: ...
