"""
Stubs for ``mypy`` for ``numpy``.
"""

from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy.char as char
import numpy.random as random


class int32(int):
    pass


class float32(float):
    pass


class ArrayLike:
    def __contains__(self, v: Any) -> bool: ...

    def __getitem__(self, i: Any) -> Any: ...

    def __len__(self) -> int: ...

    def __setitem__(self, i: Any, v: Any) -> Any: ...

    def __iter__(self) -> Iterator: ...

    def max(*args: Any, **kwargs: Any) -> Any: ...

    def min(*args: Any, **kwargs: Any) -> Any: ...

    def sum(*args: Any, **kwargs: Any) -> Any: ...


class ndarray(ArrayLike):
    dtype: Any
    size: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    T: ndarray

    def __eq__(self, v: Any) -> ndarray: ...  # type: ignore

    def __add__(self, v: Any) -> ndarray: ...

    def __and__(self, v: Any) -> ndarray: ...

    def __ge__(self, v: Any) -> ndarray: ...

    def __gt__(self, v: Any) -> ndarray: ...

    def __iadd__(self, v: Any) -> ndarray: ...

    def __idiv__(self, v: Any) -> ndarray: ...

    def __imul__(self, v: Any) -> ndarray: ...

    def __invert__(self) -> ndarray: ...

    def __le__(self, v: Any) -> ndarray: ...

    def __lt__(self, v: Any) -> ndarray: ...

    def __neg__(self) -> ndarray: ...

    def __ne__(self, v: Any) -> ndarray: ...  # type: ignore

    def __or__(self, v: Any) -> ndarray: ...

    def __sub__(self, v: Any) -> ndarray: ...

    def all(self, **kwargs: Any) -> bool: ...

    def any(self, **kwargs: Any) -> bool: ...

    def argmax(self, **kwargs: Any) -> Any: ...

    def argmin(self, **kwargs: Any) -> Any: ...

    def astype(self, t: str) -> ndarray: ...

    def copy(self) -> ndarray: ...

    def fill(self, v: Any) -> ndarray: ...

    def mean(*args: Any, **kwargs: Any) -> Any: ...

    def partition(self, *args: Any, **kwargs: Any) -> ndarray: ...

    def sort(self) -> None: ...

    def std(*args: Any, **kwargs: Any) -> Any: ...

    def var(*args: Any, **kwargs: Any) -> Any: ...


newaxis: Any


def all(a: ndarray) -> bool: ...


def absolute(a: ndarray) -> ndarray: ...


def any(a: ndarray) -> bool: ...


def arange(*args: Any, **kwargs: Any) -> ndarray: ...


def argpartition(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def argsort(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def argmax(a: ndarray, **kwargs: Any) -> Any: ...


def argmin(a: ndarray, **kwargs: Any) -> Any: ...


def array(*args: Any, **kwargs: Any) -> ndarray: ...


def average(*args: Any, **kwargs: Any) -> Any: ...


def bincount(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def clip(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def concatenate(l: Iterable[ndarray], **kwargs: Any) -> ndarray: ...


def corrcoef(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def count_nonzero(a: ndarray, *args: Any, **kwargs: Any) -> int: ...


def cumsum(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def diag(a: ndarray) -> ndarray: ...


def delete(a: ndarray, b: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def diagonal(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def empty(*args: Any, **kwargs: Any) -> ndarray: ...


def exp(*args: Any, **kwargs: Any) -> Any: ...


def fill_diagonal(a: ndarray, *args: Any, **kwargs: Any) -> None: ...


def floor(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def fmax(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def frombuffer(*args: Any, **kwargs: Any) -> ndarray: ...


def isnan(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def linspace(start: float, end: float, steps: int) -> ndarray: ...


def load(p: str, *args: Any, **kwargs: Any) -> ndarray: ...


def log(*args: Any, **kwargs: Any) -> Any: ...


def log10(*args: Any, **kwargs: Any) -> Any: ...


def log2(*args: Any, **kwargs: Any) -> Any: ...


def median(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def mean(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def min(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def minimum(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def max(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def maximum(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def nan_to_num(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def nanmedian(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def nanmax(a: Union[ndarray, List[ndarray]], *args: Any, **kwargs: Any) -> Any: ...


def nansum(a: Union[ndarray, List[ndarray]], *args: Any, **kwargs: Any) -> Any: ...


def ones(*args: Any, **kwargs: Any) -> ndarray: ...


def outer(a: ndarray, b: ndarray) -> ndarray: ...


def quantile(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def repeat(*args: Any, **kwargs: Any) -> ndarray: ...


def reshape(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def save(p: str, a: ndarray) -> None: ...


def searchsorted(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def seterr(**kwargs: Any) -> None: ...


def sort(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def sqrt(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def std(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def sum(a: Union[ndarray, List[ndarray]], *args: Any, **kwargs: Any) -> Any: ...


def unique(a: ndarray, *args: Any, **kwargs: Any) -> ndarray: ...


def var(a: ndarray, *args: Any, **kwargs: Any) -> Any: ...


def where(*args: Any, **kwargs: Any) -> ndarray: ...


def zeros(*args: Any, **kwargs: Any) -> ndarray: ...


nan: float
