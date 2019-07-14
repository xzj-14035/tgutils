"""
Stubs for ``mypy`` for ``pandas``.
"""

from typing import Any
from typing import Iterable
from typing import List
from typing import Sized
from typing import Tuple
from typing import Union

import numpy as _np
import pandas.core as core


class Indexed(_np.ArrayLike):
    iloc: _np.ArrayLike
    index: Index
    loc: _np.ArrayLike
    name: str
    shape: Tuple[int, ...]
    values: _np.ndarray


class DataFrame(Indexed):
    axes: List[Index]
    columns: Index
    T: DataFrame

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def __add__(self, v: Any) -> DataFrame: ...

    def __and__(self, v: Any) -> DataFrame: ...

    def __div__(self, v: Any) -> DataFrame: ...

    def __ge__(self, v: Any) -> DataFrame: ...

    def __gt__(self, v: Any) -> DataFrame: ...

    def __iadd__(self, v: Any) -> DataFrame: ...

    def __idiv__(self, v: Any) -> DataFrame: ...

    def __imul__(self, v: Any) -> DataFrame: ...

    def __invert__(self) -> DataFrame: ...

    def __le__(self, v: Any) -> DataFrame: ...

    def __lt__(self, v: Any) -> DataFrame: ...

    def __neg__(self) -> DataFrame: ...

    def __ne__(self, v: Any) -> DataFrame: ...  # type: ignore

    def __or__(self, v: Any) -> DataFrame: ...

    def __sub__(self, v: Any) -> DataFrame: ...

    def __truediv__(self, v: Any) -> DataFrame: ...

    def apply(self, *args: Any, **kwargs: Any) -> DataFrame: ...

    def astype(self, t: str) -> DataFrame: ...

    def copy(self) -> DataFrame: ...

    def corr(self) -> DataFrame: ...

    def div(self, *args: Any, **kwargs: Any) -> DataFrame: ...

    def dot(self, *args: Any, **kwargs: Any) -> DataFrame: ...

    def equals(self, o: Any) -> bool: ...

    def fillna(self, *args: Any, **kwargs: Any) -> DataFrame: ...

    def iterrows(self) -> Iterable: ...

    def groupby(self, *args: Any, **kwargs: Any) -> 'GroupedDataFrame': ...

    def mean(self, *args: Any, **kwargs: Any) -> Series: ...

    def median(self, *args: Any, **kwargs: Any) -> Series: ...

    def mul(self, *args: Any, **kwargs: Any) -> DataFrame: ...

    def transpose(self) -> DataFrame: ...

    def std(self, *args: Any, **kwargs: Any) -> Series: ...

    def var(self, *args: Any, **kwargs: Any) -> Series: ...

    def where(*args: Any, **kwargs: Any) -> DataFrame: ...


class GroupedDataFrame:

    def count(self) -> Series: ...

    def mean(self) -> DataFrame: ...

    def std(self) -> DataFrame: ...

    def sum(self) -> DataFrame: ...

    def var(self) -> DataFrame: ...


class SparseDataFrame:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def to_dense(self) -> DataFrame: ...


class Series(Indexed):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def __add__(self, v: Any) -> Series: ...

    def __and__(self, v: Any) -> Series: ...

    def __div__(self, v: Any) -> Series: ...

    def __ge__(self, v: Any) -> Series: ...

    def __gt__(self, v: Any) -> Series: ...

    def __iadd__(self, v: Any) -> Series: ...

    def __idiv__(self, v: Any) -> Series: ...

    def __imul__(self, v: Any) -> Series: ...

    def __invert__(self) -> Series: ...

    def __le__(self, v: Any) -> Series: ...

    def __lt__(self, v: Any) -> Series: ...

    def __neg__(self) -> Series: ...

    def __ne__(self, v: Any) -> Series: ...  # type: ignore

    def __or__(self, v: Any) -> Series: ...

    def __sub__(self, v: Any) -> Series: ...

    def __truediv__(self, v: Any) -> Series: ...

    def abs(self) -> Series: ...

    def astype(self, t: str) -> Series: ...

    def copy(self) -> Series: ...

    def equals(self, o: Any) -> bool: ...

    def groupby(self, *args: Any, **kwargs: Any) -> 'GroupedSeries': ...

    def isnull(self) -> Series: ...

    def iteritems(self) -> Iterable: ...

    def mean(self) -> Union[int, float]: ...

    def median(self) -> Union[int, float]: ...

    def notnull(self) -> Series: ...

    def quantile(self, *args: Any, **kwargs: Any) -> Any: ...

    def rolling(self, window_size: int) -> core.window.Rolling: ...

    def sort_values(self) -> Series: ...

    def std(self) -> Union[int, float]: ...

    def value_counts(self) -> Series: ...

    def var(self) -> Union[int, float]: ...

    def where(*args: Any, **kwargs: Any) -> DataFrame: ...


class GroupedSeries:

    def mean(self) -> Series: ...

    def median(self) -> Series: ...

    def std(self) -> Series: ...

    def sum(self) -> Series: ...

    def var(self) -> Series: ...


class Index(_np.ArrayLike):
    size: int
    str: Any
    values: _np.ndarray

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def equals(self, o: Any) -> bool: ...

    def get_indexer(self, l: Any) -> Series: ...

    def get_loc(self, l: Any) -> Union[int, float]: ...

    def intersection(self, o: Index) -> Index: ...

    def isin(self, l: List[Any]) -> Series: ...


class RangeIndex(Index):

    def __init__(self, a: int, b: int = 0) -> None: ...


def concat(*args: Any, **kwargs: Any) -> Any: ...


def isnull(v: Any) -> Any: ...


def read_csv(p: str, *args: Any, **kwargs: Any) -> DataFrame: ...
