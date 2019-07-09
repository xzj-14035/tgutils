"""
Pandas utilities.

Import this as ``pd`` instead of directly importing the ``pandas`` module. It exports the same
symbols, with the addition of strongly-typed phantom classes for tracking the exact dimensions and
type of each variable using ``mypy``. It also provides some additional utilities (I/O).
"""

from pandas import *  # pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sized
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import tgutils.numpy as np

# pylint: disable=too-many-ancestors,redefined-outer-name


#: Short name for ``DataFrame``.
Frame = DataFrame

#: Type variable for data series.
S = TypeVar('S', bound='BaseSeries')  # pylint: disable=invalid-name

#: type variable for data frames.
F = TypeVar('F', bound='BaseFrame')  # pylint: disable=invalid-name


class BaseSeries(Series):
    """
    Base class for all Numpy data series phantom types.
    """

    #: The expected data type of a data series of the (derived) class.
    dtype: str

    @classmethod
    def read(cls: Type[S], path: str, mmap_mode: Optional[str] = None) -> S:
        """
        Read a Pandas data series of the concrete type from the disk.

        If an additional file with an ``.index`` suffix exists, it is loaded into the index labels.
        """
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        array = np.BaseArray._read(path, mmap_mode)  # pylint: disable=protected-access
        series = cls.am(Series(array))

        index_path = path + '.index'
        if np.BaseArray.exists(index_path):
            index = np.BaseArray._read(index_path, mmap_mode)  # pylint: disable=protected-access
            series.set_axis(index, axis=0, inplace=True)  # type: ignore

        return series

    @classmethod
    def write(cls, series: Series, path: str) -> None:
        """
        Write a Pandas data series of the concrete type to a file.

        If necessary, creates additional file with an ``.index`` suffix to preserve the index
        labels.
        """
        cls.am(series)

        np.BaseArray._write(series.values, path)  # pylint: disable=protected-access

        if not series.index.equals(RangeIndex(len(series.index))):
            np.BaseArray._write(series.index.values,  # pylint: disable=protected-access
                                path + '.index')

    @classmethod
    def am(cls: Type[S], data: Series) -> S:  # pylint: disable=invalid-name
        """
        Declare a data series as being of this type.
        """
        BaseSeries._am_series(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:
            raise ValueError('unexpected data type: %s instead of: %s'
                             % (array.dtype, cls.dtype))
        return data  # type: ignore

    @classmethod
    def be(cls: Type[S],  # pylint: disable=invalid-name
           data: Union[Series, np.ndarray, List[Any]]) -> S:
        """
        Convert an array to this type.
        """
        if isinstance(data, list):
            data = np.array(data, dtype=cls.dtype)
        if isinstance(data, np.ndarray):
            data = Series(data)

        BaseSeries._am_series(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:
            data = data.astype(cls.dtype)
        return data  # type: ignore

    @staticmethod
    def _am_series(data: Series) -> None:
        if not isinstance(data, Series):
            raise ValueError('unexpected type: %s.%s instead of: %s.%s'
                             % (data.__class__.__module__, data.__class__.__qualname__,
                                Series.__module__, Series.__qualname__))
        array = data.values
        np.BaseArray._am_shape(array, 1)  # pylint: disable=protected-access

    @classmethod
    def zeros(cls: Type[S], index: Union[int, Sized]) -> S:
        """
        Return a series full of zeros.
        """
        if isinstance(index, int):
            return cls.am(Series(np.zeros(index, dtype=cls.dtype)))
        return cls.am(Series(np.zeros(len(index), dtype=cls.dtype), index=index))

    @classmethod
    def empty(cls: Type[S], index: Union[int, Sized]) -> S:  # pylint: disable=arguments-differ
        """
        Return an uninitialized series
        """
        if isinstance(index, int):
            return cls.am(Series(np.empty(index, dtype=cls.dtype)))
        return cls.am(Series(np.empty(len(index), dtype=cls.dtype), index=index))

    @classmethod
    def filled(cls: Type[S], value: Any, shape: Union[int, Tuple[int, ...]]) -> S:
        """
        Return a series full of zeros.
        """
        series = cls.empty(shape)
        series.values.fill(value)
        return series

    @classmethod
    def shared_memory_zeros(cls: Type[S], index: Union[int, Sized]) -> S:
        """
        Create a shared memory series, initialized to zeros.
        """
        if isinstance(index, int):
            size: int = index
            index = None  # type: ignore
        else:
            size = len(index)
        return cls.am(Series(np.ARRAY_OF_DTYPE[cls.dtype].shared_memory_zeros(size), index=index))


class BaseFrame(Frame):
    """
    Base class for all Numpy data series phantom types.
    """

    #: The expected data type of a data frame of the (derived) class.
    dtype: str

    @classmethod
    def read(cls: Type[F], path: str, mmap_mode: Optional[str] = None) -> F:
        """
        Read a Pandas data frame of the concrete type from the disk.

        If additional file(s) with an ``.index`` and/or ``.columns`` suffix exist, they are loaded
        into the index and/or column labels.
        """
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        array = np.BaseArray._read(path, mmap_mode)  # pylint: disable=protected-access
        frame = cls.am(Frame(array))

        index_path = path + '.index'
        if np.BaseArray.exists(index_path):
            index = np.BaseArray._read(index_path, mmap_mode)  # pylint: disable=protected-access
            frame.set_axis(index, axis=0, inplace=True)  # type: ignore

        columns_path = path + '.columns'
        if np.BaseArray.exists(columns_path):
            columns = np.BaseArray._read(columns_path,  # pylint: disable=protected-access
                                         mmap_mode)
            frame.set_axis(columns, axis=1, inplace=True)  # type: ignore

        return frame

    @classmethod
    def write(cls, frame: Frame, path: str) -> None:
        """
        Write a Pandas data frame of the concrete type to a file.

        If necessary, creates additional file(s) with an ``.index`` and/or ``.columns`` suffix to
        preserve the index and/or column labels.
        """
        cls.am(frame)

        np.BaseArray._write(frame.values, path)  # pylint: disable=protected-access

        if not frame.index.equals(RangeIndex(len(frame.index))):
            np.BaseArray._write(frame.index.values,  # pylint: disable=protected-access
                                path + '.index')

        if not frame.columns.equals(RangeIndex(len(frame.columns))):
            np.BaseArray._write(frame.columns.values,  # pylint: disable=protected-access
                                path + '.columns')

    @classmethod
    def am(cls: Type[F], data: Frame) -> F:  # pylint: disable=invalid-name
        """
        Declare a data frame as being of this type.
        """
        BaseFrame._am_frame(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:
            raise ValueError('unexpected data type: %s instead of: %s'
                             % (array.dtype, cls.dtype))
        return data  # type: ignore

    @classmethod
    def be(cls: Type[F],  # pylint: disable=invalid-name
           data: Union[Frame, np.ndarray, List[List[Any]]]) -> F:
        """
        Convert an array to this type.
        """
        if isinstance(data, list):
            data = np.array(data, dtype=cls.dtype)
        if isinstance(data, np.ndarray):
            data = Frame(data)

        BaseFrame._am_frame(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:
            data = data.astype(cls.dtype)
        return data  # type: ignore

    @staticmethod
    def _am_frame(data: Frame) -> None:
        if not isinstance(data, Frame):
            raise ValueError('unexpected type: %s.%s instead of: %s.%s'
                             % (data.__class__.__module__, data.__class__.__qualname__,
                                Frame.__module__, Frame.__qualname__))
        array = data.values
        np.BaseArray._am_shape(array, 2)  # pylint: disable=protected-access

    @classmethod
    def zeros(cls: Type[F], *, index: Union[int, Sized], columns: Union[int, Sized]) -> F:
        """
        Return a frame full of zeros.
        """
        return cls._make(np.zeros, index=index, columns=columns)

    @classmethod
    def empty(cls: Type[F], *,  # pylint: disable=arguments-differ
              index: Union[int, Sized], columns: Union[int, Sized]) -> F:
        """
        Return an uninitialized frame
        """
        return cls._make(np.empty, index=index, columns=columns)

    @classmethod
    def filled(cls: Type[F], value: Any, *,
               index: Union[int, Sized], columns: Union[int, Sized]) -> F:
        """
        Return a frame full of some value.
        """
        frame = cls.empty(index=index, columns=columns)
        frame.values.fill(value)
        return frame

    @classmethod
    def shared_memory_zeros(cls: Type[F], *,
                            index: Union[int, Sized], columns: Union[int, Sized]) -> F:
        """
        Create a shared memory frame, initialized to zeros.
        """
        def _maker(shape: Tuple[int, int]) -> np.ndarray:
            return np.ARRAY_OF_DTYPE[cls.dtype].shared_memory_zeros(shape)
        return cls._make(np.MATRIX_OF_DTYPE[cls.dtype].shared_memory_zeros,
                         index=index, columns=columns)

    @classmethod
    def _make(cls: Type[F], maker: Callable, *,
              index: Union[int, Sized], columns: Union[int, Sized]) -> F:
        if isinstance(index, int):
            rows_count = index
            index = None  # type: ignore
        else:
            rows_count = len(index)

        if isinstance(columns, int):
            columns_count = columns
            columns = None  # type: ignore
        else:
            columns_count = len(columns)

        return cls.am(Frame(maker((rows_count, columns_count), dtype=cls.dtype),
                            index=index, columns=columns))


class SeriesStr(BaseSeries):
    """
    A data series of Unicode strings.
    """
    dtype = 'O'


class SeriesBool(BaseSeries):
    """
    A data series of booleans.
    """
    dtype = 'bool'


class FrameBool(BaseFrame):
    """
    A data frame of booleans.
    """
    dtype = 'bool'


class SeriesInt8(BaseSeries):
    """
    A data series of 8-bit integers.
    """
    dtype = 'int8'


class FrameInt8(BaseFrame):
    """
    A data frame of 8-bit integers.
    """
    dtype = 'int8'


class SeriesInt16(BaseSeries):
    """
    A data series of 16-bit integers.
    """
    dtype = 'int16'


class FrameInt16(BaseFrame):
    """
    A data frame of 16-bit integers.
    """
    dtype = 'int16'


class SeriesInt32(BaseSeries):
    """
    A data series of 32-bit integers.
    """
    dtype = 'int32'


class FrameInt32(BaseFrame):
    """
    A data frame of 32-bit integers.
    """
    dtype = 'int32'


class SeriesInt64(BaseSeries):
    """
    A data series of 64-bit integers.
    """
    dtype = 'int64'


class FrameInt64(BaseFrame):
    """
    A data frame of 64-bit integers.
    """
    dtype = 'int64'


class SeriesFloat32(BaseSeries):
    """
    A data series of 32-bit floating-point numbers.
    """
    dtype = 'float32'


class FrameFloat32(BaseFrame):
    """
    A data frame of 32-bit floating-point numbers.
    """
    dtype = 'float32'


class SeriesFloat64(BaseSeries):
    """
    A data series of 64-bit floating-point numbers.
    """
    dtype = 'float64'


class FrameFloat64(BaseFrame):
    """
    A data frame of 64-bit floating-point numbers.
    """
    dtype = 'float64'


#: The phantom type for a data series by its type name.
SERIES_OF_DTYPE = dict(  #
    str=SeriesStr,
    bool=SeriesBool,
    int8=SeriesInt8,
    int16=SeriesInt16,
    int32=SeriesInt32,
    int64=SeriesInt64,
    float32=SeriesFloat32,
    float64=SeriesFloat64,
)

#: The phantom type for a data frame by its type name.
FRAME_OF_DTYPE = dict(  #
    bool=FrameBool,
    int8=FrameInt8,
    int16=FrameInt16,
    int32=FrameInt32,
    int64=FrameInt64,
    float32=FrameFloat32,
    float64=FrameFloat64,
)
