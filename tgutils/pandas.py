"""
Pandas utilities.

Import this as ``pd`` instead of directly importing the ``pandas`` module. It exports the same
symbols, with the addition of strongly-typed phantom classes for tracking the exact dimensions and
type of each variable using ``mypy``. It also provides some additional utilities (I/O).
"""

from .numpy import BaseArray
from pandas import *  # pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
from typing import Optional
from typing import Type
from typing import TypeVar

#: Short name for ``DataFrame``.
Frame = DataFrame

_S = TypeVar('_S', bound='BaseSeries')
_F = TypeVar('_F', bound='BaseFrame')


class BaseSeries(Series):
    """
    Base class for all Numpy data series phantom types.
    """

    @classmethod
    def read(cls: Type[_S], path: str, mmap_mode: Optional[str] = None) -> _S:
        """
        Read a Pandas data series of the concrete type from the disk.

        If an additional file with an ``.index`` suffix exists, it is loaded into the index labels.
        """
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        array = BaseArray._read(path, mmap_mode)  # pylint: disable=protected-access
        series = cls.am(Series(array))

        index_path = path + '.index'
        if BaseArray.exists(index_path):
            index = BaseArray._read(index_path, mmap_mode)  # pylint: disable=protected-access
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

        BaseArray._write(series.values, path)  # pylint: disable=protected-access

        if not series.index.equals(RangeIndex(len(series.index))):
            BaseArray._write(series.index.values,  # pylint: disable=protected-access
                             path + '.index')

    @classmethod
    def am(cls: Type[_S], data: Series) -> _S:  # pylint: disable=invalid-name
        """
        Declare a data series as being of this type.
        """
        BaseSeries._am_series(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:  # type: ignore
            raise ValueError('unexpected data type: %s instead of: %s'
                             % (array.dtype, cls.dtype))  # type: ignore
        return data  # type: ignore

    @classmethod
    def be(cls: Type[_S], data: Series) -> _S:  # pylint: disable=invalid-name
        """
        Convert an array to this type.
        """
        BaseSeries._am_series(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:  # type: ignore
            data = data.astype(cls.dtype)  # type: ignore
        return data  # type: ignore

    @staticmethod
    def _am_series(data: Series) -> None:
        if not isinstance(data, Series):
            raise ValueError('unexpected type: %s.%s instead of: %s.%s'
                             % (data.__class__.__module__, data.__class__.__qualname__,
                                Series.__module__, Series.__qualname__))
        array = data.values
        BaseArray._am_shape(array, 1)  # pylint: disable=protected-access


class BaseFrame(Frame):
    """
    Base class for all Numpy data series phantom types.
    """

    @classmethod
    def read(cls: Type[_F], path: str, mmap_mode: Optional[str] = None) -> _F:
        """
        Read a Pandas data frame of the concrete type from the disk.

        If additional file(s) with an ``.index`` and/or ``.columns`` suffix exist, they are loaded
        into the index and/or column labels.
        """
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        array = BaseArray._read(path, mmap_mode)  # pylint: disable=protected-access
        frame = cls.am(Frame(array))

        index_path = path + '.index'
        if BaseArray.exists(index_path):
            index = BaseArray._read(index_path, mmap_mode)  # pylint: disable=protected-access
            frame.set_axis(index, axis=0, inplace=True)  # type: ignore

        columns_path = path + '.columns'
        if BaseArray.exists(columns_path):
            columns = BaseArray._read(columns_path, mmap_mode)  # pylint: disable=protected-access
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

        BaseArray._write(frame.values, path)  # pylint: disable=protected-access

        if not frame.index.equals(RangeIndex(len(frame.index))):
            BaseArray._write(frame.index.values,  # pylint: disable=protected-access
                             path + '.index')

        if not frame.columns.equals(RangeIndex(len(frame.columns))):
            BaseArray._write(frame.columns.values,  # pylint: disable=protected-access
                             path + '.columns')

    @classmethod
    def am(cls: Type[_F], data: Frame) -> _F:  # pylint: disable=invalid-name
        """
        Declare a data frame as being of this type.
        """
        BaseFrame._am_frame(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:  # type: ignore
            raise ValueError('unexpected data type: %s instead of: %s'
                             % (array.dtype, cls.dtype))  # type: ignore
        return data  # type: ignore

    @classmethod
    def be(cls: Type[_F], data: Frame) -> _F:  # pylint: disable=invalid-name
        """
        Convert an array to this type.
        """
        BaseFrame._am_frame(data)
        array = data.values
        if cls.dtype not in [array.dtype.name, array.dtype.kind]:  # type: ignore
            data = data.astype(cls.dtype)  # type: ignore
        return data  # type: ignore

    @staticmethod
    def _am_frame(data: Frame) -> None:
        if not isinstance(data, Frame):
            raise ValueError('unexpected type: %s.%s instead of: %s.%s'
                             % (data.__class__.__module__, data.__class__.__qualname__,
                                Frame.__module__, Frame.__qualname__))
        array = data.values
        BaseArray._am_shape(array, 2)  # pylint: disable=protected-access


class SeriesStr(BaseSeries):
    """
    A data series of Unicode strings.
    """
    dtype = 'O'


class FrameStr(BaseFrame):
    """
    A data frame of Unicode strings.
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
