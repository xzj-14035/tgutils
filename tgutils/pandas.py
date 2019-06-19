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

import feather  # type: ignore
import os

#: Short name for ``DataFrame``.
Frame = DataFrame

_S = TypeVar('_S', bound=Series)
_F = TypeVar('_F', bound=Frame)


class BaseSeries(Series):
    """
    Base class for all Numpy data series phantom types.
    """

    @classmethod
    def read(cls: Type[_S], path: str) -> _S:
        """
        Read a Pandas data series of the concrete type from the disk.
        """
        frame = BaseFrame._read(path)  # pylint: disable=protected-access
        assert len(frame.columns) == 1
        series = frame.iloc[:, 0]
        return cls.am(series)  # type: ignore

    @classmethod
    def write(cls, series: Series, path: str, *, name: Optional[str] = None) -> None:
        """
        Write a Pandas series of the concrete type to a file.

        This wraps the series as a column in a data frame, then writes it as if it were a normal
        data frame.
        """
        frame = Frame({name or series.name or 'series': cls.am(series)})
        BaseFrame._write(frame, path)  # pylint: disable=protected-access

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
    def read(cls: Type[_F], path: str) -> _F:
        """
        Read a Pandas data frame of the concrete type from the disk.

        If additional file(s) with an ``.index`` and/or ``.columns`` suffix exist, they are loaded
        into the index and/or column labels.
        """
        return cls.am(BaseFrame._read(path))  # type: ignore  # pylint: disable=protected-access

    @classmethod
    def write(cls, frame: Frame, path: str) -> None:
        """
        Write a Pandas data frame of the concrete type to a file.

        If necessary, creates additional file(s) with an ``.index`` and/or ``.columns`` suffix to
        preserve the index and/or column labels.
        """
        BaseFrame._write(cls.am(frame), path)  # pylint: disable=protected-access

    @staticmethod
    def _write(frame: Frame, path: str) -> None:
        if not path.endswith('.feather'):
            path += '.feather'
        with open(path, 'wb') as file:
            feather.write_dataframe(frame, file)

        if not frame.index.equals(RangeIndex(len(frame.index))):
            with open(path[:-8] + '.index.feather', 'wb') as file:
                feather.write_dataframe(Frame(dict(index=frame.index)), file)

        if not isinstance(frame.columns[0], str):
            with open(path[:-8] + '.columns.feather', 'wb') as file:
                feather.write_dataframe(Frame(dict(columns=frame.columns)), file)

    @staticmethod
    def _read(path: str) -> Frame:
        if not path.endswith('.feather'):
            path += '.feather'
        with open(path, 'rb') as file:
            frame = feather.read_dataframe(file)

        index_path = path[:-8] + '.index.feather'
        if os.path.exists(index_path):
            with open(index_path, 'rb') as file:
                index = feather.read_dataframe(file)
            frame.set_axis(index.loc[:, 'index'].values, axis=0, inplace=True)

        columns_path = path[:-8] + '.columns.feather'
        if os.path.exists(columns_path):
            with open(columns_path, 'rb') as file:
                columns = feather.read_dataframe(file)
            frame.set_axis(columns.loc[:, 'columns'].values, axis=1, inplace=True)

        return frame

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
