"""
Numpy utilities.

Import this as ``np`` instead of importing the ``numpy`` module. It exports the same symbols, with
the addition of strongly-typed phantom classes for tracking the exact dimensions and type of each
variable using ``mypy``. It also provides some additional utilities (I/O).
"""

from numpy import *  # pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
from typing import Optional
from typing import Type
from typing import TypeVar

import os

_A = TypeVar('_A', bound=ndarray)


class BaseArray(ndarray):
    """
    Base class for all Numpy array and matrix phantom types.
    """

    @staticmethod
    def exists(path: str) -> bool:
        """
        Whether there exists a disk file with the specified path to load an array from.

        This checks for either a ``.txt`` or a ``.npy`` suffix to allow for loading either
        an array of strings or an array or matrix of numeric values.
        """
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        return os.path.exists(path + '.npy') or os.path.exists(path + '.txt')

    @classmethod
    def read(cls: Type[_A], path: str, mmap_mode: Optional[str] = None) -> _A:
        """
        Read a Numpy array of the concrete type from the disk.

        If a disk file with a ``.txt`` suffix exists, this will read an array of strings. Otherwise,
        a file with a ``.npy`` suffix must exist, and this will memory map the array or matrix of
        values contained in it.
        """
        return cls.am(BaseArray._read(path, mmap_mode))  # type: ignore

    @staticmethod
    def _read(path: str, mmap_mode: Optional[str] = None) -> ndarray:
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        text_path = path + '.txt'
        if os.path.exists(text_path):
            with open(text_path, 'r') as file:
                strings = file.read().split('\n')[:-1]
                values = array(strings, dtype='O')
        else:
            values = load(path + '.npy', mmap_mode)

        return values

    @classmethod
    def write(cls, data: ndarray, path: str) -> None:
        """
        Write a Numpy array of the concrete type to the disk.

        If writing an array of strings, this will create a file with a ``.txt`` suffix containing
        one string value per line. Otherwise, the data may be an array or a matrix of numeric
        values, which will be written to a file with a ``.npy`` format allowing for memory mapped
        access.
        """
        cls.am(data)
        BaseArray._write(data, path)

    @staticmethod
    def _write(data: ndarray, path: str) -> None:
        assert not path.endswith('.npy')
        assert not path.endswith('.txt')

        if data.dtype == 'O':
            BaseArray._am_shape(data, 1)
            with open(path + '.txt', 'w') as file:
                file.write('\n'.join(data))
                file.write('\n')
        else:
            save(path + '.npy', data)

    @classmethod
    def am(cls: Type[_A], data: ndarray) -> _A:  # pylint: disable=invalid-name
        """
        Declare an array as being of this type.
        """
        BaseArray._am_shape(data, cls.dimensions)  # type: ignore
        if cls.dtype not in [data.dtype.name, data.dtype.kind]:
            raise ValueError('unexpected data type: %s instead of: %s'
                             % (data.dtype, cls.dtype))
        return data  # type: ignore

    @classmethod
    def be(cls: Type[_A], data: ndarray) -> _A:  # pylint: disable=invalid-name
        """
        Convert an array to this type.
        """
        BaseArray._am_shape(data, cls.dimensions)  # type: ignore
        if cls.dtype not in [data.dtype.name, data.dtype.kind]:
            data = data.astype(cls.dtype)
        return data  # type: ignore

    @staticmethod
    def _am_shape(data: ndarray, expected_dimensions: int) -> None:
        if not isinstance(data, ndarray):
            raise ValueError('unexpected type: %s.%s instead of: %s.%s'
                             % (data.__class__.__module__, data.__class__.__qualname__,
                                ndarray.__module__, ndarray.__qualname__))

        if len(data.shape) != expected_dimensions:
            raise ValueError('unexpected dimensions: %s instead of: %s'
                             % (len(data.shape), expected_dimensions))


class ArrayStr(BaseArray):
    """
    An array of Unicode strings.
    """
    dimensions = 1
    dtype = 'O'


class MatrixStr(BaseArray):
    """
    A matrix of Unicode strings.
    """
    dimensions = 2
    dtype = 'O'


class ArrayBool(BaseArray):
    """
    An array of booleans.
    """
    dimensions = 1
    dtype = 'bool'


class MatrixBool(BaseArray):
    """
    A matrix of booleans.
    """
    dimensions = 2
    dtype = 'bool'


class ArrayInt8(BaseArray):
    """
    An array of 8-bit integers.
    """
    dimensions = 1
    dtype = 'int8'


class MatrixInt8(BaseArray):
    """
    A matrix of 8-bit integers.
    """
    dimensions = 2
    dtype = 'int8'


class ArrayInt16(BaseArray):
    """
    An array of 16-bit integers.
    """
    dimensions = 1
    dtype = 'int16'


class MatrixInt16(BaseArray):
    """
    A matrix of 16-bit integers.
    """
    dimensions = 2
    dtype = 'int16'


class ArrayInt32(BaseArray):
    """
    An array of 32-bit integers.
    """
    dimensions = 1
    dtype = 'int32'


class MatrixInt32(BaseArray):
    """
    A matrix of 32-bit integers.
    """
    dimensions = 2
    dtype = 'int32'


class ArrayInt64(BaseArray):
    """
    An array of 64-bit integers.
    """
    dimensions = 1
    dtype = 'int64'


class MatrixInt64(BaseArray):
    """
    A matrix of 64-bit integers.
    """
    dimensions = 2
    dtype = 'int64'


class ArrayFloat32(BaseArray):
    """
    An array of 32-bit floating point numbers.
    """
    dimensions = 1
    dtype = 'float32'


class MatrixFloat32(BaseArray):
    """
    A matrix of 32-bit floating point numbers.
    """
    dimensions = 2
    dtype = 'float32'


class ArrayFloat64(BaseArray):
    """
    An array of 64-bit floating point numbers.
    """
    dimensions = 1
    dtype = 'float64'


class MatrixFloat64(BaseArray):
    """
    A matrix of 64-bit floating point numbers.
    """
    dimensions = 2
    dtype = 'float64'
