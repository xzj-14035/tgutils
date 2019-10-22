"""
Numpy utilities.

Import this as ``np`` instead of importing the ``numpy`` module. It exports the same symbols, with
the addition of strongly-typed phantom classes for tracking the exact dimensions and type of each
variable using ``mypy``. It also provides some additional utilities (I/O).
"""

from numpy import *  # pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
from typing import Any
from typing import Collection
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import ctypes
import multiprocessing
import os

# pylint: disable=redefined-outer-name


#: Type variable for arrays.
A = TypeVar('A', bound='BaseArray')  # pylint: disable=invalid-name

_C_TYPES = dict(float32=ctypes.c_float, int32=ctypes.c_int32, float64=ctypes.c_double)


class BaseArray(ndarray):
    """
    Base class for all Numpy array and matrix phantom types.
    """

    #: The expected dimensions of an array of the (derived) class.
    dimensions: int

    #: The expected data type of an array of the (derived) class.
    dtype: str

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
    def read(cls: Type[A], path: str, mmap_mode: Optional[str] = None) -> A:
        """
        Read a Numpy array of the concrete type from the disk.

        If a disk file with a ``.txt`` suffix exists, this will read an array of strings. Otherwise,
        a file with a ``.npy`` suffix must exist, and this will memory map the array or matrix of
        values contained in it.
        """
        return cls.am(BaseArray._read(path, mmap_mode))

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

    @staticmethod
    def read_array(path: str, mmap_mode: Optional[str] = None) -> ndarray:
        """
        Read a 1D array of any type from the disk.
        """
        array = BaseArray._read(path, mmap_mode)
        BaseArray.am_array(array)
        return array

    @staticmethod
    def read_matrix(path: str, mmap_mode: Optional[str] = None) -> ndarray:
        """
        Read a 2D array of any type from the disk.
        """
        array = BaseArray._read(path, mmap_mode)
        BaseArray.am_matrix(array)
        return array

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
            BaseArray.am_array(data)
            with open(path + '.txt', 'w') as file:
                file.write('\n'.join(data))
                file.write('\n')
        else:
            save(path + '.npy', data)

    @classmethod
    def am(cls: Type[A], data: ndarray) -> A:  # pylint: disable=invalid-name
        """
        Declare an array as being of this type.
        """
        BaseArray.has_dimensions(data, cls.dimensions)
        if cls.dtype not in [data.dtype.name, data.dtype.kind]:
            raise ValueError('unexpected data type: %s instead of: %s'
                             % (data.dtype, cls.dtype))
        return data  # type: ignore

    @classmethod
    def be(cls: Type[A], data: Collection) -> A:  # pylint: disable=invalid-name
        """
        Convert an array to this type.
        """
        if not isinstance(data, ndarray):
            data = array(data, dtype=cls.dtype)

        BaseArray.has_dimensions(data, cls.dimensions)
        if cls.dtype not in [data.dtype.name, data.dtype.kind]:
            data = data.astype(cls.dtype)

        return data  # type: ignore

    @staticmethod
    def am_array(data: ndarray) -> None:
        """
        Declare an array as being 1-dimensional.
        """
        BaseArray.has_dimensions(data, 1)

    @staticmethod
    def am_matrix(data: ndarray) -> None:
        """
        Declare an array as being 2-dimensional.
        """
        BaseArray.has_dimensions(data, 2)

    @staticmethod
    def has_dimensions(data: ndarray, expected_dimensions: int) -> None:
        """
        Declare an array has the specified number of dimensions.
        """
        if not isinstance(data, ndarray):
            raise ValueError('unexpected type: %s.%s instead of: %s.%s'
                             % (data.__class__.__module__, data.__class__.__qualname__,
                                ndarray.__module__, ndarray.__qualname__))

        if len(data.shape) != expected_dimensions:
            raise ValueError('unexpected dimensions: %s instead of: %s'
                             % (len(data.shape), expected_dimensions))

    @classmethod
    def zeros(cls: Type[A], shape: Union[int, Tuple[int, ...]]) -> A:
        """
        Return an array full of zeros.
        """
        if isinstance(shape, tuple):
            assert len(shape) == 2
        return cls.am(zeros(shape, dtype=cls.dtype))

    @classmethod
    def uninitialized(cls: Type[A], shape: Union[int, Tuple[int, ...]]) -> A:
        """
        Return an uninitialized array.
        """
        if isinstance(shape, tuple):
            assert len(shape) == 2
        return cls.am(empty(shape, dtype=cls.dtype))

    @classmethod
    def filled(cls: Type[A], value: Any, shape: Union[int, Tuple[int, ...]]) -> A:
        """
        Return an array full of some value.
        """
        array = cls.uninitialized(shape)
        array.fill(value)
        return array

    @classmethod
    def shared_memory_zeros(cls: Type[A], shape: Union[int, Tuple[int, ...]]) -> A:
        """
        Create a shared memory array, initialized to zeros.
        """
        if isinstance(shape, int):
            size = shape
            shape = (size,)
        else:
            assert len(shape) == 2
            size = int(shape[0] * shape[1])

        c_type = _C_TYPES[cls.dtype]

        shared_buffer = multiprocessing.Array(c_type, size)  # type: ignore
        shared_array = frombuffer(shared_buffer.get_obj(), dtype=cls.dtype)
        return cls.am(reshape(shared_array, shape, order='F'))


class ArrayStr(BaseArray):
    """
    An array of Unicode strings.
    """
    dimensions = 1
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


#: The phantom type for an array by its data type name.
ARRAY_OF_DTYPE = dict(  #
    str=ArrayStr,
    bool=ArrayBool,
    int8=ArrayInt8,
    int16=ArrayInt16,
    int32=ArrayInt32,
    int64=ArrayInt64,
    float32=ArrayFloat32,
    float64=ArrayFloat64,
)

#: The phantom type for a matrix by its data type name.
MATRIX_OF_DTYPE = dict(  #
    bool=MatrixBool,
    int8=MatrixInt8,
    int16=MatrixInt16,
    int32=MatrixInt32,
    int64=MatrixInt64,
    float32=MatrixFloat32,
    float64=MatrixFloat64,
)
