"""
Test the Numpy utilities.
"""

from parameterized import parameterized  # type: ignore
from tgutils.tests import TestWithFiles
from typing import Any
from typing import List
from typing import Type

import tgutils.numpy as np
import tgutils.pandas as pd

# pylint: disable=missing-docstring,too-many-public-methods,no-self-use
# pylint: disable=blacklisted-name,too-few-public-methods


class TestNumpy(TestWithFiles):

    def check_write_read_array(self, cls: Type[np.A], data: List[Any]) -> None:
        written_array = cls.be(data)
        cls.write(written_array, 'disk_file')
        read_array = cls.read('disk_file')
        self.assertEqual(read_array.dtype, written_array.dtype)

        written_list = \
            [None if not isinstance(value, str) and np.isnan(value) else value
             for value in written_array]
        read_list = \
            [None if not isinstance(value, str) and np.isnan(value) else value
             for value in read_array]
        self.assertEqual(read_list, written_list)

        zeros = cls.am(cls.zeros(len(data)))
        self.assertEqual(list(zeros), [0] * len(data))

        filled = cls.am(cls.filled(1, len(data)))
        self.assertEqual(list(filled), [1] * len(data))

    def check_write_read_matrix(self, cls: Type[np.A], data: List[List[Any]]) -> None:
        written_matrix = cls.be(data)
        cls.write(written_matrix, 'disk_file')
        read_matrix = cls.read('disk_file')
        self.assertEqual(read_matrix.dtype, written_matrix.dtype)

        written_list = \
            [[None if not isinstance(value, str) and np.isnan(value) else value
              for value in row]
             for row in written_matrix]
        read_list = \
            [[None if not isinstance(value, str) and np.isnan(value) else value
              for value in row]
             for row in read_matrix]
        self.assertEqual(read_list, written_list)

        zeros: np.ndarray = cls.am(cls.zeros(read_matrix.shape))
        self.assertEqual([list(array) for array in list(zeros)],
                         [[0] * read_matrix.shape[1]] * read_matrix.shape[0])

        filled: np.ndarray = cls.am(cls.filled(1, read_matrix.shape))
        self.assertEqual([list(array) for array in list(filled)],
                         [[1] * read_matrix.shape[1]] * read_matrix.shape[0])

    def test_array_bool(self) -> None:
        self.check_write_read_array(np.ArrayBool, [True, False])

    def test_matrix_bool(self) -> None:
        self.check_write_read_matrix(np.MatrixBool, [[True, False, True], [False, True, False]])

    def test_array_str(self) -> None:
        self.check_write_read_array(np.ArrayStr, ['foo', 'bar'])

    @parameterized.expand([(np.ArrayInt8,), (np.ArrayInt16,), (np.ArrayInt32,), (np.ArrayInt64,)])
    def test_array_int(self, cls: Type[np.A]) -> None:
        self.check_write_read_array(cls, [0, 1])

    @parameterized.expand([(np.MatrixInt8,),
                           (np.MatrixInt16,),
                           (np.MatrixInt32,),
                           (np.MatrixInt64,)])
    def test_matrix_int(self, cls: Type[np.A]) -> None:
        self.check_write_read_matrix(cls, [[0, 1, 2], [3, 4, 5]])

    @parameterized.expand([(np.ArrayFloat32,), (np.ArrayFloat64,)])
    def test_array_float(self, cls: Type[np.A]) -> None:
        self.check_write_read_array(cls, [0.0, None, 2.0])

    @parameterized.expand([(np.MatrixFloat32,), (np.MatrixFloat64,)])
    def test_matrix_float(self, cls: Type[np.A]) -> None:
        self.check_write_read_matrix(cls, [[0.0, None, 2.0], [3.0, np.nan, 5.0]])

    def test_bad_data_type(self) -> None:
        array_float32 = np.ArrayFloat32.am(np.array([0, 1], dtype='float32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected data type: float32 instead of: int32',
                               np.ArrayInt32.am, array_float32)

    def test_bad_dimensions(self) -> None:
        matrix_int32 = np.MatrixInt32.am(np.array([[0, 1], [2, 3]], dtype='int32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected dimensions: 2 instead of: 1',
                               np.ArrayInt32.am, matrix_int32)

    def test_bad_class(self) -> None:
        series_int32 = pd.SeriesInt32.am(pd.Series([0, 1], dtype='int32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected type: pandas.*Series instead of: numpy.ndarray',
                               np.ArrayInt32.am, series_int32)

    def test_cast(self) -> None:
        array_int32 = np.ArrayInt32.am(np.array([0, 1], dtype='int32'))

        cast_int32 = np.ArrayInt32.be(array_int32)
        self.assertIs(cast_int32, array_int32)

        cast_int64 = np.ArrayInt64.be(array_int32)
        np.ArrayInt64.am(cast_int64)
        self.assertEqual(list(cast_int64), list(array_int32))
