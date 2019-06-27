"""
Test the Pandas utilities.
"""

from parameterized import parameterized  # type: ignore
from tests import TestWithFiles
from typing import Any
from typing import List
from typing import Type

import tgutils.numpy as np
import tgutils.pandas as pd

# pylint: disable=missing-docstring,too-many-public-methods,no-self-use
# pylint: disable=blacklisted-name,too-few-public-methods


class TestPandas(TestWithFiles):

    def check_write_read_series(self, cls: Type[pd.S], data: List[Any],
                                *, index: Any = None) -> None:
        if index is None:
            written_series = cls.be(data)
        else:
            written_series = cls.be(pd.Series(data, index=index))
        cls.write(written_series, 'disk_file')
        read_series = cls.read('disk_file')
        self.assertEqual(read_series.values.dtype, written_series.values.dtype)

        written_list = \
            [None if not isinstance(value, str) and np.isnan(value) else value
             for value in written_series.values]
        read_list = \
            [None if not isinstance(value, str) and np.isnan(value) else value
             for value in read_series.values]
        self.assertEqual(read_list, written_list)

        written_index = list(written_series.index)
        read_index = list(read_series.index)
        self.assertEqual(read_index, written_index)

    def check_write_read_frame(self, cls: Type[pd.F], data: List[List[Any]],
                               *, index: Any = None, columns: Any = None) -> None:
        if index is None and columns is None:
            written_frame = cls.be(data)
        else:
            written_frame = cls.be(pd.Frame(data, index=index, columns=columns))
        cls.write(written_frame, 'disk_file')
        read_frame = cls.read('disk_file')
        self.assertEqual(read_frame.values.dtype, written_frame.values.dtype)

        written_list = \
            [[None if not isinstance(value, str) and np.isnan(value) else value
              for value in row]
             for row in written_frame.values]
        read_list = \
            [[None if not isinstance(value, str) and np.isnan(value) else value
              for value in row]
             for row in read_frame.values]
        self.assertEqual(read_list, written_list)

        written_index = list(written_frame.index)
        read_index = list(read_frame.index)
        self.assertEqual(read_index, written_index)

        written_columns = list(written_frame.columns)
        read_columns = list(read_frame.columns)
        self.assertEqual(read_columns, written_columns)

    def test_series_str(self) -> None:
        self.check_write_read_series(pd.SeriesStr, ['foo', 'bar'])

    def test_series_str_with_index(self) -> None:
        self.check_write_read_series(pd.SeriesStr, ['foo', 'bar'], index=['a', 'b'])

    def test_series_bool(self) -> None:
        self.check_write_read_series(pd.SeriesBool, [True, False])

    def test_frame_bool(self) -> None:
        self.check_write_read_frame(pd.FrameBool, [[True, False, True], [False, True, False]])

    def test_frame_bool_with_index(self) -> None:
        self.check_write_read_frame(pd.FrameBool, [[True, False, True], [False, True, False]],
                                    index=['x', 'y'])

    def test_frame_bool_with_columns(self) -> None:
        self.check_write_read_frame(pd.FrameBool, [[True, False, True], [False, True, False]],
                                    columns=['a', 'b', 'c'])

    def test_frame_bool_with_both(self) -> None:
        self.check_write_read_frame(pd.FrameBool, [[True, False, True], [False, True, False]],
                                    index=['x', 'y'], columns=['a', 'b', 'c'])

    @parameterized.expand([(pd.SeriesInt8,),
                           (pd.SeriesInt16,),
                           (pd.SeriesInt32,),
                           (pd.SeriesInt64,)])
    def test_series_int(self, cls: Type[pd.S]) -> None:
        self.check_write_read_series(cls, [0, 1])

    @parameterized.expand([(pd.FrameInt8,), (pd.FrameInt16,), (pd.FrameInt32,), (pd.FrameInt64,)])
    def test_frame_int(self, cls: Type[pd.F]) -> None:
        self.check_write_read_frame(cls, [[0, 1, 2], [3, 4, 5]])

    @parameterized.expand([(pd.SeriesFloat32,), (pd.SeriesFloat64,)])
    def test_series_float(self, cls: Type[pd.S]) -> None:
        self.check_write_read_series(cls, [0.0, None])

    @parameterized.expand([(pd.FrameFloat32,), (pd.FrameFloat64,)])
    def test_frame_float(self, cls: Type[pd.F]) -> None:
        self.check_write_read_frame(cls, [[0.0, None, 2.0], [3.0, np.nan, 5.0]])

    def test_bad_series_data_type(self) -> None:
        series_float32 = pd.SeriesFloat32.am(pd.Series([0, 1], dtype='float32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected data type: float32 instead of: int32',
                               pd.SeriesInt32.am, series_float32)

    def test_bad_frame_data_type(self) -> None:
        frame_float32 = pd.FrameFloat32.am(pd.Frame([[0, 1], [2, 3]], dtype='float32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected data type: float32 instead of: int32',
                               pd.FrameInt32.am, frame_float32)

    def test_bad_series_class(self) -> None:
        array_int32 = np.ArrayInt32.am(np.array([0, 1], dtype='int32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected type: numpy.ndarray instead of: pandas.*Series',
                               pd.SeriesInt32.am, array_int32)

    def test_bad_frame_class(self) -> None:
        matrix_int32 = np.MatrixInt32.am(np.array([[0, 1], [2, 3]], dtype='int32'))
        self.assertRaisesRegex(ValueError,
                               'unexpected type: numpy.ndarray instead of: pandas.*DataFrame',
                               pd.FrameInt32.am, matrix_int32)

    def test_cast_series(self) -> None:
        series_int32 = pd.SeriesInt32.am(pd.Series([0, 1], dtype='int32'))

        cast_int32 = pd.SeriesInt32.be(series_int32)
        self.assertIs(cast_int32, series_int32)

        cast_int64 = pd.SeriesInt64.be(series_int32)
        pd.SeriesInt64.am(cast_int64)
        self.assertEqual(list(cast_int64), list(series_int32))

    def test_cast_frame(self) -> None:
        frame_int32 = pd.FrameInt32.am(pd.Frame([[0, 1], [2, 3]], dtype='int32'))

        cast_int32 = pd.FrameInt32.be(frame_int32)
        self.assertIs(cast_int32, frame_int32)

        cast_int64 = pd.FrameInt64.be(frame_int32)
        pd.FrameInt64.am(cast_int64)
        self.assertEqual(list(cast_int64), list(frame_int32))
