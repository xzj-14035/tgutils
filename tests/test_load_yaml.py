"""
Test loading data from YAML files.
"""

from tgutils.load_yaml import load_dictionary
from tgutils.tests import TestWithFiles
from tgutils.tests import write_file

# pylint: disable=missing-docstring,too-many-public-methods,no-self-use


class TestLoadYaml(TestWithFiles):

    def test_load_missing_yaml(self) -> None:
        self.assertRaisesRegex(FileNotFoundError,
                               "No such file.*: 'missing.yaml'",
                               load_dictionary, 'missing.yaml')

    def test_load_sequence(self) -> None:
        write_file('sequence.yaml', '[]')
        self.assertRaisesRegex(RuntimeError,
                               'file: sequence.yaml',
                               load_dictionary, 'sequence.yaml')

    def test_load_non_string_key(self) -> None:
        write_file('non_string_key.yaml', '1: a')
        self.assertRaisesRegex(RuntimeError,
                               "key is a: <class 'int'>",
                               load_dictionary, 'non_string_key.yaml')

    def test_load_non_string_value(self) -> None:
        write_file('non_string_value.yaml', 'a: 1')
        self.assertRaisesRegex(RuntimeError,
                               "key: a .* type: <class 'int'>",
                               load_dictionary, 'non_string_value.yaml', value_type=str)

    def test_load_unlisted_key(self) -> None:
        write_file('unlisted_key.yaml', 'a: A\nb: B\nc: C\n')

        self.assertEqual(load_dictionary('unlisted_key.yaml',
                                         required_keys={'a': str, 'b': str}),
                         {'a': 'A', 'b': 'B', 'c': 'C'})

        self.assertRaisesRegex(RuntimeError,
                               'Unexpected key: c',
                               load_dictionary, 'unlisted_key.yaml',
                               allowed_keys={'a': str}, required_keys={'b': str})

        self.assertRaisesRegex(RuntimeError,
                               'Unexpected key: c',
                               load_dictionary, 'unlisted_key.yaml',
                               allowed_keys={'a': str, 'b': str})

        self.assertRaisesRegex(RuntimeError,
                               'Unexpected key: c',
                               load_dictionary, 'unlisted_key.yaml',
                               allowed_keys=[],
                               required_keys={'a': str, 'b': str})

        self.assertRaisesRegex(RuntimeError,
                               'Missing key: d',
                               load_dictionary, 'unlisted_key.yaml',
                               required_keys={'d': str})
