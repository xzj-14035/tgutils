"""
Common utilities for tests.
"""

from textwrap import dedent
from unittest import TestCase

import os
import shutil
import sys
import tempfile

# pylint: disable=missing-docstring


def write_file(path: str, content: str = '') -> None:
    with open(path, 'w') as file:
        file.write(undent(content))


def undent(content: str) -> str:
    content = dedent(content)
    if content and content[0] == '\n':
        content = content[1:]
    return content


class TestWithFiles(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None  # pylint: disable=invalid-name
        if sys.path[0] != os.getcwd():
            sys.path.insert(0, os.getcwd())
        self.previous_directory = os.getcwd()
        self.temporary_directory = tempfile.mkdtemp()
        os.chdir(os.path.expanduser(self.temporary_directory))
        sys.path.insert(0, os.getcwd())

    def tearDown(self) -> None:
        super().tearDown()
        os.chdir(self.previous_directory)
        shutil.rmtree(self.temporary_directory)

    def expect_file(self, path: str, expected: str) -> None:
        with open(path, 'r') as file:
            actual = file.read()
            self.assertEqual(actual, undent(expected))
