"""
Common utilities for tests.
"""

from unittest import TestCase

import os
import shutil
import sys
import tempfile

# pylint: disable=missing-docstring


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
