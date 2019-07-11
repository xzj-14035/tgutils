"""
Test the test utilities.
"""

from tgutils.tests import TestWithFiles
from tgutils.tests import write_file

# pylint: disable=missing-docstring,too-many-public-methods,no-self-use
# pylint: disable=blacklisted-name,too-few-public-methods


class TestFiles(TestWithFiles):

    def test_files(self) -> None:
        write_file('foo', """
            bar
              baz
        """)
        self.expect_file('foo', 'bar\n  baz\n')
