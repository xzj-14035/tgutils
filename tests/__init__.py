"""
Test the tgutils module.
"""

from unittest import TestCase

import tgutils

# pylint: disable=missing-docstring,too-many-public-methods


class TestTGUtils(TestCase):

    def test_do_nothing(self) -> None:
        tgutils.do_nothing()
        self.assertEqual(1, 1)
