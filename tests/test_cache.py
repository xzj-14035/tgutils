"""
Test the thread-safe cache.
"""

from tgutils.cache import Cache
from unittest import TestCase

# pylint: disable=missing-docstring,too-many-public-methods,no-self-use
# pylint: disable=blacklisted-name,too-few-public-methods


class TestCache(TestCase):

    def test_lookup(self) -> None:
        cache: Cache[str, int] = Cache()
        self.assertFalse('one' in cache)
        self.assertEqual(cache.lookup('one', lambda: 1), 1)
        self.assertEqual(cache.lookup('one', lambda: 2), 1)
        self.assertTrue('one' in cache)
