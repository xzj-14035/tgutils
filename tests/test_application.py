"""
Test the tgutils.parallel module.
"""

from testfixtures import OutputCapture  # type: ignore
from tgutils.application import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tgutils.tests import TestWithReset

import tgutils.numpy as np

# pylint: disable=missing-docstring,too-many-public-methods,no-self-use


class TestApplication(TestWithReset):

    def test_indexed_range(self) -> None:
        self.assertEqual(indexed_range(0, invocations=2, size=4), range(0, 2))
        self.assertEqual(indexed_range(1, invocations=2, size=4), range(2, 4))
        self.assertEqual(indexed_range(0, invocations=2, size=5), range(0, 2))
        self.assertEqual(indexed_range(1, invocations=2, size=5), range(2, 5))

    def test_random_seed(self) -> None:
        @config(top=True, random=True)
        def top() -> None:  # pylint: disable=unused-variable
            print(np.random.random())

        np.random.seed(11)

        sys.argv += ['top', '--random_seed', '17']
        with OutputCapture() as output:
            main(ArgumentParser(description='Test'))

        np.random.seed(17)
        output.compare('%s\n' % np.random.random())

    def test_random_parallel(self) -> None:
        @config(random=True)
        def _roll() -> float:
            return np.random.random()

        @config(top=True, parallel=True)
        def top() -> None:  # pylint: disable=unused-variable
            results = parallel(2, _roll)
            print(sorted(results))

        np.random.seed(11)

        sys.argv += ['top', '--random_seed', '17']
        with OutputCapture() as output:
            main(ArgumentParser(description='Test'))

        np.random.seed(17)
        first = np.random.random()
        np.random.seed(18)
        second = np.random.random()
        results = sorted([first, second])

        output.compare('%s\n' % results)
