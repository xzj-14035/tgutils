"""
Stubs for ``mypy`` for ``pandas.core.window``.
"""

from pandas import Series


class Rolling():

    def mean(self) -> Series: ...

    def median(self) -> Series: ...

    def std(self) -> Series: ...
