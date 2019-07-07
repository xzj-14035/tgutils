"""
Import this module in the ``setup.py`` file to use the provided Numpy/Pandas ``mypy`` stubs.

TODO: This is a horrible hack.
"""

import os

os.environ['MYPYPATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numpy_stubs')
