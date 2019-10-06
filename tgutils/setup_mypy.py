"""
Import this module in the ``setup.py`` file to use the provided Numpy/Pandas ``mypy`` stubs.

TODO: This is a horrible hack.
"""

import os

if os.path.exists('.poor_stubs'):
    os.remove('.poor_stubs')
os.symlink(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'poor_stubs'), '.poor_stubs')
os.environ['MYPYPATH'] = '.poor_stubs'
