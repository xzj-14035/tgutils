"""
Utilities for main functions.
"""

from .logging import tg_qsub_logger
from argparse import Namespace

import dynamake.application as da
import numpy as np
import resource


def main_adapter(args: Namespace) -> None:  # pylint: disable=unused-argument
    """
    Perform last minute adaptation of the program following parsing the command line options.
    """
    np.seterr(all='raise')

    (_soft, hard) = resource.getrlimit(resource.RLIMIT_NOFILE)  # pylint: disable=invalid-name
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    da.Prog.logger = tg_qsub_logger(da.Prog.logger)
