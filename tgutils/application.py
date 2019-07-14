"""
Utilities for main functions.
"""

from argparse import Namespace
from contextlib import contextmanager
from dynamake.application import *  # pylint: disable=wildcard-import,unused-wildcard-import
from dynamake.application import main as da_main
from dynamake.application import reset_application as da_reset_application
from logging import Logger
from logging import LoggerAdapter
from multiprocessing import Lock
from time import sleep
from typing import Any
from typing import Iterator

import fcntl
import numpy as _np
import os
import resource


def main(parser: ArgumentParser,  # pylint: disable=function-redefined
         functions: Optional[List[str]] = None,
         *, adapter: Optional[Callable[[Namespace], None]] = None) -> None:
    """
    A generic ``main`` function for configurable functions.

    See :py:func:`dynamake.application.main`.
    """
    def _adapter(args: Namespace) -> None:
        tgutils_adapter(args)
        if adapter is not None:
            adapter(args)
    da_main(parser, functions, adapter=_adapter)


def tgutils_adapter(args: Namespace) -> None:  # pylint: disable=unused-argument
    """
    Perform last minute adaptation of the program following parsing the command line options.
    """
    _np.seterr(all='raise')

    (_soft, hard) = resource.getrlimit(resource.RLIMIT_NOFILE)  # pylint: disable=invalid-name
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    _set_numpy_random_seed()

    Prog.logger = tg_qsub_logger(Prog.logger)


def reset_application() -> None:  # pylint: disable=function-redefined
    """
    Reset the global state (for tests).
    """
    da_reset_application()
    if _set_numpy_random_seed not in Prog.on_parallel_calls:
        Prog.on_parallel_calls.append(_set_numpy_random_seed)


def _set_numpy_random_seed() -> None:
    random_seed = Prog.parameter_values.get('random_seed')
    if random_seed is not None:
        _np.random.seed(random_seed)


reset_application()


def indexed_range(index: int, *, size: int, invocations: int = 0) -> range:
    """
    Return a range of indices for an indexed invocation.

    Each invocation will get its own range, where the range sizes will be the same (as much as
    possible) for each invocation.

    If the number of invocations is zero, it is assumed to be the number of available parallel
    processes, that is, that there will be one invocation per parallel process.
    """
    invocations = invocations or min(processes(), size)
    start = int(round(index * size / invocations))
    stop = int(round((index + 1) * size / invocations))
    return range(start, stop)


class FileLockLoggerAdapter(LoggerAdapter):
    """
    A logger adapter that performs a file lock around each logged messages.

    If used consistently in multiple applications, this ensures that logging does not get garbled,
    even when running across multiple machines.
    """

    def __init__(self, logger: Logger, path: str) -> None:
        """
        Create a logger adapter that locks the specified directory path.
        """
        super().__init__(logger, None)  # type: ignore
        self._path = path
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self._fd = os.open(self._path, os.O_CREAT)
        self._lock = Lock()

    def log(self, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=arguments-differ
        """
        Log a message while locking the directory.
        """
        with self._lock:
            with lock_file(self._path, self._fd):
                super().log(*args, **kwargs)


@contextmanager
def lock_file(lock_path: str, lock_fd: int) -> Iterator[None]:
    """
    Perform some action while holding a file lock.
    """
    slept = 0.0
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BaseException:  # pylint: disable=broad-except
            slept += 0.1
            if slept > 60:
                raise RuntimeError('Failed to obtain lock file: %s '
                                   'for more than 60 seconds'
                                   % lock_path)
            sleep(0.1)
            continue

    try:
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)


def tg_qsub_logger(logger: Logger) -> Logger:
    """
    Wrap a logger so that messages will not get interleaved with other program invocations and/or
    the messages from the ``tg_qsub`` script.
    """
    if os.getenv('ENVIRONMENT') == 'BATCH':
        return logger
    return FileLockLoggerAdapter(logger,  # type: ignore
                                 os.path.join(os.getenv('QSUB_TMP_DIR', '.qsub'),
                                              'lock'))