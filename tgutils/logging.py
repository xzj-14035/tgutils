"""
Utilities for improved logging.
"""

from contextlib import contextmanager
from logging import Logger
from logging import LoggerAdapter
from multiprocessing import Lock
from time import sleep
from typing import Any
from typing import Iterator

import fcntl
import os


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
