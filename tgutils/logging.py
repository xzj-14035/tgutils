"""
Utilities for improved logging.
"""

from logging import Logger
from logging import LoggerAdapter
from multiprocessing import Lock
from time import sleep
from typing import Any

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
            slept = 0.0
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BaseException:  # pylint: disable=broad-except
                    slept += 0.1
                    if slept > 60:
                        raise RuntimeError('Failed to obtain lock directory: %s '
                                           'for more than 60 seconds'
                                           % self._path)
                    sleep(0.1)
                    continue

            try:
                super().log(*args, **kwargs)
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)


def qsub_logger(logger: Logger) -> Logger:
    """
    Wrap a logger with directory locking so that both parallel and distributed logs will not get
    garbled.
    """
    if os.getenv('ENVIRONMENT') == 'BATCH':
        return logger
    return FileLockLoggerAdapter(logger,  # type: ignore
                                 os.path.join(os.getenv('QSUB_TMP_DIR', '.qsub'),
                                              'lock'))
