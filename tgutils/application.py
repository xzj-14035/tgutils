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
from socket import gethostname
from time import sleep
from typing import Any
from typing import Iterator

import fcntl
import numpy as _np
import os
import resource
import sys


def main(parser: ArgumentParser,  # type: ignore # pylint: disable=function-redefined
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


def maximal_processors() -> None:
    """
    Limit the maximal number of processors to use if we are running inside some cluster environment.
    """


def maximal_open_files() -> None:
    """
    Ensure we can use the maximal number of open files at the same time.
    """
    (_soft, hard) = resource.getrlimit(resource.RLIMIT_NOFILE)  # pylint: disable=invalid-name
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def tgutils_adapter(args: Namespace) -> None:  # pylint: disable=unused-argument
    """
    Perform last minute adaptation of the program following parsing the command line options.
    """
    _np.seterr(all='raise')

    maximal_processors()

    maximal_open_files()

    _set_numpy_random_seed()

    Prog.logger = tg_qsub_logger(Prog.logger)


def reset_application() -> None:  # type: ignore # pylint: disable=function-redefined
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
        Prog.logger.debug('Using numpy random seed: %s', random_seed)


reset_application()


def indexed_range(index: int, *, size: int, invocations: int = 0) -> range:
    """
    Return a range of indices for an indexed invocation.

    Each invocation ``index`` will get its own range, where the range sizes will be the same (as
    much as possible) for each invocation.

    If the number of ``invocations`` is zero, it is assumed to be the number of available parallel
    processes, that is, that there will be one invocation per parallel process (at most ``size``).
    """
    invocations = invocations or processes_for(size)
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
            with lock_file(self._path, self._fd, is_just_for_log=True):
                super().log(*args, **kwargs)


@contextmanager
def lock_file(lock_path: str, lock_fd: int, *,
              shared: bool = False, is_just_for_log: bool = False) -> Iterator[None]:
    """
    Perform some action while holding a file lock.
    """
    slept = 0.0
    step = 1 / 8
    locked = True

    while True:
        try:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_fd, mode | fcntl.LOCK_NB)
            break

        except BaseException:  # pylint: disable=broad-except
            if is_just_for_log and slept > 60.0:
                sys.stderr.write('WARNING: %s @ %s: Timeout waiting %s seconds for lock file: %s\n'
                                 % (os.getpid(), gethostname(), round(slept), lock_path))
                locked = False
                break

            sleep(step)
            slept += step
            if step < 1.0:
                step *= 2.0
            continue

    try:
        if locked and slept >= 60.0:
            sys.stderr.write('WARNING: %s @ %s: Waited %s seconds for lock file: %s\n'
                             % (os.getpid(), gethostname(), round(slept), lock_path))
        yield
    finally:
        if locked:
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


class Loop:  # pylint: disable=too-many-instance-attributes
    """
    Log progress for a (possibly parallel) loop.
    """

    def __init__(self, *, start: str, progress: str, completed: str, log_every: int = 1,
                 log_with: Optional[int] = None, expected: Optional[int] = None) -> None:
        #: The format of the start message.
        self.start = start

        #: The format of the progress message.
        self.progress = progress

        #: The format of the completion messages.
        self.completed = completed

        #: Emit a log message every this amount of iterations (typically a power of 10).
        self.log_every = log_every

        #: The value in the log message is divided by this amount (typically a power of 1000).
        self.log_with = log_with or log_every

        #: The shared memory iteration counter.
        self.shared_counter = Value(ctypes.c_int32)

        #: Granularity of parallel counting.
        self.local_every = self.log_every // 10

        #: The expected number of increments.
        self.expected = expected

        if self.expected is None or self.expected >= self.log_every:
            Prog.logger.info(self.start)

    def __enter__(self) -> 'Loop':
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.done()

    def step(self, fraction: Optional[float] = None) -> None:
        """
        Indicate a loop iteration.

        Ideally is called at the end of the iteration to indicate the iteration has completed. If
        the loop code is complex (contains ``continue`` etc.) then it is placed at the start of the
        code.
        """
        with self.shared_counter.get_lock():
            self.shared_counter.value += 1
            total = self.shared_counter.value

        if total % self.log_every > 0:
            return

        if fraction is None and self.expected is not None:
            fraction = total / self.expected

        if fraction is None:
            Prog.logger.info(self.progress, total // self.log_with)
        else:
            Prog.logger.info(self.progress, total // self.log_with, 100 * fraction)

    def done(self) -> None:
        """
        Indicate the loop has completed.
        """
        total = self.shared_counter.value
        if total >= self.log_every or self.expected is None or self.expected >= self.log_every:
            Prog.logger.info(self.completed, total)


def each_file_line(path: str, loop: Optional[Loop] = None) -> Iterator[Tuple[int, str]]:
    """
    Loop on each file line.
    """
    size = Stat.stat(path).st_size
    number = 0
    offset = 0
    with open(path, 'r') as file:
        for line in file:
            number += 1
            yield number, line
            if loop is None:
                continue
            offset += len(line)
            fraction = offset / size
            loop.step(fraction)
