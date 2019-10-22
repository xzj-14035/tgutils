"""
Utilities for using DynaMake.
"""

from dynamake.make import *  # pylint: disable=wildcard-import,unused-wildcard-import
from dynamake.make import reset_make as dm_reset_make
from tgutils.application import *  # pylint: disable=wildcard-import,unused-wildcard-import


def tg_require(*paths: Strings) -> None:
    """
    Require all the specified paths with a parallel context.

    This sets up the invocation context(s) of all the actions needed to build these files, and any
    of their dependencies, such that ``parallel_size`` contains the number of paths and
    ``parallel_index`` contains the index of the specific path. If nested, the context of the
    innermost call is used.

    The ``parallel_size`` and ``parallel_index`` context can then be embedded in the ``run_prefix``
    of the actions, to be passed to ``tg_qsub`` which uses this information to optimize the
    assignment of CPUs to SunGrid jobs.

    For example, suppose you wrote the following in ``DynaMake.yaml``:

    .. code-block:: yaml

        - when:
            is_parallel: True
            step: my_expensive_multi_processing_step
          then:
            run_prefix:
              'tg_qsub -v -I {parallel_index} -S {parallel_size} -j job-{action_id} -s 8- --'

    Then ``tg_qsub`` will allocate at least 8 CPUs for each action invoked by ``some_step``. If
    there are only a few such invocations (say, up to one per cluster server), it may assign more
    CPUs to each invocation (up to all the CPUs on each server). If there are many invocations, it
    will assign less, to ensure as many invocations as possible run in parallel.

    This is due to the unfortunate fact that speedup gained by using more CPUs is not linear; that
    is, a 16-CPU action takes longer than half the time it takes using 8 CPUs. Therefore, if all we
    have is a 16-CPU machine, we are better off running two 8-CPU actions in parallel than one
    16-CPU action followed by another.

    This is overly convoluted, sub-optimal, and very specific to the way we distribute actions on
    the SunGrid cluster in the Tanay Group lab. The cluster manager should arguably do much better
    without all these complications. However, all we have is ``qsub``.
    """
    paths_list = list(each_string(*paths))
    if len(paths_list) <= 1:
        require(*paths_list)
        return

    context = require_context()
    old_is_parallel = context['is_parallel']
    old_size = context['parallel_size']
    old_index = context['parallel_index']
    try:
        context['is_parallel'] = True
        context['parallel_size'] = len(paths_list)
        for index, path in enumerate(paths_list):
            context['parallel_index'] = index
            require(path)
    finally:
        context['is_parallel'] = old_is_parallel
        context['parallel_size'] = old_size
        context['parallel_index'] = old_index


def parallel_jobs() -> int:
    """
    Return the number of jobs to use for a parallel sub-process in the current context (can be
    passed to ``--jobs``).

    This assumes all the actions of the innermost ``tg_require`` in the current context are
    executed, and tries to utilize all the available CPUs for them.
    """
    context = require_context()

    is_parallel = context.get('is_parallel', False)
    if not is_parallel:
        return 0

    jobs = Resources.total['jobs']
    if jobs == 0:
        return 0

    size = context['parallel_size']
    if size > jobs:
        return 1

    index = context['parallel_index']
    return int(((index + 1) / size) * jobs) - int((index / size) * jobs)


def reset_make() -> None:  # type: ignore # pylint: disable=function-redefined
    """
    Reset the persistent context (for tests).
    """
    dm_reset_make()
    Invocation.top.require_context['is_parallel'] = False
    Invocation.top.require_context['parallel_size'] = -1
    Invocation.top.require_context['parallel_index'] = -1


reset_make()
