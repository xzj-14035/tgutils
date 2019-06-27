"""
Utilities for using DynaMake.
"""

from dynamake.make import Invocation
from dynamake.make import require
from dynamake.patterns import each_string
from dynamake.patterns import Strings


def tg_require(*paths: Strings) -> None:
    """
    Require all the specified paths with a parallel context.

    This sets up the invocation context(s) of all the actions needed to build these files, and any
    of their dependencies, such that ``parallel_size`` contains the number of paths and
    ``parallel_index`` contains the index of the specific path. If nested, the context of the
    innermost call is used.

    The ``parallel_size`` and ``parallel_index`` context can then be embedded in the ``run_prefix``
    of the actions, to be passed to ``qsubber`` which uses this information to optimize the
    assignment of CPUs to SunGrid jobs.

    For example, suppose you wrote the following in ``DynaMake.yaml``:

    .. code-block:: yaml

        - when:
            is_parallel: True
            step: my_expensive_multi_processing_step
          then:
            run_prefix:
              'qsubber -v -I {parallel_index} -S {parallel_size} -j job-{action_id} -s 8- --'

    Then ``qsubber`` will allocate at least 8 CPUs for each action invoked by ``some_step``. If
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

    current = Invocation.current
    old_is_parallel = current.context['is_parallel']
    old_size = current.context['parallel_size']
    old_index = current.context['parallel_index']
    try:
        current.context['is_parallel'] = True
        current.context['parallel_size'] = len(paths_list)
        for index, path in enumerate(paths_list):
            current.context['parallel_index'] = index
            require(path)
    finally:
        current.context['is_parallel'] = old_is_parallel
        current.context['parallel_size'] = old_size
        current.context['parallel_index'] = old_index


def tg_reset() -> None:
    """
    Reset the persistent context (for tests).
    """
    Invocation.top.context['is_parallel'] = False
    Invocation.top.context['parallel_size'] = -1
    Invocation.top.context['parallel_index'] = -1


tg_reset()
