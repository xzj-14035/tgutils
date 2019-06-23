TGUtils - Utilities for Tanay Group lab code
============================================

This package contains common utilities used by the Tanay Group Python lab code (for example,
``metacell``). These utilities are generally useful and not associated with any specific project.

Usage
-----

Installing
..........

Go to some directory where local versions of Python packages may be installed. Run:

.. code-block:: bash

   pip install -e hg+https://bitbucket.org/orenbenkiki/tgutils#egg=tgutils

You might need to specify ``pip install --user ...`` if you do not have ``sudo`` privileges.

.. note::

    This will create a ``./src/tgutils`` directory where the actuall installation will reside. All
    that will be installed in the Python packages directory is a link to this location. This isn't
    even a symbolic link - it is just a text file named ``tgutils.egg-link`` containing the absolute
    path of the real installation directory. Despite the name, there's no egg file involved. Pip
    does leave a ``./src/pip-delete-this-directory.txt`` file, signifying this is meant to be just a
    temporary, but in actuallity, if you remove the ``src`` directory, you will have lost the
    installation.

Importing
.........

Instead of the vanilla:

.. code-block:: python

    import numpy as np
    import pandas as pd

Write the modified:

.. code-block:: python

    import tgutils.numpy as np
    import tgutils.pandas as pd

This this will provide access to the vanilla symbols using the ``np.`` and/or ``pd.`` prefixes,
and will also provide access to the enhanced functionality described below.

Phantom Types
-------------

The vanilla ``np.ndarray``, ``pd.Series`` and ``pd.DataFrame`` types are very generic. They are the
same regardless of the element data type used, and in the case of ``np.ndarray``, the number of
dimensions (array vs. matrix).

To understand the code, it is helpful to keep track of a more detailed data type - whether the
variable is an array or a matrix, and what the element data type is. To facilitate this, ``tgutils``
provides what is known as "Phantom Types". These types can be used in ``mypy`` type declarations,
even though the actual data type of the variables remains the vanilla Numpy/Pandas data type.

See :py:mod:`tgutils.numpy` and :py:mod:`tgutils.pandas` for the list of provided phantom types.

For example, instead of writing:

.. code-block:: python

    def compute_some_integers() -> np.ndarray:
        ...

    def compute_some_floats() -> np.ndarray:
        ...

    def compute_stuff(foo: np.ndarray, bar: np.ndarray) -> ...:
        ...

    def workflow() -> ...:
        foo = compute_some_integers()
        bar = compute_some_floats()
        compute_stuff(foo, bar)

You should write:

.. code-block:: python

    def compute_some_integers() -> np.ArrayInt32:
        ...

    def compute_some_floats() -> np.MatrixFloat64:
        ...

    def compute_stuff(foo: np.ArrayInt32, bar: np.MatrixFloat64) -> ...:
        ...

    def workflow() -> ...:
        foo = compute_some_integers()
        bar = compute_some_floats()
        compute_stuff(foo, bar)

This will allow the reader to understand the exact data types involved. Even better, it will allow
``mypy`` to verify that you actually pass the correct data type to each function invocation.
For example, if you by mistake write ``compute_stuff(bar, foo)`` then ``mypy`` will complain that
the data types do not match - even though, under the covers, both ``foo`` and ``bar`` have exactly
the same data type at run-time: ``np.ndarray``.

Type Operations
...............

Control over the data types is also important when performing computations. It affects performance,
memory consumption and even the semantics of some operations. For example, integer elements can
never be ``NaN`` while floating point elements can, boolean elements have their own logic, and
string elements are different from numeric elements.

To help with this, ``tgutils`` provides two functions, ``am`` and ``be``. Both these functions
return the requested data type, but ``am`` is just an assertion while ``be`` is a cast operation.
That is, writing ``ArrayInt32.am(foo)`` will return ``foo`` as an ``ArrayInt32``, or will raise an
error if ``foo`` is not an array of ``int32``; while writing ``ArrayInt32.be(foo)`` will always
return an ``ArrayInt32``, which is either ``foo`` if it is an array of ``int32``, or a copy of
``foo`` whose elements are the conversion of the elements of ``foo`` to ``int32``.

De/serialization
................

The phantom types also provide read and write operations for efficiently storing data on the disk.
That is, writing ``ArrayInt32.read(path)`` will read an array of ``int32`` elements from the
specified path, and ``ArrayInt32.write(foo, path)`` will write an array of ``int32`` elements
into the specified path.

DynaMake and Qsubber
--------------------

The :py:mod:`tgutils.tg_qsub` script deals with submitting jobs to run on the SunGrid cluster in the
Tanay Group lab.

A :py:func:`tgutils.tg_require_in_parallel` function allows for collecting context for optimizing
the slot allocation of ``tg_qsub`` for maximizing the cluster utilization and minimizing wait times.
This has no effect unless the collected context values are explicitly used in the ``run_prefix``
and/or ``run_suffix`` action wrapper of some step.

This is a convoluted and sub-optimal mechanism but has significant performance benefits in the
specific environment it was designed for.

Logging
-------

The default Python logging that prints to ``stderr`` works well for a single application. However,
when running multiple applications in parallel, log messages may get interleaved resulting in
garbled output.

This can be avoided using the :py:class:`tgutils.logging.FileLockLoggerAdapter`, which uses a file
lock operation around each emitted log messages.

If using :py:func:`tgutils.logging.tg_qsub_logger`, then the lock file is shared with our
``tg_qsub`` script, so that its log messages will not be interleaved with any application's log
messages. The processes running on the cluster servers will not use any locking, since the output of
each one is collected to a separate file which is only reported (atomically) when it is done.
