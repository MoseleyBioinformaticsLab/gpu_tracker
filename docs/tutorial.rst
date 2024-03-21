Tutorial
========

.. code:: python3

    import gpu_tracker.tracker as track

The ``gpu_tracker.tracker`` module provides the ``Tracker`` class which
uses an underlying thread to measure computational resource usage,
namely the compute time, maximum RAM used, and maximum GPU RAM used. The
``start()`` method starts this thread which tracks usage in the
background. After calling ``start()``, write the code to measure
resource usage, followed by calling the ``stop()`` method. The compute
time will be the time from the call to ``start()`` to the call to
``stop()`` and the RAM and GPU quantities will be the amount of RAM used
by the code that’s in between ``start()`` and ``stop()``. The
``Tracker`` class additionally has a ``__str__`` method so it can be
printed as a string that formats the values and units of each
computational resource.

.. code:: python3

    tracker = track.Tracker()
    tracker.start()
    # Perform expensive operations
    tracker.stop()
    print(tracker)


.. parsed-literal::

    Max RAM: 0.068 gigabytes
    Max GPU: 0.000 gigabytes
    Compute time: 0.000 hours


The equivalent can be accomplished using ``Tracker`` as a context
manager rather than explicitly calling ``start()`` and ``stop()``.

.. code:: python3

    with track.Tracker() as tracker:
        # Perform expensive operations
        pass
    print(tracker)


.. parsed-literal::

    Max RAM: 0.068 gigabytes
    Max GPU: 0.000 gigabytes
    Compute time: 0.000 hours


The units of the computational resources can be modified as desired. For
example, to measure the RAM in megabytes, the GPU RAM in kilobytes, and
the compute time in seconds:

.. code:: python3

    with track.Tracker(ram_unit='megabytes', gpu_unit='kilobytes', time_unit='seconds') as tracker:
        # Perform expensive operations
        pass
    print(tracker)


.. parsed-literal::

    Max RAM: 67.662 megabytes
    Max GPU: 0.000 kilobytes
    Compute time: 1.043 seconds


Additionally, the individual measurements and units are available as
attributes in the ``Tracker`` class.

.. code:: python3

    print(tracker.max_ram, tracker.ram_unit)


.. parsed-literal::

    67.661824 megabytes

