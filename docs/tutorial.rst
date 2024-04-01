.. _tutorial-label:

Tutorial
========

API
---

The ``gpu_tracker`` package provides the ``Tracker`` class which uses an
underlying thread to measure computational resource usage, namely the
compute time, maximum RAM used, and maximum GPU RAM used. The
``start()`` method starts this thread which tracks usage in the
background. After calling ``start()``, write the code to measure
resource usage, followed by calling the ``stop()`` method. The compute
time will be the time from the call to ``start()`` to the call to
``stop()`` and the RAM and GPU RAM quantities will be the amount of RAM
used by the code that’s in between ``start()`` and ``stop()``.

.. code:: python3

    import gpu_tracker as gput

.. code:: python3

    tracker = gput.Tracker()
    tracker.start()
    # Perform expensive operations
    tracker.stop()

The ``Tracker`` class implements the ``__str__`` method so it can be
printed as a string with the values and units of each computational
resource formatted.

.. code:: python3

    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 4.545
       Main:
          Total RSS: 0.061
          Private RSS: 0.05
          Shared RSS: 0.011
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.063
          Private RSS: 0.052
          Shared RSS: 0.011
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.0
       Combined: 0.0
    Compute time:
       Unit: hours
       Time: 0.0


The system capacity is a constant for the total RAM capacity across the
entire operating system, not to be confused with the maximum system RAM
which is the maximum OS RAM that was actually used over the duration of
the computational-resource tracking. Both the RAM and GPU RAM are split
up into 3 sections, namely the usage of the main process itself followed
by the summed usage of any descendent processes it may have (i.e. child
processes, grandchild processes, etc.), and combined usage which is the
sum of the main and its descendent processes. RAM is divided further to
include the private RSS (RAM usage unique to the process), shared RSS
(RAM that’s shared by a process and at least one other process), and
total RSS (the sum of private and shared RSS). The private and shared
RSS values are only available on Linux distributions. So for non-linux
operating systems, the privated and shared RSS will remain 0 and only
the total RSS will be reported. Theoretically, the combined total RSS
would never exceed the overall system RAM usage, but inaccuracies
resulting from shared RSS can cause this to happen, especially for
non-linux operating systems (see note below).

The ``Tracker`` assumes that GPU memory is not shared accross multiple
processes and if it is, the reported GPU RAM of “descendent” and
“combined” may be an overestimation.

The compute time is the real time that the computational-resource
tracking lasted (as compared to CPU time).

**NOTE** *The keywords “descendents” and “combined” in the output above
indicate a sum of the RSS used by multiple processes. It’s important to
keep in mind that on non-linux operating systems, this sum does not take
into account shared memory but rather adds up the total RSS of all
processes, which can lead to an overestimation. For Linux distributions,
however, pieces of shared memory are only counted once.*

The ``Tracker`` can alternatively be used as a context manager rather
than explicitly calling ``start()`` and ``stop()``.

.. code:: python3

    with gput.Tracker() as tracker:
        # Perform expensive operations
        pass
    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 4.549
       Main:
          Total RSS: 0.063
          Private RSS: 0.052
          Shared RSS: 0.011
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.063
          Private RSS: 0.052
          Shared RSS: 0.011
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.0
       Combined: 0.0
    Compute time:
       Unit: hours
       Time: 0.0


The units of the computational resources can be modified as desired. For
example, to measure the RAM in megabytes, the GPU RAM in kilobytes, and
the compute time in seconds:

.. code:: python3

    with gput.Tracker(ram_unit='megabytes', gpu_ram_unit='kilobytes', time_unit='seconds') as tracker:
        # Perform expensive operations
        pass
    print(tracker)


.. code:: none

    Max RAM:
       Unit: megabytes
       System capacity: 67254.161
       System: 4548.833
       Main:
          Total RSS: 63.279
          Private RSS: 52.064
          Shared RSS: 11.215
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 63.283
          Private RSS: 52.068
          Shared RSS: 11.215
    Max GPU RAM:
       Unit: kilobytes
       Main: 0.0
       Descendents: 0.0
       Combined: 0.0
    Compute time:
       Unit: seconds
       Time: 0.059


The same information as the text format can be provided as a dictionary
via the ``to_json()`` method of the ``Tracker``.

.. code:: python3

    import json
    print(json.dumps(tracker.to_json(), indent=1))


.. code:: none

    {
     "max_ram": {
      "unit": "megabytes",
      "system_capacity": 67254.161408,
      "system": 4548.83328,
      "main": {
       "total_rss": 63.279104000000004,
       "private_rss": 52.064256,
       "shared_rss": 11.214848
      },
      "descendents": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 63.283199999999994,
       "private_rss": 52.068352,
       "shared_rss": 11.214848
      }
     },
     "max_gpu_ram": {
      "unit": "kilobytes",
      "main": 0.0,
      "descendents": 0.0,
      "combined": 0.0
     },
     "compute_time": {
      "unit": "seconds",
      "time": 0.058912038803100586
     }
    }


The ``Tracker`` class additionally has fields that provide the usage
information for each computational resource as python data classes.

.. code:: python3

    tracker.max_ram




.. code:: none

    MaxRAM(unit='megabytes', system_capacity=67254.161408, system=4548.83328, main=RSSValues(total_rss=63.279104000000004, private_rss=52.064256, shared_rss=11.214848), descendents=RSSValues(total_rss=0.0, private_rss=0.0, shared_rss=0.0), combined=RSSValues(total_rss=63.283199999999994, private_rss=52.068352, shared_rss=11.214848))



.. code:: python3

    trac

.. code:: python3

    tracker.max_gpu_ram




.. code:: none

    MaxGPURAM(unit='kilobytes', main=0.0, descendents=0.0, combined=0.0)



.. code:: python3

    tracker.compute_time




.. code:: none

    ComputeTime(unit='seconds', time=0.058912038803100586)



CLI
---

The ``gpu-tracker`` package also comes with a commandline interface that
can track the computational-resource-usage of any shell command, not
just python code. Entering ``gpu-tracker -h`` in a shell will show the
help message.

.. code:: none

    $ gpu-tracker -h


.. code:: none

    Tracks the computational resource usage (RAM, GPU RAM, and compute time) of a process corresponding to a given shell command.
    
    Usage:
        gpu-tracker --execute=<command> [--output=<output>] [--format=<format>] [--st=<sleep-time>] [--ru=<ram-unit>] [--gru=<gpu-ram-unit>] [--tu=<time-unit>]
    
    Options:
        -h --help               Show this help message.
        -e --execute=<command>  The command to run along with its arguments all within quotes e.g. "ls -l -a".
        -o --output=<output>    File path to store the computational-resource-usage measurements. If not set, prints measurements to the screen.
        -f --format=<format>    File format of the output. Either 'json' or 'text'. Defaults to 'text'.
        --st=<sleep-time>       The number of seconds to sleep in between usage-collection iterations.
        --ru=<ram-unit>         One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        --gru=<gpu-ram-unit>    One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        --tu=<time-unit>        One of 'seconds', 'minutes', 'hours', or 'days'.


The ``-e`` or ``--execute`` is a required option where the desired shell
command is provided, with both the command and its proceeding arguments
surrounded by quotes. Below is an example of running the ``sleep``
command with an argument of 2 seconds. When the command completes, its
status code is reported.

.. code:: none

    $ gpu-tracker -e 'sleep 2'


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 4.548
       Main:
          Total RSS: 0.002
          Private RSS: 0.0
          Shared RSS: 0.002
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.002
          Private RSS: 0.0
          Shared RSS: 0.002
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.0
       Combined: 0.0
    Compute time:
       Unit: hours
       Time: 0.0


The units of the computational resources can be modified. For example,
–tu stands for time-unit and –ru stands for ram-unit.

.. code:: none

    $ gpu-tracker -e 'sleep 2' --tu=seconds --ru=megabytes


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: megabytes
       System capacity: 67254.161
       System: 4550.529
       Main:
          Total RSS: 1.831
          Private RSS: 0.135
          Shared RSS: 1.696
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 1.831
          Private RSS: 0.135
          Shared RSS: 1.696
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.0
       Combined: 0.0
    Compute time:
       Unit: seconds
       Time: 1.075


By default, the computational-resource-usage statistics are printed to
the screen. The ``-o`` or ``--output`` option can be specified to store
that same content in a file.

.. code:: none

    $ gpu-tracker -e 'sleep 2' -o out.txt 


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.txt


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 4.567
       Main:
          Total RSS: 0.002
          Private RSS: 0.0
          Shared RSS: 0.002
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.002
          Private RSS: 0.0
          Shared RSS: 0.002
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.0
       Combined: 0.0
    Compute time:
       Unit: hours
       Time: 0.0

By default, the format of the output is “text”. The ``-f`` or
``--format`` option can specify the format to be “json” instead.

.. code:: none

    $ gpu-tracker -e 'sleep 2' -f json


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    {
     "max_ram": {
      "unit": "gigabytes",
      "system_capacity": 67.254161408,
      "system": 4.582764544000001,
      "main": {
       "total_rss": 0.001662976,
       "private_rss": 0.00013516800000000002,
       "shared_rss": 0.001527808
      },
      "descendents": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 0.001662976,
       "private_rss": 0.00013516800000000002,
       "shared_rss": 0.001527808
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "main": 0.0,
      "descendents": 0.0,
      "combined": 0.0
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.00030033310254414875
     }
    }


.. code:: none

    $ gpu-tracker -e 'sleep 2' -f json -o out.json


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.json


.. code:: none

    {
     "max_ram": {
      "unit": "gigabytes",
      "system_capacity": 67.254161408,
      "system": 4.584312832,
      "main": {
       "total_rss": 0.0017162240000000001,
       "private_rss": 0.00013516800000000002,
       "shared_rss": 0.0015810560000000002
      },
      "descendents": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 0.0017162240000000001,
       "private_rss": 0.00013516800000000002,
       "shared_rss": 0.0015810560000000002
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "main": 0.0,
      "descendents": 0.0,
      "combined": 0.0
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.0002998979224099053
     }
    }
