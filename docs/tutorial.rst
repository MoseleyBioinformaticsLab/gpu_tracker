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
    import torch
    
    def example_function() -> torch.Tensor:
        t1 = torch.tensor(list(range(10000000))).cuda()
        t2 = torch.tensor(list(range(10000000))).cuda()
        return t1 * t2

.. code:: python3

    tracker = gput.Tracker()
    tracker.start()
    example_function()
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
       System: 1.799
       Main:
          Total RSS: 0.596
          Private RSS: 0.578
          Shared RSS: 0.018
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.523
          Private RSS: 0.505
          Shared RSS: 0.018
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.506
       Descendents: 0.0
       Combined: 0.506
    Compute time:
       Unit: hours
       Time: 0.001


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
        example_function()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 1.734
       Main:
          Total RSS: 0.603
          Private RSS: 0.585
          Shared RSS: 0.018
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.523
          Private RSS: 0.505
          Shared RSS: 0.018
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.506
       Descendents: 0.0
       Combined: 0.506
    Compute time:
       Unit: hours
       Time: 0.001


The units of the computational resources can be modified as desired. For
example, to measure the RAM in megabytes, the GPU RAM in megabytes, and
the compute time in seconds:

.. code:: python3

    with gput.Tracker(ram_unit='megabytes', gpu_ram_unit='megabytes', time_unit='seconds') as tracker:
        example_function()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: megabytes
       System capacity: 67254.166
       System: 1847.591
       Main:
          Total RSS: 603.525
          Private RSS: 585.269
          Shared RSS: 18.256
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 523.522
          Private RSS: 505.266
          Shared RSS: 18.256
    Max GPU RAM:
       Unit: megabytes
       Main: 506.0
       Descendents: 0.0
       Combined: 506.0
    Compute time:
       Unit: seconds
       Time: 2.768


The same information as the text format can be provided as a dictionary
via the ``to_json()`` method of the ``Tracker``.

.. code:: python3

    import json
    print(json.dumps(tracker.to_json(), indent=1))


.. code:: none

    {
     "max_ram": {
      "unit": "megabytes",
      "system_capacity": 67254.165504,
      "system": 1847.590912,
      "main": {
       "total_rss": 603.5251199999999,
       "private_rss": 585.269248,
       "shared_rss": 18.255872
      },
      "descendents": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 523.5220479999999,
       "private_rss": 505.266176,
       "shared_rss": 18.255872
      }
     },
     "max_gpu_ram": {
      "unit": "megabytes",
      "main": 506.0,
      "descendents": 0.0,
      "combined": 506.0
     },
     "compute_time": {
      "unit": "seconds",
      "time": 2.767793655395508
     }
    }


The ``Tracker`` class additionally has fields that provide the usage
information for each computational resource as python data classes.

.. code:: python3

    tracker.max_ram




.. code:: none

    MaxRAM(unit='megabytes', system_capacity=67254.165504, system=1847.590912, main=RSSValues(total_rss=603.5251199999999, private_rss=585.269248, shared_rss=18.255872), descendents=RSSValues(total_rss=0.0, private_rss=0.0, shared_rss=0.0), combined=RSSValues(total_rss=523.5220479999999, private_rss=505.266176, shared_rss=18.255872))



.. code:: python3

    tracker.max_ram.unit




.. code:: none

    'megabytes'



.. code:: python3

    tracker.max_ram.main




.. code:: none

    RSSValues(total_rss=603.5251199999999, private_rss=585.269248, shared_rss=18.255872)



.. code:: python3

    tracker.max_ram.main.total_rss




.. code:: none

    603.5251199999999



.. code:: python3

    tracker.max_gpu_ram




.. code:: none

    MaxGPURAM(unit='megabytes', main=506.0, descendents=0.0, combined=506.0)



.. code:: python3

    tracker.compute_time




.. code:: none

    ComputeTime(unit='seconds', time=2.767793655395508)



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
surrounded by quotes. Below is an example of running the ``bash``
command with an argument of ``example-script.sh``. When the command
completes, its status code is reported.

.. code:: none

    $ gpu-tracker -e "bash example-script.sh"


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 2.55
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendents:
          Total RSS: 0.83
          Private RSS: 0.708
          Shared RSS: 0.122
       Combined:
          Total RSS: 0.832
          Private RSS: 0.709
          Shared RSS: 0.123
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.314
       Combined: 0.314
    Compute time:
       Unit: hours
       Time: 0.001


*Notice that the RAM and GPU RAM usage primarily takes place in the
descendent processes since the bash command itself calls the commands
relevant to resource usage.*

The units of the computational resources can be modified. For example,
–tu stands for time-unit, –gru stands for gpu-ram-unit, and –ru stands
for ram-unit.

.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' --tu=seconds --gru=megabytes --ru=megabytes


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: megabytes
       System capacity: 67254.166
       System: 2458.182
       Main:
          Total RSS: 3.072
          Private RSS: 0.373
          Shared RSS: 2.699
       Descendents:
          Total RSS: 830.271
          Private RSS: 708.19
          Shared RSS: 122.081
       Combined:
          Total RSS: 831.537
          Private RSS: 708.563
          Shared RSS: 122.974
    Max GPU RAM:
       Unit: megabytes
       Main: 0.0
       Descendents: 314.0
       Combined: 314.0
    Compute time:
       Unit: seconds
       Time: 3.316


By default, the computational-resource-usage statistics are printed to
the screen. The ``-o`` or ``--output`` option can be specified to store
that same content in a file.

.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' -o out.txt 


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.txt


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 2.394
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendents:
          Total RSS: 0.831
          Private RSS: 0.709
          Shared RSS: 0.122
       Combined:
          Total RSS: 0.832
          Private RSS: 0.709
          Shared RSS: 0.123
    Max GPU RAM:
       Unit: gigabytes
       Main: 0.0
       Descendents: 0.314
       Combined: 0.314
    Compute time:
       Unit: hours
       Time: 0.001

By default, the format of the output is “text”. The ``-f`` or
``--format`` option can specify the format to be “json” instead.

.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' -f json


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    {
     "max_ram": {
      "unit": "gigabytes",
      "system_capacity": 67.254165504,
      "system": 2.3758110720000003,
      "main": {
       "total_rss": 0.0031457280000000004,
       "private_rss": 0.000376832,
       "shared_rss": 0.0027688960000000003
      },
      "descendents": {
       "total_rss": 0.8303943680000001,
       "private_rss": 0.708313088,
       "shared_rss": 0.12208128000000001
      },
      "combined": {
       "total_rss": 0.8316641280000001,
       "private_rss": 0.7086899200000001,
       "shared_rss": 0.122974208
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "main": 0.0,
      "descendents": 0.314,
      "combined": 0.314
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.0009229619635476007
     }
    }


.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' -f json -o out.json


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.json


.. code:: none

    {
     "max_ram": {
      "unit": "gigabytes",
      "system_capacity": 67.254165504,
      "system": 2.3479746560000003,
      "main": {
       "total_rss": 0.0030228480000000003,
       "private_rss": 0.000323584,
       "shared_rss": 0.0026992640000000003
      },
      "descendents": {
       "total_rss": 0.830509056,
       "private_rss": 0.708481024,
       "shared_rss": 0.12202803200000001
      },
      "combined": {
       "total_rss": 0.831725568,
       "private_rss": 0.708804608,
       "shared_rss": 0.12292096000000001
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "main": 0.0,
      "descendents": 0.314,
      "combined": 0.314
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.000929061041937934
     }
    }
