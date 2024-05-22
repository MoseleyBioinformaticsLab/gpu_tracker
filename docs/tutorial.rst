.. _tutorial-label:

Tutorial
========

API
---

The ``gpu_tracker`` package provides the ``Tracker`` class which uses a
subprocess to measure computational resource usage, namely the compute
time, maximum CPU utilization, mean CPU utilization, maximum RAM used,
and maximum GPU RAM used. The ``start()`` method starts this process
which tracks usage in the background. After calling ``start()``, one can
write the code for which resource usage is measured, followed by calling
the ``stop()`` method. The compute time will be the time from the call
to ``start()`` to the call to ``stop()`` and the RAM, GPU RAM, and CPU
utilization quantities will be the respective computational resources
used by the code that’s in between ``start()`` and ``stop()``.

.. code:: python3

    import gpu_tracker as gput
    from example_module import example_function

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
       System: 5.21
       Main:
          Total RSS: 0.827
          Private RSS: 0.674
          Shared RSS: 0.154
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.834
          Private RSS: 0.681
          Shared RSS: 0.154
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 0.535
       Main: 0.314
       Descendents: 0.0
       Combined: 0.314
    CPU utilization:
       System core count: 12
       System:
          Max core percent: 150.6
          Max CPU percent: 12.55
          Mean core percent: 122.9
          Mean CPU percent: 10.242
       Main:
          Max core percent: 98.6
          Max CPU percent: 8.217
          Mean core percent: 96.8
          Mean CPU percent: 8.067
       Descendents:
          Max core percent: 0.0
          Max CPU percent: 0.0
          Mean core percent: 0.0
          Mean CPU percent: 0.0
       Combined:
          Max core percent: 98.6
          Max CPU percent: 8.217
          Mean core percent: 96.8
          Mean CPU percent: 8.067
       Main number of threads: 15
       Descendents number of threads: 0
       Combined number of threads: 15
    Compute time:
       Unit: hours
       Time: 0.001


The output is organized by computational resource followed by
information specific to that resource. The system capacity is a constant
for the total RAM capacity across the entire operating system. There is
a system capacity field both for RAM and GPU RAM. This is not to be
confused with the system field, which measures the maximum RAM / GPU RAM
(operating system wide) that was actually used over the duration of the
computational-resource tracking. Both the RAM and GPU RAM have 3
additional fields, namely the usage of the main process itself followed
by the summed usage of any descendent processes it may have (i.e. child
processes, grandchild processes, etc.), and combined usage which is the
sum of the main and its descendent processes. RAM is divided further to
include the private RSS (RAM usage unique to the process), shared RSS
(RAM that’s shared by a process and at least one other process), and
total RSS (the sum of private and shared RSS). The private and shared
RSS values are only available on Linux distributions. So for non-linux
operating systems, the private and shared RSS will remain 0 and only the
total RSS will be reported. Theoretically, the combined total RSS would
never exceed the overall system RAM usage, but inaccuracies resulting
from shared RSS can cause this to happen, especially for non-linux
operating systems (see note below).

The ``Tracker`` assumes that GPU memory is not shared across multiple
processes and if it is, the reported GPU RAM of “descendent” and
“combined” may be an overestimation.

The CPU utilization includes the system core count field which is the
total number of cores available system-wide. Utilization is measured for
the main process, its descendents, the main process and its descendents
combined, and CPU utilization across the entire system. The core percent
is the sum of the percentages of all the cores being used. The CPU
percent is that divided by the system core count. The max percent is the
highest percentage detected through the duration of tracking while the
mean percent is the average of all the percentages detected over that
duration. The CPU utilization concludes with the maximum number of
threads used at any time for the main process and the sum of the threads
used across its descendent processes and combined.

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
       System capacity: 67254.17
       System: 5721.395
       Main:
          Total RSS: 850.399
          Private RSS: 634.077
          Shared RSS: 216.547
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 858.763
          Private RSS: 642.445
          Shared RSS: 216.527
    Max GPU RAM:
       Unit: megabytes
       System capacity: 16376.0
       System: 727.0
       Main: 506.0
       Descendents: 0.0
       Combined: 506.0
    CPU utilization:
       System core count: 12
       System:
          Max core percent: 148.9
          Max CPU percent: 12.408
          Mean core percent: 124.7
          Mean CPU percent: 10.392
       Main:
          Max core percent: 99.9
          Max CPU percent: 8.325
          Mean core percent: 97.533
          Mean CPU percent: 8.128
       Descendents:
          Max core percent: 0.0
          Max CPU percent: 0.0
          Mean core percent: 0.0
          Mean CPU percent: 0.0
       Combined:
          Max core percent: 99.9
          Max CPU percent: 8.325
          Mean core percent: 97.533
          Mean CPU percent: 8.128
       Main number of threads: 15
       Descendents number of threads: 0
       Combined number of threads: 15
    Compute time:
       Unit: seconds
       Time: 2.52


The same information as the text format can be provided as a dictionary
via the ``to_json()`` method of the ``Tracker``.

.. code:: python3

    import json
    print(json.dumps(tracker.to_json(), indent=1))


.. code:: none

    {
     "max_ram": {
      "unit": "megabytes",
      "system_capacity": 67254.1696,
      "system": 5721.3952,
      "main": {
       "total_rss": 850.399232,
       "private_rss": 634.077184,
       "shared_rss": 216.547328
      },
      "descendents": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 858.7632639999999,
       "private_rss": 642.445312,
       "shared_rss": 216.526848
      }
     },
     "max_gpu_ram": {
      "unit": "megabytes",
      "system_capacity": 16376.0,
      "system": 727.0,
      "main": 506.0,
      "descendents": 0.0,
      "combined": 506.0
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "system": {
       "max_core_percent": 148.90000000000003,
       "max_cpu_percent": 12.408333333333337,
       "mean_core_percent": 124.70000000000003,
       "mean_cpu_percent": 10.39166666666667
      },
      "main": {
       "max_core_percent": 99.9,
       "max_cpu_percent": 8.325000000000001,
       "mean_core_percent": 97.53333333333335,
       "mean_cpu_percent": 8.127777777777778
      },
      "descendents": {
       "max_core_percent": 0.0,
       "max_cpu_percent": 0.0,
       "mean_core_percent": 0.0,
       "mean_cpu_percent": 0.0
      },
      "combined": {
       "max_core_percent": 99.9,
       "max_cpu_percent": 8.325000000000001,
       "mean_core_percent": 97.53333333333335,
       "mean_cpu_percent": 8.127777777777778
      },
      "main_n_threads": 15,
      "descendents_n_threads": 0,
      "combined_n_threads": 15
     },
     "compute_time": {
      "unit": "seconds",
      "time": 2.5198354721069336
     }
    }


Using Python data classes, the ``Tracker`` class additionally has a
``resource_usage`` attribute containing fields that provide the usage
information for each individual computational resource.

.. code:: python3

    tracker.resource_usage.max_ram




.. code:: none

    MaxRAM(unit='megabytes', system_capacity=67254.1696, system=5721.3952, main=RSSValues(total_rss=850.399232, private_rss=634.077184, shared_rss=216.547328), descendents=RSSValues(total_rss=0.0, private_rss=0.0, shared_rss=0.0), combined=RSSValues(total_rss=858.7632639999999, private_rss=642.445312, shared_rss=216.526848))



.. code:: python3

    tracker.resource_usage.max_ram.unit




.. code:: none

    'megabytes'



.. code:: python3

    tracker.resource_usage.max_ram.main




.. code:: none

    RSSValues(total_rss=850.399232, private_rss=634.077184, shared_rss=216.547328)



.. code:: python3

    tracker.resource_usage.max_ram.main.total_rss




.. code:: none

    850.399232



.. code:: python3

    tracker.resource_usage.max_gpu_ram




.. code:: none

    MaxGPURAM(unit='megabytes', system_capacity=16376.0, system=727.0, main=506.0, descendents=0.0, combined=506.0)



.. code:: python3

    tracker.resource_usage.compute_time




.. code:: none

    ComputeTime(unit='seconds', time=2.5198354721069336)



Sometimes the code can fail. In order to collect the resource usage up
to the point of failure, use a try/except block like so:

.. code:: python3

    try:
        with gput.Tracker() as tracker:
            example_function()
            raise RuntimeError('AN ERROR')
    except Exception as error:
        print(f'The following error occured while tracking: {error}')
    finally:
        print(tracker.resource_usage.max_gpu_ram.main)


.. code:: none

    The following error occured while tracking: AN ERROR
    0.506


Below is an example of using a child process. Notice the descendents
fields are now non-zero.

.. code:: python3

    import multiprocessing as mp
    ctx = mp.get_context(method='spawn')
    child_process = ctx.Process(target=example_function)
    with gput.Tracker() as tracker:
        child_process.start()
        example_function()
        child_process.join()
        child_process.close()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 5.938
       Main:
          Total RSS: 0.798
          Private RSS: 0.491
          Shared RSS: 0.311
       Descendents:
          Total RSS: 0.85
          Private RSS: 0.728
          Shared RSS: 0.122
       Combined:
          Total RSS: 1.451
          Private RSS: 1.144
          Shared RSS: 0.311
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.043
       Main: 0.506
       Descendents: 0.314
       Combined: 0.82
    CPU utilization:
       System core count: 12
       System:
          Max core percent: 225.5
          Max CPU percent: 18.792
          Mean core percent: 187.575
          Mean CPU percent: 15.631
       Main:
          Max core percent: 99.6
          Max CPU percent: 8.3
          Mean core percent: 74.15
          Mean CPU percent: 6.179
       Descendents:
          Max core percent: 101.2
          Max CPU percent: 8.433
          Mean core percent: 74.125
          Mean CPU percent: 6.177
       Combined:
          Max core percent: 198.7
          Max CPU percent: 16.558
          Mean core percent: 148.275
          Mean CPU percent: 12.356
       Main number of threads: 15
       Descendents number of threads: 5
       Combined number of threads: 20
    Compute time:
       Unit: hours
       Time: 0.001


CLI
---

The ``gpu-tracker`` package also comes with a commandline interface that
can track the computational-resource-usage of any shell command, not
just Python code. Entering ``gpu-tracker -h`` in a shell will show the
help message.

.. code:: none

    $ gpu-tracker -h


.. code:: none

    Tracks the computational resource usage (RAM, GPU RAM, and compute time) of a process corresponding to a given shell command.
    
    Usage:
        gpu-tracker -h | --help
        gpu-tracker -v | --version
        gpu-tracker --execute=<command> [--output=<output>] [--format=<format>] [--st=<sleep-time>] [--ru=<ram-unit>] [--gru=<gpu-ram-unit>] [--tu=<time-unit>] [--disable-logs]
    
    Options:
        -h --help               Show this help message and exit.
        -v --version            Show package version and exit.
        -e --execute=<command>  The command to run along with its arguments all within quotes e.g. "ls -l -a".
        -o --output=<output>    File path to store the computational-resource-usage measurements. If not set, prints measurements to the screen.
        -f --format=<format>    File format of the output. Either 'json' or 'text'. Defaults to 'text'.
        --st=<sleep-time>       The number of seconds to sleep in between usage-collection iterations.
        --ru=<ram-unit>         One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        --gru=<gpu-ram-unit>    One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        --tu=<time-unit>        One of 'seconds', 'minutes', 'hours', or 'days'.
        --disable-logs          If set, warnings are suppressed during tracking. Otherwise, the Tracker logs warnings as usual.


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
       System: 5.964
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendents:
          Total RSS: 0.847
          Private RSS: 0.724
          Shared RSS: 0.122
       Combined:
          Total RSS: 0.856
          Private RSS: 0.733
          Shared RSS: 0.123
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.043
       Main: 0.0
       Descendents: 0.314
       Combined: 0.314
    CPU utilization:
       System core count: 12
       System:
          Max core percent: 177.6
          Max CPU percent: 14.8
          Mean core percent: 134.375
          Mean CPU percent: 11.198
       Main:
          Max core percent: 0.0
          Max CPU percent: 0.0
          Mean core percent: 0.0
          Mean CPU percent: 0.0
       Descendents:
          Max core percent: 100.4
          Max CPU percent: 8.367
          Mean core percent: 95.45
          Mean CPU percent: 7.954
       Combined:
          Max core percent: 100.4
          Max CPU percent: 8.367
          Mean core percent: 95.45
          Mean CPU percent: 7.954
       Main number of threads: 1
       Descendents number of threads: 4
       Combined number of threads: 5
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
       System capacity: 67254.17
       System: 5784.379
       Main:
          Total RSS: 3.076
          Private RSS: 0.324
          Shared RSS: 2.753
       Descendents:
          Total RSS: 838.545
          Private RSS: 716.681
          Shared RSS: 121.864
       Combined:
          Total RSS: 847.249
          Private RSS: 724.492
          Shared RSS: 122.757
    Max GPU RAM:
       Unit: megabytes
       System capacity: 16376.0
       System: 1043.0
       Main: 0.0
       Descendents: 314.0
       Combined: 314.0
    CPU utilization:
       System core count: 12
       System:
          Max core percent: 188.7
          Max CPU percent: 15.725
          Mean core percent: 136.45
          Mean CPU percent: 11.371
       Main:
          Max core percent: 0.0
          Max CPU percent: 0.0
          Mean core percent: 0.0
          Mean CPU percent: 0.0
       Descendents:
          Max core percent: 96.2
          Max CPU percent: 8.017
          Mean core percent: 94.55
          Mean CPU percent: 7.879
       Combined:
          Max core percent: 96.2
          Max CPU percent: 8.017
          Mean core percent: 94.55
          Mean CPU percent: 7.879
       Main number of threads: 1
       Descendents number of threads: 4
       Combined number of threads: 5
    Compute time:
       Unit: seconds
       Time: 3.566


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
       System: 5.584
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendents:
          Total RSS: 0.853
          Private RSS: 0.731
          Shared RSS: 0.122
       Combined:
          Total RSS: 0.862
          Private RSS: 0.739
          Shared RSS: 0.123
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.043
       Main: 0.0
       Descendents: 0.314
       Combined: 0.314
    CPU utilization:
       System core count: 12
       System:
          Max core percent: 187.6
          Max CPU percent: 15.633
          Mean core percent: 137.675
          Mean CPU percent: 11.473
       Main:
          Max core percent: 0.0
          Max CPU percent: 0.0
          Mean core percent: 0.0
          Mean CPU percent: 0.0
       Descendents:
          Max core percent: 101.3
          Max CPU percent: 8.442
          Mean core percent: 97.675
          Mean CPU percent: 8.14
       Combined:
          Max core percent: 101.3
          Max CPU percent: 8.442
          Mean core percent: 97.675
          Mean CPU percent: 8.14
       Main number of threads: 1
       Descendents number of threads: 4
       Combined number of threads: 5
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
      "system_capacity": 67.2541696,
      "system": 5.720379392000001,
      "main": {
       "total_rss": 0.003084288,
       "private_rss": 0.00031948800000000004,
       "shared_rss": 0.0027648
      },
      "descendents": {
       "total_rss": 0.854237184,
       "private_rss": 0.73218048,
       "shared_rss": 0.122056704
      },
      "combined": {
       "total_rss": 0.863256576,
       "private_rss": 0.7403069440000001,
       "shared_rss": 0.122949632
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "system_capacity": 16.376,
      "system": 1.043,
      "main": 0.0,
      "descendents": 0.314,
      "combined": 0.314
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "system": {
       "max_core_percent": 260.00000000000006,
       "max_cpu_percent": 21.66666666666667,
       "mean_core_percent": 159.35000000000002,
       "mean_cpu_percent": 13.279166666666669
      },
      "main": {
       "max_core_percent": 0.0,
       "max_cpu_percent": 0.0,
       "mean_core_percent": 0.0,
       "mean_cpu_percent": 0.0
      },
      "descendents": {
       "max_core_percent": 102.9,
       "max_cpu_percent": 8.575000000000001,
       "mean_core_percent": 97.475,
       "mean_cpu_percent": 8.122916666666667
      },
      "combined": {
       "max_core_percent": 102.9,
       "max_cpu_percent": 8.575000000000001,
       "mean_core_percent": 97.475,
       "mean_cpu_percent": 8.122916666666667
      },
      "main_n_threads": 1,
      "descendents_n_threads": 4,
      "combined_n_threads": 5
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.001005272732840644
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
      "system_capacity": 67.2541696,
      "system": 5.560373248,
      "main": {
       "total_rss": 0.002957312,
       "private_rss": 0.000323584,
       "shared_rss": 0.002633728
      },
      "descendents": {
       "total_rss": 0.848539648,
       "private_rss": 0.726519808,
       "shared_rss": 0.12201984
      },
      "combined": {
       "total_rss": 0.857731072,
       "private_rss": 0.734818304,
       "shared_rss": 0.122912768
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "system_capacity": 16.376,
      "system": 1.043,
      "main": 0.0,
      "descendents": 0.314,
      "combined": 0.314
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "system": {
       "max_core_percent": 192.5,
       "max_cpu_percent": 16.041666666666668,
       "mean_core_percent": 154.22500000000002,
       "mean_cpu_percent": 12.852083333333335
      },
      "main": {
       "max_core_percent": 0.0,
       "max_cpu_percent": 0.0,
       "mean_core_percent": 0.0,
       "mean_cpu_percent": 0.0
      },
      "descendents": {
       "max_core_percent": 104.1,
       "max_cpu_percent": 8.674999999999999,
       "mean_core_percent": 97.7,
       "mean_cpu_percent": 8.141666666666667
      },
      "combined": {
       "max_core_percent": 104.1,
       "max_cpu_percent": 8.674999999999999,
       "mean_core_percent": 97.7,
       "mean_cpu_percent": 8.141666666666667
      },
      "main_n_threads": 1,
      "descendents_n_threads": 4,
      "combined_n_threads": 5
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.000995432734489441
     }
    }
