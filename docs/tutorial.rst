.. _tutorial-label:

Tutorial
========

API
---

The ``gpu_tracker`` package provides the ``Tracker`` class which uses a
subprocess to measure computational resource usage, namely the compute
time, maximum CPU utilization, mean CPU utilization, maximum RAM used,
maximum GPU utilization, mean GPU utilization, and maximum GPU RAM used.
The ``start()`` method starts this process which tracks usage in the
background. After calling ``start()``, one can write the code for which
resource usage is measured, followed by calling the ``stop()`` method.
The compute time will be the time from the call to ``start()`` to the
call to ``stop()`` and the RAM, GPU RAM, CPU utilization, and GPU
utilization quantities will be the respective computational resources
used by the code that’s in between ``start()`` and ``stop()``.

.. code:: python3

    import gpu_tracker as gput
    from example_module import example_function

.. code:: python3

    tracker = gput.Tracker(n_expected_cores=1, sleep_time=0.1)
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
       System capacity: 63.088
       System: 1.899
       Main:
          Total RSS: 0.914
          Private RSS: 0.753
          Shared RSS: 0.161
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.883
          Private RSS: 0.723
          Shared RSS: 0.161
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 2.048
       System: 0.353
       Main: 0.277
       Descendents: 0.0
       Combined: 0.277
    CPU utilization:
       System core count: 12
       Number of expected cores: 1
       System:
          Max sum percent: 169.7
          Max hardware percent: 14.142
          Mean sum percent: 150.183
          Mean hardware percent: 12.515
       Main:
          Max sum percent: 101.2
          Max hardware percent: 101.2
          Mean sum percent: 93.158
          Mean hardware percent: 93.158
       Descendents:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Combined:
          Max sum percent: 101.2
          Max hardware percent: 101.2
          Mean sum percent: 93.158
          Mean hardware percent: 93.158
       Main number of threads: 24
       Descendents number of threads: 0
       Combined number of threads: 24
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 4.0
          Max hardware percent: 4.0
          Mean sum percent: 0.333
          Mean hardware percent: 0.333
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
combined, and CPU utilization across the entire system. The sum percent
is the sum of the percentages of all the cores being used. The hardware
percent is that divided by the expected number of cores being used
i.e. the optional ``n_expected_cores`` parameter (defaults to the number
of cores in the entire system) for the main, descendents, and combined
measurements. For the system measurements, hardware percent is divided
by the total number of cores in the system regardless of the value of
``n_expected_cores``. The max percent is the highest percentage detected
through the duration of tracking while the mean percent is the average
of all the percentages detected over that duration. The CPU utilization
concludes with the maximum number of threads used at any time for the
main process and the sum of the threads used across its descendent
processes and combined.

The GPU utilization is similar to the CPU utilization but rather than
being based on utilization of processes, it can only measure the
utilization percentages of the GPUs themselves, regardless of what
processes are using them. To ameliorate this limitation, the optional
``gpu_uuids`` parameter can be set to specify which GPUs to measure
utilization for (defaults to all the GPUs in the system). The system GPU
count is the total number of GPUs in the system. The sum percent is the
sum of all the percentages of these GPUs and the hardware percent is
that divided by the expected number of GPUs being used
(i.e. ``len(gpu_uuids)``). Likewise with CPU utilization, the max and
mean of both the sum and hardware percentages are provided.

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

    with gput.Tracker(ram_unit='megabytes', gpu_ram_unit='megabytes', time_unit='seconds', sleep_time=0.1) as tracker:
        example_function()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: megabytes
       System capacity: 63088.23
       System: 2399.92
       Main:
          Total RSS: 890.704
          Private RSS: 674.058
          Shared RSS: 216.924
       Descendents:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 901.263
          Private RSS: 684.618
          Shared RSS: 216.678
    Max GPU RAM:
       Unit: megabytes
       System capacity: 2048.0
       System: 353.0
       Main: 277.0
       Descendents: 0.0
       Combined: 277.0
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 164.3
          Max hardware percent: 13.692
          Mean sum percent: 152.325
          Mean hardware percent: 12.694
       Main:
          Max sum percent: 102.6
          Max hardware percent: 8.55
          Mean sum percent: 91.258
          Mean hardware percent: 7.605
       Descendents:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Combined:
          Max sum percent: 102.6
          Max hardware percent: 8.55
          Mean sum percent: 91.258
          Mean hardware percent: 7.605
       Main number of threads: 24
       Descendents number of threads: 0
       Combined number of threads: 24
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 6.0
          Max hardware percent: 6.0
          Mean sum percent: 0.5
          Mean hardware percent: 0.5
    Compute time:
       Unit: seconds
       Time: 3.346


The same information as the text format can be provided as a dictionary
via the ``to_json()`` method of the ``Tracker``.

.. code:: python3

    import json
    print(json.dumps(tracker.to_json(), indent=1))


.. code:: none

    {
     "max_ram": {
      "unit": "megabytes",
      "system_capacity": 63088.2304,
      "system": 2399.9201279999997,
      "main": {
       "total_rss": 890.7038719999999,
       "private_rss": 674.05824,
       "shared_rss": 216.92416
      },
      "descendents": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 901.2633599999999,
       "private_rss": 684.6177279999999,
       "shared_rss": 216.67839999999998
      }
     },
     "max_gpu_ram": {
      "unit": "megabytes",
      "system_capacity": 2048.0,
      "system": 353.0,
      "main": 277.0,
      "descendents": 0.0,
      "combined": 277.0
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 164.3,
       "max_hardware_percent": 13.691666666666668,
       "mean_sum_percent": 152.325,
       "mean_hardware_percent": 12.693750000000001
      },
      "main": {
       "max_sum_percent": 102.6,
       "max_hardware_percent": 8.549999999999999,
       "mean_sum_percent": 91.25833333333334,
       "mean_hardware_percent": 7.604861111111112
      },
      "descendents": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "combined": {
       "max_sum_percent": 102.6,
       "max_hardware_percent": 8.549999999999999,
       "mean_sum_percent": 91.25833333333334,
       "mean_hardware_percent": 7.604861111111112
      },
      "main_n_threads": 24,
      "descendents_n_threads": 0,
      "combined_n_threads": 24
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 6.0,
       "max_hardware_percent": 6.0,
       "mean_sum_percent": 0.5,
       "mean_hardware_percent": 0.5
      }
     },
     "compute_time": {
      "unit": "seconds",
      "time": 3.345628023147583
     }
    }


Using Python data classes, the ``Tracker`` class additionally has a
``resource_usage`` attribute containing fields that provide the usage
information for each individual computational resource.

.. code:: python3

    tracker.resource_usage.max_ram




.. code:: none

    MaxRAM(unit='megabytes', system_capacity=63088.2304, system=2399.9201279999997, main=RSSValues(total_rss=890.7038719999999, private_rss=674.05824, shared_rss=216.92416), descendents=RSSValues(total_rss=0.0, private_rss=0.0, shared_rss=0.0), combined=RSSValues(total_rss=901.2633599999999, private_rss=684.6177279999999, shared_rss=216.67839999999998))



.. code:: python3

    tracker.resource_usage.max_ram.unit




.. code:: none

    'megabytes'



.. code:: python3

    tracker.resource_usage.max_ram.main




.. code:: none

    RSSValues(total_rss=890.7038719999999, private_rss=674.05824, shared_rss=216.92416)



.. code:: python3

    tracker.resource_usage.max_ram.main.total_rss




.. code:: none

    890.7038719999999



.. code:: python3

    tracker.resource_usage.max_gpu_ram




.. code:: none

    MaxGPURAM(unit='megabytes', system_capacity=2048.0, system=353.0, main=277.0, descendents=0.0, combined=277.0)



.. code:: python3

    tracker.resource_usage.compute_time




.. code:: none

    ComputeTime(unit='seconds', time=3.345628023147583)



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
    0.277


Below is an example of using a child process. Notice the descendents
fields are now non-zero.

.. code:: python3

    import multiprocessing as mp
    ctx = mp.get_context(method='spawn')
    child_process = ctx.Process(target=example_function)
    with gput.Tracker(n_expected_cores=2, sleep_time=0.2) as tracker:
        child_process.start()
        example_function()
        child_process.join()
    child_process.close()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 63.088
       System: 2.877
       Main:
          Total RSS: 0.844
          Private RSS: 0.525
          Shared RSS: 0.319
       Descendents:
          Total RSS: 0.831
          Private RSS: 0.704
          Shared RSS: 0.127
       Combined:
          Total RSS: 1.462
          Private RSS: 1.148
          Shared RSS: 0.32
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 2.048
       System: 0.631
       Main: 0.277
       Descendents: 0.277
       Combined: 0.554
    CPU utilization:
       System core count: 12
       Number of expected cores: 2
       System:
          Max sum percent: 398.9
          Max hardware percent: 33.242
          Mean sum percent: 222.255
          Mean hardware percent: 18.521
       Main:
          Max sum percent: 103.8
          Max hardware percent: 51.9
          Mean sum percent: 66.009
          Mean hardware percent: 33.005
       Descendents:
          Max sum percent: 308.5
          Max hardware percent: 154.25
          Mean sum percent: 117.109
          Mean hardware percent: 58.555
       Combined:
          Max sum percent: 409.2
          Max hardware percent: 204.6
          Mean sum percent: 183.118
          Mean hardware percent: 91.559
       Main number of threads: 24
       Descendents number of threads: 16
       Combined number of threads: 40
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 6.0
          Max hardware percent: 6.0
          Mean sum percent: 0.545
          Mean hardware percent: 0.545
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

    Tracks the computational resource usage (RAM, GPU RAM, CPU utilization, GPU utilization, and compute time) of a process corresponding to a given shell command.
    
    Usage:
        gpu-tracker -h | --help
        gpu-tracker -v | --version
        gpu-tracker --execute=<command> [--output=<output>] [--format=<format>] [--st=<sleep-time>] [--ru=<ram-unit>] [--gru=<gpu-ram-unit>] [--tu=<time-unit>] [--nec=<num-cores>] [--guuids=<gpu-uuids>] [--disable-logs]
    
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
        --nec=<num-cores>       The number of cores expected to be used. Defaults to the number of cores in the entire operating system.
        --guuids=<gpu-uuids>    Comma separated list of the UUIDs of the GPUs for which to track utilization e.g. gpu-uuid1,gpu-uuid2,etc. Defaults to all the GPUs in the system.
        --disable-logs          If set, warnings are suppressed during tracking. Otherwise, the Tracker logs warnings as usual.


The ``-e`` or ``--execute`` is a required option where the desired shell
command is provided, with both the command and its proceeding arguments
surrounded by quotes. Below is an example of running the ``bash``
command with an argument of ``example-script.sh``. When the command
completes, its status code is reported.

.. code:: none

    $ gpu-tracker -e "bash example-script.sh" --st=0.3


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: gigabytes
       System capacity: 63.088
       System: 2.3
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendents:
          Total RSS: 0.917
          Private RSS: 0.905
          Shared RSS: 0.012
       Combined:
          Total RSS: 0.925
          Private RSS: 0.912
          Shared RSS: 0.013
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 2.048
       System: 0.193
       Main: 0.0
       Descendents: 0.117
       Combined: 0.117
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 309.5
          Max hardware percent: 25.792
          Mean sum percent: 159.073
          Mean hardware percent: 13.256
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendents:
          Max sum percent: 493.1
          Max hardware percent: 41.092
          Mean sum percent: 134.427
          Mean hardware percent: 11.202
       Combined:
          Max sum percent: 493.1
          Max hardware percent: 41.092
          Mean sum percent: 134.427
          Mean hardware percent: 11.202
       Main number of threads: 1
       Descendents number of threads: 15
       Combined number of threads: 16
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 4.0
          Max hardware percent: 4.0
          Mean sum percent: 0.364
          Mean hardware percent: 0.364
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

    $ gpu-tracker -e 'bash example-script.sh' --tu=seconds --gru=megabytes --ru=megabytes --st=0.2


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: megabytes
       System capacity: 63088.23
       System: 2242.593
       Main:
          Total RSS: 3.039
          Private RSS: 0.315
          Shared RSS: 2.724
       Descendents:
          Total RSS: 832.487
          Private RSS: 705.831
          Shared RSS: 126.657
       Combined:
          Total RSS: 841.482
          Private RSS: 713.867
          Shared RSS: 127.992
    Max GPU RAM:
       Unit: megabytes
       System capacity: 2048.0
       System: 631.0
       Main: 0.0
       Descendents: 277.0
       Combined: 277.0
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 362.6
          Max hardware percent: 30.217
          Mean sum percent: 156.853
          Mean hardware percent: 13.071
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendents:
          Max sum percent: 512.8
          Max hardware percent: 42.733
          Mean sum percent: 120.333
          Mean hardware percent: 10.028
       Combined:
          Max sum percent: 512.8
          Max hardware percent: 42.733
          Mean sum percent: 120.333
          Mean hardware percent: 10.028
       Main number of threads: 1
       Descendents number of threads: 15
       Combined number of threads: 16
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 4.0
          Max hardware percent: 4.0
          Mean sum percent: 0.267
          Mean hardware percent: 0.267
    Compute time:
       Unit: seconds
       Time: 4.931


By default, the computational-resource-usage statistics are printed to
the screen. The ``-o`` or ``--output`` option can be specified to store
that same content in a file.

.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' -o out.txt --st=0.2


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.txt


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 63.088
       System: 2.683
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendents:
          Total RSS: 0.843
          Private RSS: 0.717
          Shared RSS: 0.127
       Combined:
          Total RSS: 0.852
          Private RSS: 0.725
          Shared RSS: 0.128
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 2.048
       System: 0.631
       Main: 0.0
       Descendents: 0.277
       Combined: 0.277
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 383.8
          Max hardware percent: 31.983
          Mean sum percent: 166.507
          Mean hardware percent: 13.876
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendents:
          Max sum percent: 528.4
          Max hardware percent: 44.033
          Mean sum percent: 128.014
          Mean hardware percent: 10.668
       Combined:
          Max sum percent: 528.4
          Max hardware percent: 44.033
          Mean sum percent: 128.014
          Mean hardware percent: 10.668
       Main number of threads: 1
       Descendents number of threads: 15
       Combined number of threads: 16
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 7.0
          Max hardware percent: 7.0
          Mean sum percent: 0.643
          Mean hardware percent: 0.643
    Compute time:
       Unit: hours
       Time: 0.001

By default, the format of the output is “text”. The ``-f`` or
``--format`` option can specify the format to be “json” instead.

.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' -f json --st=0.2


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    {
     "max_ram": {
      "unit": "gigabytes",
      "system_capacity": 63.0882304,
      "system": 3.111936,
      "main": {
       "total_rss": 0.003059712,
       "private_rss": 0.000339968,
       "shared_rss": 0.002719744
      },
      "descendents": {
       "total_rss": 0.846565376,
       "private_rss": 0.7198023680000001,
       "shared_rss": 0.12713984
      },
      "combined": {
       "total_rss": 0.8552325120000001,
       "private_rss": 0.727576576,
       "shared_rss": 0.12803276800000002
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "system_capacity": 2.048,
      "system": 0.631,
      "main": 0.0,
      "descendents": 0.277,
      "combined": 0.277
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 384.5999999999999,
       "max_hardware_percent": 32.04999999999999,
       "mean_sum_percent": 167.49285714285716,
       "mean_hardware_percent": 13.957738095238097
      },
      "main": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "descendents": {
       "max_sum_percent": 526.0,
       "max_hardware_percent": 43.833333333333336,
       "mean_sum_percent": 128.65,
       "mean_hardware_percent": 10.720833333333333
      },
      "combined": {
       "max_sum_percent": 526.0,
       "max_hardware_percent": 43.833333333333336,
       "mean_sum_percent": 128.65,
       "mean_hardware_percent": 10.720833333333333
      },
      "main_n_threads": 1,
      "descendents_n_threads": 15,
      "combined_n_threads": 16
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 7.0,
       "max_hardware_percent": 7.0,
       "mean_sum_percent": 0.5,
       "mean_hardware_percent": 0.5
      }
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.0012672905127207438
     }
    }


.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' -f json -o out.json --st=0.3


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.json


.. code:: none

    {
     "max_ram": {
      "unit": "gigabytes",
      "system_capacity": 63.0882304,
      "system": 2.878910464,
      "main": {
       "total_rss": 0.0029777920000000004,
       "private_rss": 0.00031948800000000004,
       "shared_rss": 0.0026583040000000002
      },
      "descendents": {
       "total_rss": 0.8333844480000001,
       "private_rss": 0.7066091520000001,
       "shared_rss": 0.127152128
      },
      "combined": {
       "total_rss": 0.841486336,
       "private_rss": 0.713818112,
       "shared_rss": 0.12804505600000002
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "system_capacity": 2.048,
      "system": 0.631,
      "main": 0.0,
      "descendents": 0.277,
      "combined": 0.277
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 306.09999999999997,
       "max_hardware_percent": 25.50833333333333,
       "mean_sum_percent": 161.4272727272727,
       "mean_hardware_percent": 13.452272727272724
      },
      "main": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "descendents": {
       "max_sum_percent": 440.2,
       "max_hardware_percent": 36.68333333333333,
       "mean_sum_percent": 128.27272727272728,
       "mean_hardware_percent": 10.68939393939394
      },
      "combined": {
       "max_sum_percent": 440.2,
       "max_hardware_percent": 36.68333333333333,
       "mean_sum_percent": 128.27272727272728,
       "mean_hardware_percent": 10.68939393939394
      },
      "main_n_threads": 1,
      "descendents_n_threads": 15,
      "combined_n_threads": 16
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 7.0,
       "max_hardware_percent": 7.0,
       "mean_sum_percent": 0.6363636363636364,
       "mean_hardware_percent": 0.6363636363636364
      }
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.0012816817230648465
     }
    }
