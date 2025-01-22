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
       System capacity: 67.254
       System: 2.001
       Main:
          Total RSS: 0.94
          Private RSS: 0.786
          Shared RSS: 0.165
       Descendants:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.94
          Private RSS: 0.786
          Shared RSS: 0.165
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 0.535
       Main: 0.314
       Descendants: 0.0
       Combined: 0.314
    CPU utilization:
       System core count: 12
       Number of expected cores: 1
       System:
          Max sum percent: 162.3
          Max hardware percent: 13.525
          Mean sum percent: 144.283
          Mean hardware percent: 12.024
       Main:
          Max sum percent: 101.4
          Max hardware percent: 101.4
          Mean sum percent: 96.7
          Mean hardware percent: 96.7
       Descendants:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Combined:
          Max sum percent: 101.4
          Max hardware percent: 101.4
          Mean sum percent: 96.7
          Mean hardware percent: 96.7
       Main number of threads: 15
       Descendants number of threads: 0
       Combined number of threads: 15
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 5.0
          Max hardware percent: 5.0
          Mean sum percent: 0.417
          Mean hardware percent: 0.417
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
by the summed usage of any descendant processes it may have (i.e. child
processes, grandchild processes, etc.), and combined usage which is the
sum of the main and its descendant processes. RAM is divided further to
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
processes and if it is, the reported GPU RAM of “descendant” and
“combined” may be an overestimation.

The CPU utilization includes the system core count field which is the
total number of cores available system-wide. Utilization is measured for
the main process, its descendants, the main process and its descendants
combined, and CPU utilization across the entire system. The sum percent
is the sum of the percentages of all the cores being used. The hardware
percent is that divided by the expected number of cores being used
i.e. the optional ``n_expected_cores`` parameter (defaults to the number
of cores in the entire system) for the main, descendants, and combined
measurements. For the system measurements, hardware percent is divided
by the total number of cores in the system regardless of the value of
``n_expected_cores``. The max percent is the highest percentage detected
through the duration of tracking while the mean percent is the average
of all the percentages detected over that duration. The CPU utilization
concludes with the maximum number of threads used at any time for the
main process and the sum of the threads used across its descendant
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

**NOTE** *The keywords “descendants” and “combined” in the output above
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
       System capacity: 67254.17
       System: 2336.362
       Main:
          Total RSS: 919.99
          Private RSS: 699.384
          Shared RSS: 230.269
       Descendants:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 919.99
          Private RSS: 699.384
          Shared RSS: 230.269
    Max GPU RAM:
       Unit: megabytes
       System capacity: 16376.0
       System: 727.0
       Main: 506.0
       Descendants: 0.0
       Combined: 506.0
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 166.5
          Max hardware percent: 13.875
          Mean sum percent: 144.55
          Mean hardware percent: 12.046
       Main:
          Max sum percent: 104.8
          Max hardware percent: 8.733
          Mean sum percent: 97.458
          Mean hardware percent: 8.122
       Descendants:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Combined:
          Max sum percent: 104.8
          Max hardware percent: 8.733
          Mean sum percent: 97.458
          Mean hardware percent: 8.122
       Main number of threads: 15
       Descendants number of threads: 0
       Combined number of threads: 15
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
    Compute time:
       Unit: seconds
       Time: 2.685


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
      "system": 2336.3624959999997,
      "main": {
       "total_rss": 919.9902719999999,
       "private_rss": 699.3838079999999,
       "shared_rss": 230.268928
      },
      "descendants": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 919.9902719999999,
       "private_rss": 699.3838079999999,
       "shared_rss": 230.268928
      }
     },
     "max_gpu_ram": {
      "unit": "megabytes",
      "system_capacity": 16376.0,
      "system": 727.0,
      "main": 506.0,
      "descendants": 0.0,
      "combined": 506.0
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 166.5,
       "max_hardware_percent": 13.875,
       "mean_sum_percent": 144.55,
       "mean_hardware_percent": 12.045833333333333
      },
      "main": {
       "max_sum_percent": 104.8,
       "max_hardware_percent": 8.733333333333333,
       "mean_sum_percent": 97.45833333333333,
       "mean_hardware_percent": 8.121527777777779
      },
      "descendants": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "combined": {
       "max_sum_percent": 104.8,
       "max_hardware_percent": 8.733333333333333,
       "mean_sum_percent": 97.45833333333333,
       "mean_hardware_percent": 8.121527777777779
      },
      "main_n_threads": 15,
      "descendants_n_threads": 0,
      "combined_n_threads": 15
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      }
     },
     "compute_time": {
      "unit": "seconds",
      "time": 2.684972047805786
     }
    }


Using Python data classes, the ``Tracker`` class additionally has a
``resource_usage`` attribute containing fields that provide the usage
information for each individual computational resource.

.. code:: python3

    tracker.resource_usage.max_ram




.. code:: none

    MaxRAM(unit='megabytes', system_capacity=67254.1696, system=2336.3624959999997, main=RSSValues(total_rss=919.9902719999999, private_rss=699.3838079999999, shared_rss=230.268928), descendants=RSSValues(total_rss=0.0, private_rss=0.0, shared_rss=0.0), combined=RSSValues(total_rss=919.9902719999999, private_rss=699.3838079999999, shared_rss=230.268928))



.. code:: python3

    tracker.resource_usage.max_ram.unit




.. code:: none

    'megabytes'



.. code:: python3

    tracker.resource_usage.max_ram.main




.. code:: none

    RSSValues(total_rss=919.9902719999999, private_rss=699.3838079999999, shared_rss=230.268928)



.. code:: python3

    tracker.resource_usage.max_ram.main.total_rss




.. code:: none

    919.9902719999999



.. code:: python3

    tracker.resource_usage.max_gpu_ram




.. code:: none

    MaxGPURAM(unit='megabytes', system_capacity=16376.0, system=727.0, main=506.0, descendants=0.0, combined=506.0)



.. code:: python3

    tracker.resource_usage.compute_time




.. code:: none

    ComputeTime(unit='seconds', time=2.684972047805786)



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


If you do not catch the error in your code or if tracking otherwise is
interrupted (e.g. you are debugging your code and you stop partway), the
``resource_usage`` attribute will not be set and that information will
not be able to be obtained in memory. In such a case, the
``resource_usage`` attribute will be stored in a hidden pickle file in
the working directory with a randomly generated name. Its file path can
be optionally overriden with the ``resource_usage_file`` parameter.

.. code:: python3

    tracker = gput.Tracker(resource_usage_file='path/to/my-file.pkl')

Below is an example of using a child process. Notice the descendants
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
       System capacity: 67.254
       System: 3.033
       Main:
          Total RSS: 0.865
          Private RSS: 0.55
          Shared RSS: 0.32
       Descendants:
          Total RSS: 0.854
          Private RSS: 0.737
          Shared RSS: 0.118
       Combined:
          Total RSS: 1.437
          Private RSS: 1.125
          Shared RSS: 0.32
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.235
       Main: 0.506
       Descendants: 0.506
       Combined: 1.012
    CPU utilization:
       System core count: 12
       Number of expected cores: 2
       System:
          Max sum percent: 456.5
          Max hardware percent: 38.042
          Mean sum percent: 216.675
          Mean hardware percent: 18.056
       Main:
          Max sum percent: 102.6
          Max hardware percent: 51.3
          Mean sum percent: 66.65
          Mean hardware percent: 33.325
       Descendants:
          Max sum percent: 175.8
          Max hardware percent: 87.9
          Mean sum percent: 105.392
          Mean hardware percent: 52.696
       Combined:
          Max sum percent: 278.4
          Max hardware percent: 139.2
          Mean sum percent: 172.042
          Mean hardware percent: 86.021
       Main number of threads: 15
       Descendants number of threads: 13
       Combined number of threads: 28
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 8.0
          Max hardware percent: 8.0
          Mean sum percent: 1.333
          Mean hardware percent: 1.333
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
       System capacity: 67.254
       System: 2.896
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendants:
          Total RSS: 0.877
          Private RSS: 0.759
          Shared RSS: 0.118
       Combined:
          Total RSS: 0.878
          Private RSS: 0.759
          Shared RSS: 0.119
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.043
       Main: 0.0
       Descendants: 0.314
       Combined: 0.314
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 324.1
          Max hardware percent: 27.008
          Mean sum percent: 164.91
          Mean hardware percent: 13.743
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendants:
          Max sum percent: 361.4
          Max hardware percent: 30.117
          Mean sum percent: 123.42
          Mean hardware percent: 10.285
       Combined:
          Max sum percent: 361.4
          Max hardware percent: 30.117
          Mean sum percent: 123.42
          Mean hardware percent: 10.285
       Main number of threads: 1
       Descendants number of threads: 12
       Combined number of threads: 13
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
    Compute time:
       Unit: hours
       Time: 0.001


*Notice that the RAM and GPU RAM usage primarily takes place in the
descendant processes since the bash command itself calls the commands
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
       System capacity: 67254.17
       System: 2420.457
       Main:
          Total RSS: 3.109
          Private RSS: 0.319
          Shared RSS: 2.789
       Descendants:
          Total RSS: 849.125
          Private RSS: 731.435
          Shared RSS: 118.125
       Combined:
          Total RSS: 850.338
          Private RSS: 731.754
          Shared RSS: 119.017
    Max GPU RAM:
       Unit: megabytes
       System capacity: 16376.0
       System: 1235.0
       Main: 0.0
       Descendants: 506.0
       Combined: 506.0
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 316.4
          Max hardware percent: 26.367
          Mean sum percent: 168.077
          Mean hardware percent: 14.006
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendants:
          Max sum percent: 517.3
          Max hardware percent: 43.108
          Mean sum percent: 130.623
          Mean hardware percent: 10.885
       Combined:
          Max sum percent: 517.3
          Max hardware percent: 43.108
          Mean sum percent: 130.623
          Mean hardware percent: 10.885
       Main number of threads: 1
       Descendants number of threads: 12
       Combined number of threads: 13
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 5.0
          Max hardware percent: 5.0
          Mean sum percent: 0.462
          Mean hardware percent: 0.462
    Compute time:
       Unit: seconds
       Time: 3.995


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
       System capacity: 67.254
       System: 2.43
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendants:
          Total RSS: 0.884
          Private RSS: 0.766
          Shared RSS: 0.118
       Combined:
          Total RSS: 0.885
          Private RSS: 0.766
          Shared RSS: 0.119
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.043
       Main: 0.0
       Descendants: 0.314
       Combined: 0.314
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 405.0
          Max hardware percent: 33.75
          Mean sum percent: 165.357
          Mean hardware percent: 13.78
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendants:
          Max sum percent: 573.7
          Max hardware percent: 47.808
          Mean sum percent: 124.871
          Mean hardware percent: 10.406
       Combined:
          Max sum percent: 573.7
          Max hardware percent: 47.808
          Mean sum percent: 124.871
          Mean hardware percent: 10.406
       Main number of threads: 1
       Descendants number of threads: 12
       Combined number of threads: 13
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 5.0
          Max hardware percent: 5.0
          Mean sum percent: 0.357
          Mean hardware percent: 0.357
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
      "system_capacity": 67.2541696,
      "system": 2.5132195840000002,
      "main": {
       "total_rss": 0.00311296,
       "private_rss": 0.000323584,
       "shared_rss": 0.002789376
      },
      "descendants": {
       "total_rss": 0.8446238720000001,
       "private_rss": 0.7268597760000001,
       "shared_rss": 0.11776409600000001
      },
      "combined": {
       "total_rss": 0.8458403840000001,
       "private_rss": 0.7271833600000001,
       "shared_rss": 0.11865702400000001
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "system_capacity": 16.376,
      "system": 1.235,
      "main": 0.0,
      "descendants": 0.506,
      "combined": 0.506
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 316.3,
       "max_hardware_percent": 26.358333333333334,
       "mean_sum_percent": 167.90769230769232,
       "mean_hardware_percent": 13.992307692307692
      },
      "main": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "descendants": {
       "max_sum_percent": 527.1,
       "max_hardware_percent": 43.925000000000004,
       "mean_sum_percent": 130.81538461538463,
       "mean_hardware_percent": 10.90128205128205
      },
      "combined": {
       "max_sum_percent": 527.1,
       "max_hardware_percent": 43.925000000000004,
       "mean_sum_percent": 130.81538461538463,
       "mean_hardware_percent": 10.90128205128205
      },
      "main_n_threads": 1,
      "descendants_n_threads": 12,
      "combined_n_threads": 13
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 5.0,
       "max_hardware_percent": 5.0,
       "mean_sum_percent": 0.38461538461538464,
       "mean_hardware_percent": 0.38461538461538464
      }
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.0010899075534608628
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
      "system_capacity": 67.2541696,
      "system": 2.325712896,
      "main": {
       "total_rss": 0.0031088640000000002,
       "private_rss": 0.00031948800000000004,
       "shared_rss": 0.002789376
      },
      "descendants": {
       "total_rss": 0.822874112,
       "private_rss": 0.705110016,
       "shared_rss": 0.11776409600000001
      },
      "combined": {
       "total_rss": 0.824086528,
       "private_rss": 0.705429504,
       "shared_rss": 0.11865702400000001
      }
     },
     "max_gpu_ram": {
      "unit": "gigabytes",
      "system_capacity": 16.376,
      "system": 1.235,
      "main": 0.0,
      "descendants": 0.392,
      "combined": 0.392
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 332.1,
       "max_hardware_percent": 27.675,
       "mean_sum_percent": 166.07,
       "mean_hardware_percent": 13.839166666666666
      },
      "main": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "descendants": {
       "max_sum_percent": 104.1,
       "max_hardware_percent": 8.674999999999999,
       "mean_sum_percent": 99.77000000000001,
       "mean_hardware_percent": 8.314166666666665
      },
      "combined": {
       "max_sum_percent": 104.1,
       "max_hardware_percent": 8.674999999999999,
       "mean_sum_percent": 99.77000000000001,
       "mean_hardware_percent": 8.314166666666665
      },
      "main_n_threads": 1,
      "descendants_n_threads": 12,
      "combined_n_threads": 13
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 5.0,
       "max_hardware_percent": 5.0,
       "mean_sum_percent": 0.5,
       "mean_hardware_percent": 0.5
      }
     },
     "compute_time": {
      "unit": "hours",
      "time": 0.0010636144214206272
     }
    }
