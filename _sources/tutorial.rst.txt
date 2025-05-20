.. _tutorial-label:

Tutorial
========

API
---

Tracking
~~~~~~~~

Basics
^^^^^^

The ``gpu_tracker`` package provides the ``Tracker`` class which uses a
subprocess to measure computational resource usage, namely the compute
time, maximum CPU utilization, mean CPU utilization, maximum RAM used,
maximum GPU utilization, mean GPU utilization, and maximum GPU RAM used.
It supports both NVIDIA and AMD GPUs. The ``start()`` method starts this
process which tracks usage in the background. The ``Tracker`` class can
be used as a context manager. Upon entering the context, one can write
the code for which resource usage is measured. The compute time will be
the time from entering the context to exiting the context and the RAM,
GPU RAM, CPU utilization, and GPU utilization quantities will be the
respective computational resources used by the code that’s within the
context.

.. code:: python3

    import gpu_tracker as gput
    from example_module import example_function

.. code:: python3

    with gput.Tracker(n_expected_cores=1, sleep_time=0.1) as tracker:
        example_function()

The ``Tracker`` class implements the ``__str__`` method so it can be
printed as a string with the values and units of each computational
resource formatted.

.. code:: python3

    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 4.307
       Main:
          Total RSS: 0.924
          Private RSS: 0.755
          Shared RSS: 0.171
       Descendants:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 0.924
          Private RSS: 0.755
          Shared RSS: 0.171
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
          Max sum percent: 222.6
          Max hardware percent: 18.55
          Mean sum percent: 149.285
          Mean hardware percent: 12.44
       Main:
          Max sum percent: 103.3
          Max hardware percent: 103.3
          Mean sum percent: 94.285
          Mean hardware percent: 94.285
       Descendants:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Combined:
          Max sum percent: 103.3
          Max hardware percent: 103.3
          Mean sum percent: 94.285
          Mean hardware percent: 94.285
       Main number of threads: 15
       Descendants number of threads: 0
       Combined number of threads: 15
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 5.0
          Max hardware percent: 5.0
          Mean sum percent: 0.385
          Mean hardware percent: 0.385
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

The ``Tracker`` can alternatively be used by explicitly calling its
``start()`` and ``stop()`` methods which behave the same as entering and
exiting the context manager respectively.

.. code:: python3

    tracker = gput.Tracker()
    tracker.start()
    example_function()
    tracker.stop()

Arguments and Attributes
^^^^^^^^^^^^^^^^^^^^^^^^

The units of the computational resources can be modified as desired. The
following example measures the RAM in megabytes, the GPU RAM in
megabytes, and the compute time in seconds.

.. code:: python3

    with gput.Tracker(ram_unit='megabytes', gpu_ram_unit='megabytes', time_unit='seconds', sleep_time=0.1) as tracker:
        example_function()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: megabytes
       System capacity: 67254.166
       System: 1984.791
       Main:
          Total RSS: 873.853
          Private RSS: 638.353
          Shared RSS: 235.68
       Descendants:
          Total RSS: 0.0
          Private RSS: 0.0
          Shared RSS: 0.0
       Combined:
          Total RSS: 873.853
          Private RSS: 638.353
          Shared RSS: 235.68
    Max GPU RAM:
       Unit: megabytes
       System capacity: 16376.0
       System: 728.0
       Main: 506.0
       Descendants: 0.0
       Combined: 506.0
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 161.6
          Max hardware percent: 13.467
          Mean sum percent: 145.517
          Mean hardware percent: 12.126
       Main:
          Max sum percent: 101.5
          Max hardware percent: 8.458
          Mean sum percent: 98.683
          Mean hardware percent: 8.224
       Descendants:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Combined:
          Max sum percent: 101.5
          Max hardware percent: 8.458
          Mean sum percent: 98.683
          Mean hardware percent: 8.224
       Main number of threads: 15
       Descendants number of threads: 0
       Combined number of threads: 15
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 3.0
          Max hardware percent: 3.0
          Mean sum percent: 0.25
          Mean hardware percent: 0.25
    Compute time:
       Unit: seconds
       Time: 2.729


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
      "system": 1984.790528,
      "main": {
       "total_rss": 873.8529279999999,
       "private_rss": 638.353408,
       "shared_rss": 235.679744
      },
      "descendants": {
       "total_rss": 0.0,
       "private_rss": 0.0,
       "shared_rss": 0.0
      },
      "combined": {
       "total_rss": 873.8529279999999,
       "private_rss": 638.353408,
       "shared_rss": 235.679744
      }
     },
     "max_gpu_ram": {
      "unit": "megabytes",
      "system_capacity": 16376.0,
      "system": 728.0,
      "main": 506.0,
      "descendants": 0.0,
      "combined": 506.0
     },
     "cpu_utilization": {
      "system_core_count": 12,
      "n_expected_cores": 12,
      "system": {
       "max_sum_percent": 161.60000000000002,
       "max_hardware_percent": 13.466666666666669,
       "mean_sum_percent": 145.51666666666668,
       "mean_hardware_percent": 12.12638888888889
      },
      "main": {
       "max_sum_percent": 101.5,
       "max_hardware_percent": 8.458333333333334,
       "mean_sum_percent": 98.68333333333334,
       "mean_hardware_percent": 8.22361111111111
      },
      "descendants": {
       "max_sum_percent": 0.0,
       "max_hardware_percent": 0.0,
       "mean_sum_percent": 0.0,
       "mean_hardware_percent": 0.0
      },
      "combined": {
       "max_sum_percent": 101.5,
       "max_hardware_percent": 8.458333333333334,
       "mean_sum_percent": 98.68333333333334,
       "mean_hardware_percent": 8.22361111111111
      },
      "main_n_threads": 15,
      "descendants_n_threads": 0,
      "combined_n_threads": 15
     },
     "gpu_utilization": {
      "system_gpu_count": 1,
      "n_expected_gpus": 1,
      "gpu_percentages": {
       "max_sum_percent": 3.0,
       "max_hardware_percent": 3.0,
       "mean_sum_percent": 0.25,
       "mean_hardware_percent": 0.25
      }
     },
     "compute_time": {
      "unit": "seconds",
      "time": 2.728560209274292
     }
    }


Using Python data classes, the ``Tracker`` class additionally has a
``resource_usage`` attribute containing fields that provide the usage
information for each individual computational resource.

.. code:: python3

    tracker.resource_usage.max_ram




.. code:: none

    MaxRAM(unit='megabytes', system_capacity=67254.165504, system=1984.790528, main=RSSValues(total_rss=873.8529279999999, private_rss=638.353408, shared_rss=235.679744), descendants=RSSValues(total_rss=0.0, private_rss=0.0, shared_rss=0.0), combined=RSSValues(total_rss=873.8529279999999, private_rss=638.353408, shared_rss=235.679744))



.. code:: python3

    tracker.resource_usage.max_ram.unit




.. code:: none

    'megabytes'



.. code:: python3

    tracker.resource_usage.max_ram.main




.. code:: none

    RSSValues(total_rss=873.8529279999999, private_rss=638.353408, shared_rss=235.679744)



.. code:: python3

    tracker.resource_usage.max_ram.main.total_rss




.. code:: none

    873.8529279999999



.. code:: python3

    tracker.resource_usage.max_gpu_ram




.. code:: none

    MaxGPURAM(unit='megabytes', system_capacity=16376.0, system=728.0, main=506.0, descendants=0.0, combined=506.0)



.. code:: python3

    tracker.resource_usage.compute_time




.. code:: none

    ComputeTime(unit='seconds', time=2.728560209274292)



Below is an example of using a child process. Notice the descendants
fields are now non-zero.

.. code:: python3

    import multiprocessing as mp
    ctx = mp.get_context(method='spawn')
    child_process = ctx.Process(target=example_function)
    with gput.Tracker(n_expected_cores=2, sleep_time=0.4) as tracker:
        child_process.start()
        example_function()
        child_process.join()
    child_process.close()
    print(tracker)


.. code:: none

    Max RAM:
       Unit: gigabytes
       System capacity: 67.254
       System: 2.388
       Main:
          Total RSS: 0.849
          Private RSS: 0.528
          Shared RSS: 0.325
       Descendants:
          Total RSS: 0.845
          Private RSS: 0.734
          Shared RSS: 0.112
       Combined:
          Total RSS: 1.371
          Private RSS: 1.05
          Shared RSS: 0.325
    Max GPU RAM:
       Unit: gigabytes
       System capacity: 16.376
       System: 1.236
       Main: 0.506
       Descendants: 0.506
       Combined: 1.012
    CPU utilization:
       System core count: 12
       Number of expected cores: 2
       System:
          Max sum percent: 338.0
          Max hardware percent: 28.167
          Mean sum percent: 183.644
          Mean hardware percent: 15.304
       Main:
          Max sum percent: 101.0
          Max hardware percent: 50.5
          Mean sum percent: 60.178
          Mean hardware percent: 30.089
       Descendants:
          Max sum percent: 354.1
          Max hardware percent: 177.05
          Mean sum percent: 109.033
          Mean hardware percent: 54.517
       Combined:
          Max sum percent: 452.2
          Max hardware percent: 226.1
          Mean sum percent: 169.211
          Mean hardware percent: 84.606
       Main number of threads: 15
       Descendants number of threads: 13
       Combined number of threads: 28
    GPU utilization:
       System GPU count: 1
       Number of expected GPUs: 1
       GPU percentages:
          Max sum percent: 5.0
          Max hardware percent: 5.0
          Mean sum percent: 0.556
          Mean hardware percent: 0.556
    Compute time:
       Unit: hours
       Time: 0.001


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

While the ``Tracker`` class automatically detects which brand of GPU is
installed (either NVIDIA or AMD), one can explicitly choose the GPU
brand with the ``gpu_brand`` parameter

.. code:: python3

    tracker = gput.Tracker(gpu_brand='nvidia')

While the ``Tracker`` by default stores aggregates of the computational
resource usage across the timepoints, one can store the individual
measured values at every timepoint in a file, either CSV or SQLite
format, using the ``tracking_file`` parameter. **NOTE** for the CSV
format, the static data (e.g. RAM system capacity, number of cores in
the OS, etc.) is stored on the the first two rows with the headers on
the first row followed by the static data on the second row. The headers
of the timepoint data is on the third row followed by the timepoint data
on the remaining rows. The SQLite file, however, stores the static data
and timepoint data in different tables: “data” and “static_data”
respectively.

.. code:: python3

    tracker = gput.Tracker(tracking_file='my-file.csv')
    tracker = gput.Tracker(tracking_file='my-file.sqlite')

Sub-tracking
~~~~~~~~~~~~

Logging Code Block Timestamps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the ``Tracker`` object by itself can track a block of code, there
are some cases where one might want to track one code block and a
smaller code block within it or track multiple code blocks at a time
without creating several tracking processes simultaneously, especially
when tracking a code block that is called within multi-processing or a
code block that is called several times. Similarly, one might want to
track the resource usage of a particular function whenever it is called.
Whether a function or some other specified code block, the
``SubTracker`` class can determine the computational resources used
during the start times and stop times of a given code block. This
includes the mean resources used during the times the code block is
called, the mean time taken to complete the code block each time it is
called, the number of times it is called, etc. Sub-tracking uses the
tracking file specified by the ``tracking_file`` parameter of the
``Tracker`` object alonside a sub-tracking file which contains the start
and stop times of each code block one desires to sub-track. The
sub-tracking file can be created in Python using the ``SubTracker``
class, a context manager around the desired code block. Setting the
``overwrite`` parameter (default ``False``) of the ``Tracker`` and
``SubTracker`` to ``True`` overwrites the ``tracking_file`` or
``sub_tracking_file`` respectively if a file of that path already
exists. Keep this paramter at ``False`` to avoid loss of data if it is
still needed.

.. code:: python3

    tracker = gput.Tracker(sleep_time=0.5, tracking_file='tracking.csv', overwrite=False)
    tracker.start()
    # Perform other computation here
    for _ in range(5):
        with gput.SubTracker(code_block_name='my-code-block', sub_tracking_file='sub-tracking.csv', overwrite=False):
            example_function()
    # Perform other computation here

In the above example, a tracking session is initiated within the context
of the ``Tracker`` object whose tracking file is ‘tracking.csv’. Then we
have a for loop wherein a function is called 5 times. Other computation
might be performed before or after this for loop, but if the
computational resource usage of the contents of the for loop is of
interest in particular, that code block can be sub-tracked by wrapping
it within the context of the ``SubTracker`` object whose sub-tracking
file is ‘sub-tracking.csv’. Alternatively, SQLite (.sqlite) files can be
used to speed up querying in the case of very long tracking sessions.
The name of the code block is ‘my-code-block’, given to distinguish it
from other code blocks being sub-tracked.

If one wants to sub-track all calls to a particular function, the
``sub_track`` function decorator can be used instead of wrapping the
function call with a ``SubTracker`` context every time it is called:

.. code:: python3

    @gput.sub_track(code_block_name='my-function', sub_tracking_file='sub-tracking.csv', overwrite=False)
    def my_function(*args, **kwargs):
        example_function()
    
    for _ in range(3):
        my_function()
    tracker.stop()

When sub-tracking a code block using the ``SubTracker`` context, the
default ``code_block_name`` is the relative path of the Python file
followed by a colon followed by the line number where the ``SubTracker``
context is initialized. When sub-tracking a function, the default
``code_block_name`` is the relative path of the Python file followed by
a colon followed by the name of the function.

Analysis
^^^^^^^^

Once a tracking file and at least one sub-tracking file have been
created, the results can be analyzed using the ``SubTrackingAnalyzer``
class, instantiated by passing in the path to the tracking file and the
path to the sub-tracking file.

.. code:: python3

    analyzer = gput.SubTrackingAnalyzer(tracking_file='tracking.csv', sub_tracking_file='sub-tracking.csv')

When sub-tracking a code block within a function that’s part of
multi-processing (i.e. called within one of multiple sub-processes), the
sub-tracking file must be unique to that process, which is why the
default ``sub_tracking_file`` is the process ID followed by “.csv”. One
way or another, a different sub-tracking file must be created per worker
to prevent multiple processes from logging to the same file. The
``SubTrackingAnalyzer`` has a ``combine_sub_tracking_files`` method that
can combine these multiple sub-tracking files into a single sub-tracking
file whose path is specified by the ``sub_tracking_file`` parameter
above. Once a sub-tracking file is created from a single process or
combined from multiple, the results can be obtained via the
``sub_tracking_results`` method.

.. code:: python3

    results = analyzer.sub_tracking_results()
    type(results)




.. code:: none

    gpu_tracker.sub_tracker.SubTrackingResults



The ``sub_tracking_results`` method returns a ``SubTrackingResults``
object which contains summary statistics of the overall resource usage
(all time points in the tracking file) and the per code block resource
usage (the timepoints within calls to a code block i.e. the start/stop
times) as ``DataFrame`` or ``Series`` objects from the ``pandas``
package.

.. code:: python3

    results.overall




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>min</th>
          <th>max</th>
          <th>mean</th>
          <th>std</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>main_ram</th>
          <td>0.341217</td>
          <td>0.920560</td>
          <td>0.861921</td>
          <td>0.100084</td>
        </tr>
        <tr>
          <th>descendants_ram</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>combined_ram</th>
          <td>0.341217</td>
          <td>0.920560</td>
          <td>0.861921</td>
          <td>0.100084</td>
        </tr>
        <tr>
          <th>system_ram</th>
          <td>4.602618</td>
          <td>5.701517</td>
          <td>5.281926</td>
          <td>0.220270</td>
        </tr>
        <tr>
          <th>main_gpu_ram</th>
          <td>0.000000</td>
          <td>0.506000</td>
          <td>0.448364</td>
          <td>0.151267</td>
        </tr>
        <tr>
          <th>descendants_gpu_ram</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>combined_gpu_ram</th>
          <td>0.000000</td>
          <td>0.506000</td>
          <td>0.448364</td>
          <td>0.151267</td>
        </tr>
        <tr>
          <th>system_gpu_ram</th>
          <td>0.215000</td>
          <td>0.727000</td>
          <td>0.668909</td>
          <td>0.152657</td>
        </tr>
        <tr>
          <th>gpu_sum_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>gpu_hardware_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>main_n_threads</th>
          <td>12.000000</td>
          <td>15.000000</td>
          <td>14.757576</td>
          <td>0.791766</td>
        </tr>
        <tr>
          <th>descendants_n_threads</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>combined_n_threads</th>
          <td>12.000000</td>
          <td>15.000000</td>
          <td>14.757576</td>
          <td>0.791766</td>
        </tr>
        <tr>
          <th>cpu_system_sum_utilization_percent</th>
          <td>15.400000</td>
          <td>138.400000</td>
          <td>121.918182</td>
          <td>19.484617</td>
        </tr>
        <tr>
          <th>cpu_system_hardware_utilization_percent</th>
          <td>1.283333</td>
          <td>11.533333</td>
          <td>10.159848</td>
          <td>1.623718</td>
        </tr>
        <tr>
          <th>cpu_main_sum_utilization_percent</th>
          <td>91.400000</td>
          <td>103.300000</td>
          <td>99.060606</td>
          <td>2.571228</td>
        </tr>
        <tr>
          <th>cpu_main_hardware_utilization_percent</th>
          <td>7.616667</td>
          <td>8.608333</td>
          <td>8.255051</td>
          <td>0.214269</td>
        </tr>
        <tr>
          <th>cpu_descendants_sum_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>cpu_descendants_hardware_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>cpu_combined_sum_utilization_percent</th>
          <td>91.400000</td>
          <td>103.300000</td>
          <td>99.060606</td>
          <td>2.571228</td>
        </tr>
        <tr>
          <th>cpu_combined_hardware_utilization_percent</th>
          <td>7.616667</td>
          <td>8.608333</td>
          <td>8.255051</td>
          <td>0.214269</td>
        </tr>
      </tbody>
    </table>
    </div>



The ``SubTrackingResults`` class additionally contains the static data
i.e. the information that remains constant throughout the tracking
session.

.. code:: python3

    results.static_data




.. code:: none

    ram_unit                   gigabytes
    gpu_ram_unit               gigabytes
    time_unit                      hours
    ram_system_capacity        67.254166
    gpu_ram_system_capacity       16.376
    system_core_count                 12
    n_expected_cores                  12
    system_gpu_count                   1
    n_expected_gpus                    1
    Name: 0, dtype: object



The ``code_block_results`` attribute of the ``SubTrackingResults`` class
is a list of ``CodeBlockResults`` objects, containing the resource usage
and compute time summary statistics. In this case, there are two
``CodeBlockResults`` objects in the list since there were two code
blocks sub-tracked in this tracking session.

.. code:: python3

    [my_code_block_results, my_function_results] = results.code_block_results
    type(my_code_block_results)




.. code:: none

    gpu_tracker.sub_tracker.CodeBlockResults



The ``compute_time`` attribute of the ``CodeBlockResults`` class
contains summary statistics for the time spent on the code block, where
``total`` is the total amount of time spent within the code block during
the tracking session, ``mean`` is the average time taken on each call to
the code block, etc. The ``resource_usage`` attribute provides summary
statistics for the computational resources used during calls to the code
block i.e. within the start/stop times.

.. code:: python3

    my_code_block_results.compute_time




.. code:: none

    min       2.630907
    max       2.869182
    mean      2.685580
    std       0.102789
    total    13.427902
    dtype: float64



.. code:: python3

    my_code_block_results.resource_usage




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>min</th>
          <th>max</th>
          <th>mean</th>
          <th>std</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>main_ram</th>
          <td>0.341217</td>
          <td>0.912278</td>
          <td>0.846999</td>
          <td>0.122948</td>
        </tr>
        <tr>
          <th>descendants_ram</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>combined_ram</th>
          <td>0.341217</td>
          <td>0.912278</td>
          <td>0.846999</td>
          <td>0.122948</td>
        </tr>
        <tr>
          <th>system_ram</th>
          <td>4.602618</td>
          <td>5.261357</td>
          <td>5.170665</td>
          <td>0.147118</td>
        </tr>
        <tr>
          <th>main_gpu_ram</th>
          <td>0.000000</td>
          <td>0.506000</td>
          <td>0.415429</td>
          <td>0.182971</td>
        </tr>
        <tr>
          <th>descendants_gpu_ram</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>combined_gpu_ram</th>
          <td>0.000000</td>
          <td>0.506000</td>
          <td>0.415429</td>
          <td>0.182971</td>
        </tr>
        <tr>
          <th>system_gpu_ram</th>
          <td>0.215000</td>
          <td>0.727000</td>
          <td>0.635714</td>
          <td>0.184676</td>
        </tr>
        <tr>
          <th>gpu_sum_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>gpu_hardware_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>main_n_threads</th>
          <td>12.000000</td>
          <td>15.000000</td>
          <td>14.619048</td>
          <td>0.973457</td>
        </tr>
        <tr>
          <th>descendants_n_threads</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>combined_n_threads</th>
          <td>12.000000</td>
          <td>15.000000</td>
          <td>14.619048</td>
          <td>0.973457</td>
        </tr>
        <tr>
          <th>cpu_system_sum_utilization_percent</th>
          <td>15.400000</td>
          <td>138.400000</td>
          <td>120.142857</td>
          <td>24.347907</td>
        </tr>
        <tr>
          <th>cpu_system_hardware_utilization_percent</th>
          <td>1.283333</td>
          <td>11.533333</td>
          <td>10.011905</td>
          <td>2.028992</td>
        </tr>
        <tr>
          <th>cpu_main_sum_utilization_percent</th>
          <td>91.400000</td>
          <td>103.300000</td>
          <td>98.652381</td>
          <td>2.733243</td>
        </tr>
        <tr>
          <th>cpu_main_hardware_utilization_percent</th>
          <td>7.616667</td>
          <td>8.608333</td>
          <td>8.221032</td>
          <td>0.227770</td>
        </tr>
        <tr>
          <th>cpu_descendants_sum_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>cpu_descendants_hardware_utilization_percent</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>cpu_combined_sum_utilization_percent</th>
          <td>91.400000</td>
          <td>103.300000</td>
          <td>98.652381</td>
          <td>2.733243</td>
        </tr>
        <tr>
          <th>cpu_combined_hardware_utilization_percent</th>
          <td>7.616667</td>
          <td>8.608333</td>
          <td>8.221032</td>
          <td>0.227770</td>
        </tr>
      </tbody>
    </table>
    </div>



Additionally, the ``CodeBlockResults`` class also has attributes for the
name of the code block, the number of times it was called during the
tracking session, the number of calls that included at least one
timepoint, and the total number of timepoints measured within all calls
to the code block.

.. code:: python3

    my_code_block_results.name, my_code_block_results.num_calls, my_code_block_results.num_non_empty_calls, my_code_block_results.num_timepoints




.. code:: none

    ('my-code-block', 5, 5, 21)



The analysis results can also be printed in their entirety.
Alternatively, the ``to_json`` method can provide this comprehensive
information in JSON format.

.. code:: python3

    print(results)


.. code:: none

    Overall:
    	                                                    min         max        mean        std
    	main_ram                                       0.341860    0.944374    0.856037   0.125014
    	descendants_ram                                0.000000    0.000000    0.000000   0.000000
    	combined_ram                                   0.341860    0.944374    0.856037   0.125014
    	system_ram                                     4.859711    5.553644    5.253445   0.134081
    	main_gpu_ram                                   0.000000    0.506000    0.429920   0.170432
    	descendants_gpu_ram                            0.000000    0.000000    0.000000   0.000000
    	combined_gpu_ram                               0.000000    0.506000    0.429920   0.170432
    	system_gpu_ram                                 0.215000    0.727000    0.650320   0.172010
    	gpu_sum_utilization_percent                    0.000000    3.000000    0.120000   0.600000
    	gpu_hardware_utilization_percent               0.000000    3.000000    0.120000   0.600000
    	main_n_threads                                12.000000   15.000000   14.720000   0.842615
    	descendants_n_threads                          0.000000    0.000000    0.000000   0.000000
    	combined_n_threads                            12.000000   15.000000   14.720000   0.842615
    	cpu_system_sum_utilization_percent            11.900000  133.400000  119.212000  22.741909
    	cpu_system_hardware_utilization_percent        0.991667   11.116667    9.934333   1.895159
    	cpu_main_sum_utilization_percent              78.000000  103.200000   96.924000   6.390767
    	cpu_main_hardware_utilization_percent          6.500000    8.600000    8.077000   0.532564
    	cpu_descendants_sum_utilization_percent        0.000000    0.000000    0.000000   0.000000
    	cpu_descendants_hardware_utilization_percent   0.000000    0.000000    0.000000   0.000000
    	cpu_combined_sum_utilization_percent          78.000000  103.200000   96.924000   6.390767
    	cpu_combined_hardware_utilization_percent      6.500000    8.600000    8.077000   0.532564
    Static Data:
    	   ram_unit gpu_ram_unit time_unit ram_system_capacity gpu_ram_system_capacity system_core_count n_expected_cores system_gpu_count n_expected_gpus
    	  gigabytes    gigabytes     hours           67.254166                  16.376                12               12                1               1
    Code Block Results:
    	Name:                my-code-block
    	Num Timepoints:      12
    	Num Calls:           3
    	Num Non Empty Calls: 3
    	Compute Time:
    		       min       max      mean       std     total
    		  2.580433  2.789909  2.651185  0.120147  7.953554
    	Resource Usage:
    		                                                    min         max        mean        std
    		main_ram                                       0.341860    0.936559    0.808736   0.167663
    		descendants_ram                                0.000000    0.000000    0.000000   0.000000
    		combined_ram                                   0.341860    0.936559    0.808736   0.167663
    		system_ram                                     4.859711    5.553644    5.231854   0.191567
    		main_gpu_ram                                   0.000000    0.506000    0.363500   0.225892
    		descendants_gpu_ram                            0.000000    0.000000    0.000000   0.000000
    		combined_gpu_ram                               0.000000    0.506000    0.363500   0.225892
    		system_gpu_ram                                 0.215000    0.727000    0.583250   0.228088
    		gpu_sum_utilization_percent                    0.000000    0.000000    0.000000   0.000000
    		gpu_hardware_utilization_percent               0.000000    0.000000    0.000000   0.000000
    		main_n_threads                                12.000000   15.000000   14.416667   1.164500
    		descendants_n_threads                          0.000000    0.000000    0.000000   0.000000
    		combined_n_threads                            12.000000   15.000000   14.416667   1.164500
    		cpu_system_sum_utilization_percent            11.900000  130.800000  113.641667  32.352363
    		cpu_system_hardware_utilization_percent        0.991667   10.900000    9.470139   2.696030
    		cpu_main_sum_utilization_percent              79.600000  103.100000   96.583333   6.726587
    		cpu_main_hardware_utilization_percent          6.633333    8.591667    8.048611   0.560549
    		cpu_descendants_sum_utilization_percent        0.000000    0.000000    0.000000   0.000000
    		cpu_descendants_hardware_utilization_percent   0.000000    0.000000    0.000000   0.000000
    		cpu_combined_sum_utilization_percent          79.600000  103.100000   96.583333   6.726587
    		cpu_combined_hardware_utilization_percent      6.633333    8.591667    8.048611   0.560549
    
    	Name:                my-function
    	Num Timepoints:      12
    	Num Calls:           3
    	Num Non Empty Calls: 3
    	Compute Time:
    		       min       max      mean       std     total
    		  2.538011  2.577679  2.553176  0.021419  7.659528
    	Resource Usage:
    		                                                     min         max        mean       std
    		main_ram                                        0.864592    0.944374    0.896998  0.034505
    		descendants_ram                                 0.000000    0.000000    0.000000  0.000000
    		combined_ram                                    0.864592    0.944374    0.896998  0.034505
    		system_ram                                      5.203415    5.315219    5.271566  0.038751
    		main_gpu_ram                                    0.314000    0.506000    0.490000  0.055426
    		descendants_gpu_ram                             0.000000    0.000000    0.000000  0.000000
    		combined_gpu_ram                                0.314000    0.506000    0.490000  0.055426
    		system_gpu_ram                                  0.535000    0.727000    0.711000  0.055426
    		gpu_sum_utilization_percent                     0.000000    3.000000    0.250000  0.866025
    		gpu_hardware_utilization_percent                0.000000    3.000000    0.250000  0.866025
    		main_n_threads                                 15.000000   15.000000   15.000000  0.000000
    		descendants_n_threads                           0.000000    0.000000    0.000000  0.000000
    		combined_n_threads                             15.000000   15.000000   15.000000  0.000000
    		cpu_system_sum_utilization_percent            120.300000  133.400000  124.566667  4.001439
    		cpu_system_hardware_utilization_percent        10.025000   11.116667   10.380556  0.333453
    		cpu_main_sum_utilization_percent               94.700000  103.200000   98.841667  2.677332
    		cpu_main_hardware_utilization_percent           7.891667    8.600000    8.236806  0.223111
    		cpu_descendants_sum_utilization_percent         0.000000    0.000000    0.000000  0.000000
    		cpu_descendants_hardware_utilization_percent    0.000000    0.000000    0.000000  0.000000
    		cpu_combined_sum_utilization_percent           94.700000  103.200000   98.841667  2.677332
    		cpu_combined_hardware_utilization_percent       7.891667    8.600000    8.236806  0.223111
    
    


Comparison
^^^^^^^^^^

The ``TrackingComparison`` class allows for comparing the resource usage
of multiple tracking sessions, both the overall usage of the sessions
and any code blocks that were sub-tracked. This is helpful if one wants
to see how changes to the process might impact the computational
efficiency of it, such as changes to implementation, input data, etc. To
do this, the ``TrackingComparison`` takes a mapping of the given name of
a tracking session to the file path where a ``SubTrackingResults``
object is stored in pickle format. Say we had two tracking sessions and
we wanted to compare them. First, we store the ``results`` of the first
tracking session in a pickle file. If we’d like to re-use the same names
for the ``tracking_file`` and ``sub_tracking_file`` in the second
tracking session, we can safely set the ``overwrite`` argument to
``True`` since their data has been saved in ‘results.pkl’.

.. code:: python3

    import pickle as pkl
    import os
    
    with open('results.pkl', 'wb') as file:
        pkl.dump(results, file)
    

Once we have the results of the first tracking session saved, we can
start a new tracking session in another run of the program that we are
profiling. Say we made some code changes and we want to compare the two
implementations, we can populate a new ``tracking_file`` and
``sub_tracking_file`` with data from the new tracking session.

.. code:: python3

    import gpu_tracker as gput
    from example_module import example_function
    import pickle as pkl
    
    @gput.sub_track(code_block_name='my-function', sub_tracking_file='sub-tracking.csv', overwrite=True)
    def my_function(*args, **kwargs):
        example_function()
    
    with gput.Tracker(sleep_time=0.5, tracking_file='tracking.csv', overwrite=True):
        for _ in range(3):
            with gput.SubTracker(code_block_name='my-code-block', sub_tracking_file='sub-tracking.csv', overwrite=True):
                example_function()
            my_function()
    results2 = gput.SubTrackingAnalyzer(tracking_file='tracking.csv', sub_tracking_file='sub-tracking.csv').sub_tracking_results()
    with open('results2.pkl', 'wb') as file:
        pkl.dump(results2, file)

The first tracking session stored its results in ‘results.pkl’ while the
second tracking session stored its results in ‘results2.pkl’. Say we
decided to call the first session ‘A’ and the second session ‘B’. The
``TrackingComparison`` object would be initialized like so:

.. code:: python3

    comparison = gput.TrackingComparison(file_path_map={'A': 'results.pkl', 'B': 'results2.pkl'})

Once the ``TrackingComparison`` is created, its compare method generates
the ``ComparisonResults`` object detailing the computational resource
usage measured in one tracking session to that of the other tracking
sessions. The ``statistic`` parameter determines which summary statistic
of the measurements to compare, defaulting to ‘mean’. In this example,
we will compare the maximum measurements by setting ``statistic`` to
‘max’.

.. code:: python3

    results = comparison.compare(statistic='max')
    type(results)




.. code:: none

    gpu_tracker.sub_tracker.ComparisonResults



The ``overall_resource_usage`` attribute of the ``ComparisonResults``
class is a dictionary mapping each measurement to a ``Series`` comparing
that measurement across all timepoints in one tracking session to
another.

.. code:: python3

    results.overall_resource_usage.keys()




.. code:: none

    dict_keys(['main_ram', 'descendants_ram', 'combined_ram', 'system_ram', 'main_gpu_ram', 'descendants_gpu_ram', 'combined_gpu_ram', 'system_gpu_ram', 'gpu_sum_utilization_percent', 'gpu_hardware_utilization_percent', 'main_n_threads', 'descendants_n_threads', 'combined_n_threads', 'cpu_system_sum_utilization_percent', 'cpu_system_hardware_utilization_percent', 'cpu_main_sum_utilization_percent', 'cpu_main_hardware_utilization_percent', 'cpu_descendants_sum_utilization_percent', 'cpu_descendants_hardware_utilization_percent', 'cpu_combined_sum_utilization_percent', 'cpu_combined_hardware_utilization_percent'])



For example, we can compare the overall maximum ‘main_ram’ of tracking
session ‘A’ to tracking session ‘B’.

.. code:: python3

    results.overall_resource_usage['main_ram']




.. code:: none

    A    0.920560
    B    0.944374
    dtype: float64



The ``code_block_resource_usage`` attribute is a dictionary that
compares the same resource usage but for each code block rather than
overall.

.. code:: python3

    results.code_block_resource_usage.keys()




.. code:: none

    dict_keys(['main_ram', 'descendants_ram', 'combined_ram', 'system_ram', 'main_gpu_ram', 'descendants_gpu_ram', 'combined_gpu_ram', 'system_gpu_ram', 'gpu_sum_utilization_percent', 'gpu_hardware_utilization_percent', 'main_n_threads', 'descendants_n_threads', 'combined_n_threads', 'cpu_system_sum_utilization_percent', 'cpu_system_hardware_utilization_percent', 'cpu_main_sum_utilization_percent', 'cpu_main_hardware_utilization_percent', 'cpu_descendants_sum_utilization_percent', 'cpu_descendants_hardware_utilization_percent', 'cpu_combined_sum_utilization_percent', 'cpu_combined_hardware_utilization_percent'])



Each measurement is a dictionary mapping each code block name to the
resources used across tracking sessions in that code block.

.. code:: python3

    results.code_block_resource_usage['main_ram'].keys()




.. code:: none

    dict_keys(['my-code-block', 'my-function'])



For example, the maximum ‘main_ram’ used by ‘my-code-block’ in tracking
session ‘A’ can be compared to that of tracking session ‘B’.

.. code:: python3

    results.code_block_resource_usage['main_ram']['my-code-block']




.. code:: none

    A    0.912278
    B    0.936559
    dtype: float64



Finally the ``code_block_compute_time`` attribute is a dictionary that
compares the compute time summary statistics for each code block and for
each tracking session.

.. code:: python3

    results.code_block_compute_time.keys()




.. code:: none

    dict_keys(['my-code-block', 'my-function'])



For example, we can compare the maximum compute time of ‘my-code-block’
in tracking session ‘A’ to that of tracking session ‘B’.

.. code:: python3

    results.code_block_compute_time['my-code-block']




.. code:: none

    B    2.789909
    A    2.869182
    dtype: float64



The comparison results can also be printed in their entirety.
Alternatively, the ``to_json`` method can provide this comprehensive
information in JSON format.

.. code:: python3

    print(results)


.. code:: none

    Overall Resource Usage:
    	Main Ram:
    		        A         B
    		  0.92056  0.944374
    	Descendants Ram:
    		    A    B
    		  0.0  0.0
    	Combined Ram:
    		        A         B
    		  0.92056  0.944374
    	System Ram:
    		         B         A
    		  5.553644  5.701517
    	Main Gpu Ram:
    		      A      B
    		  0.506  0.506
    	Descendants Gpu Ram:
    		    A    B
    		  0.0  0.0
    	Combined Gpu Ram:
    		      A      B
    		  0.506  0.506
    	System Gpu Ram:
    		      A      B
    		  0.727  0.727
    	Gpu Sum Utilization Percent:
    		    A    B
    		  0.0  3.0
    	Gpu Hardware Utilization Percent:
    		    A    B
    		  0.0  3.0
    	Main N Threads:
    		     A     B
    		  15.0  15.0
    	Descendants N Threads:
    		    A    B
    		  0.0  0.0
    	Combined N Threads:
    		     A     B
    		  15.0  15.0
    	Cpu System Sum Utilization Percent:
    		      B      A
    		  133.4  138.4
    	Cpu System Hardware Utilization Percent:
    		          B          A
    		  11.116667  11.533333
    	Cpu Main Sum Utilization Percent:
    		      B      A
    		  103.2  103.3
    	Cpu Main Hardware Utilization Percent:
    		    B         A
    		  8.6  8.608333
    	Cpu Descendants Sum Utilization Percent:
    		    A    B
    		  0.0  0.0
    	Cpu Descendants Hardware Utilization Percent:
    		    A    B
    		  0.0  0.0
    	Cpu Combined Sum Utilization Percent:
    		      B      A
    		  103.2  103.3
    	Cpu Combined Hardware Utilization Percent:
    		    B         A
    		  8.6  8.608333
    Code Block Resource Usage:
    	Main Ram:
    		my-code-block:
    			         A         B
    			  0.912278  0.936559
    		my-function:
    			        A         B
    			  0.92056  0.944374
    	Descendants Ram:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  0.0
    	Combined Ram:
    		my-code-block:
    			         A         B
    			  0.912278  0.936559
    		my-function:
    			        A         B
    			  0.92056  0.944374
    	System Ram:
    		my-code-block:
    			         A         B
    			  5.261357  5.553644
    		my-function:
    			         B         A
    			  5.315219  5.701517
    	Main Gpu Ram:
    		my-code-block:
    			      A      B
    			  0.506  0.506
    		my-function:
    			      A      B
    			  0.506  0.506
    	Descendants Gpu Ram:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  0.0
    	Combined Gpu Ram:
    		my-code-block:
    			      A      B
    			  0.506  0.506
    		my-function:
    			      A      B
    			  0.506  0.506
    	System Gpu Ram:
    		my-code-block:
    			      A      B
    			  0.727  0.727
    		my-function:
    			      A      B
    			  0.727  0.727
    	Gpu Sum Utilization Percent:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  3.0
    	Gpu Hardware Utilization Percent:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  3.0
    	Main N Threads:
    		my-code-block:
    			     A     B
    			  15.0  15.0
    		my-function:
    			     A     B
    			  15.0  15.0
    	Descendants N Threads:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  0.0
    	Combined N Threads:
    		my-code-block:
    			     A     B
    			  15.0  15.0
    		my-function:
    			     A     B
    			  15.0  15.0
    	Cpu System Sum Utilization Percent:
    		my-code-block:
    			      B      A
    			  130.8  138.4
    		my-function:
    			      A      B
    			  131.1  133.4
    	Cpu System Hardware Utilization Percent:
    		my-code-block:
    			     B          A
    			  10.9  11.533333
    		my-function:
    			       A          B
    			  10.925  11.116667
    	Cpu Main Sum Utilization Percent:
    		my-code-block:
    			      B      A
    			  103.1  103.3
    		my-function:
    			      A      B
    			  102.1  103.2
    	Cpu Main Hardware Utilization Percent:
    		my-code-block:
    			         B         A
    			  8.591667  8.608333
    		my-function:
    			         A    B
    			  8.508333  8.6
    	Cpu Descendants Sum Utilization Percent:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  0.0
    	Cpu Descendants Hardware Utilization Percent:
    		my-code-block:
    			    A    B
    			  0.0  0.0
    		my-function:
    			    A    B
    			  0.0  0.0
    	Cpu Combined Sum Utilization Percent:
    		my-code-block:
    			      B      A
    			  103.1  103.3
    		my-function:
    			      A      B
    			  102.1  103.2
    	Cpu Combined Hardware Utilization Percent:
    		my-code-block:
    			         B         A
    			  8.591667  8.608333
    		my-function:
    			         A    B
    			  8.508333  8.6
    Code Block Compute Time:
    	my-code-block:
    		         B         A
    		  2.789909  2.869182
    	my-function:
    		         A         B
    		  2.570437  2.577679
    


CLI
---

Tracking
~~~~~~~~

Basics
^^^^^^

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
        gpu-tracker --execute=<command> [--output=<output>] [--format=<format>] [--tconfig=<config-file>] [--st=<sleep-time>] [--ru=<ram-unit>] [--gru=<gpu-ram-unit>] [--tu=<time-unit>] [--nec=<num-cores>] [--guuids=<gpu-uuids>] [--disable-logs] [--gb=<gpu-brand>] [--tf=<tracking-file>] [--overwrite]
        gpu-tracker sub-track combine --stf=<sub-track-file> [-p <file-path>]...
        gpu-tracker sub-track analyze --tf=<tracking-file> --stf=<sub-track-file> [--output=<output>] [--format=<format>]
        gpu-tracker sub-track compare [--output=<output>] [--format=<format>] [--cconfig=<config-file>] [-m <name>=<file-path>...] [--stat=<statistic>]
    
    Options:
        -h --help               Show this help message and exit.
        -v --version            Show package version and exit.
        -e --execute=<command>  The command to run along with its arguments all within quotes e.g. "ls -l -a".
        -o --output=<output>    File path to store the computational-resource-usage measurements in the case of tracking or the analysis report in the case of sub-tracking. If not set, prints to the screen.
        -f --format=<format>    File format of the output. Either 'json', 'text', or 'pickle'. Defaults to 'text'.
        --tconfig=<config-file> JSON config file containing the key word arguments to the ``Tracker`` class (see API) to be optionally used instead of the corresponding commandline options. If any commandline options are set, they will override the corresponding arguments provided by the config file.
        --st=<sleep-time>       The number of seconds to sleep in between usage-collection iterations.
        --ru=<ram-unit>         One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        --gru=<gpu-ram-unit>    One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        --tu=<time-unit>        One of 'seconds', 'minutes', 'hours', or 'days'.
        --nec=<num-cores>       The number of cores expected to be used. Defaults to the number of cores in the entire operating system.
        --guuids=<gpu-uuids>    Comma separated list of the UUIDs of the GPUs for which to track utilization e.g. gpu-uuid1,gpu-uuid2,etc. Defaults to all the GPUs in the system.
        --disable-logs          If set, warnings are suppressed during tracking. Otherwise, the Tracker logs warnings as usual.
        --gb=<gpu-brand>        The brand of GPU to profile. Valid values are nvidia and amd. Defaults to the brand of GPU detected in the system, checking NVIDIA first.
        --tf=<tracking-file>    If specified, stores the individual resource usage measurements at each iteration. Valid file formats are CSV (.csv) and SQLite (.sqlite) where the SQLite file format stores the data in a table called "data" and allows for more efficient querying.
        --overwrite             Whether to overwrite the tracking file if it already existed before the beginning of this tracking session. Do not set if the data in the existing tracking file is still needed.
        sub-track               Perform sub-tracking related commands.
        combine                 Combines multiple sub-tracking files into one. This is usually a result of sub-tracking a code block that is called in multiple simultaneous processes.
        --stf=<sub-track-file>  The path to the sub-tracking file used to specify the timestamps of specific code-blocks. If not generated by the gpu-tracker API, must be either a CSV or SQLite file (where the SQLite file contains a table called "data") where the headers are precisely process_id, code_block_name, position, and timestamp. The process_id is the ID of the process where the code block is called. code_block_name is the name of the code block. position is whether it is the start or the stopping point of the code block where 0 represents start and 1 represents stop. And timestamp is the timestamp where the code block starts or where it stops.
        -p <file-path>          Paths to the sub-tracking files to combine. Must all be the same file format and the same file format as the resulting sub-tracking file (either .csv or .sqlite). If only one path is provided, it is interpreted as a path to a directory and all the files in this directory are combined.
        analyze                 Generate the sub-tracking analysis report using the tracking file and sub-tracking file for resource usage of specific code blocks.
        compare                 Compares multiple tracking sessions to determine differences in computational resource usage by loading sub-tracking results given their file paths. Sub-tracking results files must be in pickle format e.g. running the ``sub-track analyze`` command and specifying a file path for ``--output`` and 'pickle' for the ``--format`` option. If code block results are not included in the sub-tracking files (i.e. no code blocks were sub-tracked), then only overall results are compared.
        --cconfig=<config-file> JSON config file containing the ``file_path_map`` argument for the ``TrackerComparison`` class and ``statistic`` argument for its ``compare`` method (see API) that can be used instead of the corresponding ``-m <name>=<path>`` and ``--stat=<statistic>`` commandline options respectively. If additional ``-m <name>=<path>`` options are added on the commandline in addition to a config file, they will be added to the ``file_path_map`` in the config file. If a ``--stat`` option is provided on the commandline, it will override the ``statistic`` in the config file.
        -m <name>=<file-path>   Mapping of tracking session names to the path of the file containing the sub-tracking results of said tracking session. Must be in pickle format.
        --stat=<statistic>      The summary statistic of the measurements to compare. One of 'min', 'max', 'mean', or 'std'. Defaults to 'mean'.


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
       System: 5.61
       Main:
          Total RSS: 0.003
          Private RSS: 0.0
          Shared RSS: 0.003
       Descendants:
          Total RSS: 0.879
          Private RSS: 0.76
          Shared RSS: 0.119
       Combined:
          Total RSS: 0.881
          Private RSS: 0.761
          Shared RSS: 0.12
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
          Max sum percent: 324.8
          Max hardware percent: 27.067
          Mean sum percent: 152.109
          Mean hardware percent: 12.676
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendants:
          Max sum percent: 201.8
          Max hardware percent: 16.817
          Mean sum percent: 102.245
          Mean hardware percent: 8.52
       Combined:
          Max sum percent: 201.8
          Max hardware percent: 16.817
          Mean sum percent: 102.245
          Mean hardware percent: 8.52
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
descendant processes since, in this example, the bash command itself
calls the commands relevant to resource usage.*

Options
^^^^^^^

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

Alternative to typing out the tracking configuration via commandline
options, one can specify a config JSON file via the ``--tconfig``
option.

.. code:: none

    $ cat config.json


.. code:: none

    {
      "sleep_time": 0.5,
      "ram_unit": "megabytes",
      "gpu_ram_unit": "megabytes",
      "time_unit": "seconds"
    }


.. code:: none

    $ gpu-tracker -e 'bash example-script.sh' --tconfig=config.json


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM:
       Unit: megabytes
       System capacity: 67254.166
       System: 4511.437
       Main:
          Total RSS: 2.957
          Private RSS: 0.319
          Shared RSS: 2.638
       Descendants:
          Total RSS: 894.923
          Private RSS: 781.222
          Shared RSS: 113.701
       Combined:
          Total RSS: 896.135
          Private RSS: 781.541
          Shared RSS: 114.594
    Max GPU RAM:
       Unit: megabytes
       System capacity: 16376.0
       System: 727.0
       Main: 0.0
       Descendants: 314.0
       Combined: 314.0
    CPU utilization:
       System core count: 12
       Number of expected cores: 12
       System:
          Max sum percent: 259.3
          Max hardware percent: 21.608
          Mean sum percent: 160.9
          Mean hardware percent: 13.408
       Main:
          Max sum percent: 0.0
          Max hardware percent: 0.0
          Mean sum percent: 0.0
          Mean hardware percent: 0.0
       Descendants:
          Max sum percent: 102.8
          Max hardware percent: 8.567
          Mean sum percent: 96.529
          Mean hardware percent: 8.044
       Combined:
          Max sum percent: 102.8
          Max hardware percent: 8.567
          Mean sum percent: 96.529
          Mean hardware percent: 8.044
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
       Unit: seconds
       Time: 3.913


Sub-tracking
~~~~~~~~~~~~

Basics
^^^^^^

The ``sub-track`` subcommand introduces functionality related to
sub-tracking i.e. analyzing computational resource usage for individual
code blocks rather than the entire process. This requires a tracking
file and a sub-tracking file. The tracking file can be created by
specifying the ``--tf`` option when profiling a process using
``--execute``. The sub-tracking file can be created using the
gpu-tracker API i.e. the ``SubTracker`` class. If the process being
profiled is not a python script, the sub-tracking file can be generated
in any programming language as long as it follows the following format:

It is either a CSV or SQLite file where the headers are
``process_id,code_block_name,position,timestamp``. The ``process_id``
column is the ID (integer) of the process where the code block was
called. The ``code_block_name`` is the given name (string) of the code
block to distinguish it from other code blocks being sub-tracked. The
``position`` is an integer of either the value 0 or 1 where 0 indicates
the start of the code block and 1 indicates the stopping point of the
code block. Finally ``timestamp`` (float) is the timestamp when the code
block either starts (where ``position`` is 0) or when it stops (where
``position`` is 1). Both a start timestamp and stop timestamp must be
logged for every call to the code block of interest. If using an SQLite
file for more efficient querying of longer tracking sessions, the name
of the table must be ‘data’.

If sub-tracking a code block that is called in multiple processes, the
sub-tracking files of that code block must be unique to each process.
For convenience, the ``sub-track combine`` subcommand allows for
combining these into a single sub-tracking file that can be used for
downstream analysis. This example combines ‘sub-tracking1.csv’ and
‘sub-tracking2.csv’ into a single sub-tracking file of the name
‘combined-file.csv’. Alternatively, if the ``-p`` option is only used
once, rather than being interpretted as list of files, it is instead
interpretted as the path to a directory containing the sub-tracking
files to combine.

.. code:: none

    $ gpu-tracker sub-track combine --stf=combined-file.csv -p sub-tracking1.csv -p sub-tracking2.csv

Analysis
^^^^^^^^

Once a tracking and sub-tracking file is available, the
``sub-track analyze`` subcommand can generate the sub-tracking results.
These can be stored in JSON, text, or pickle format where the pickle
format is the same as the ``SubTrackingResults`` object from the API. If
the ``--output`` option is specified, the content can be stored in the
given file path. By default, the content prints to the screen and it is
in text format by default.

.. code:: none

    $ gpu-tracker sub-track analyze --tf=tracking.csv --stf=sub-tracking.csv


.. code:: none

    Overall:
    	                                                    min         max        mean        std
    	main_ram                                       0.341860    0.944374    0.856037   0.125014
    	descendants_ram                                0.000000    0.000000    0.000000   0.000000
    	combined_ram                                   0.341860    0.944374    0.856037   0.125014
    	system_ram                                     4.859711    5.553644    5.253445   0.134081
    	main_gpu_ram                                   0.000000    0.506000    0.429920   0.170432
    	descendants_gpu_ram                            0.000000    0.000000    0.000000   0.000000
    	combined_gpu_ram                               0.000000    0.506000    0.429920   0.170432
    	system_gpu_ram                                 0.215000    0.727000    0.650320   0.172010
    	gpu_sum_utilization_percent                    0.000000    3.000000    0.120000   0.600000
    	gpu_hardware_utilization_percent               0.000000    3.000000    0.120000   0.600000
    	main_n_threads                                12.000000   15.000000   14.720000   0.842615
    	descendants_n_threads                          0.000000    0.000000    0.000000   0.000000
    	combined_n_threads                            12.000000   15.000000   14.720000   0.842615
    	cpu_system_sum_utilization_percent            11.900000  133.400000  119.212000  22.741909
    	cpu_system_hardware_utilization_percent        0.991667   11.116667    9.934333   1.895159
    	cpu_main_sum_utilization_percent              78.000000  103.200000   96.924000   6.390767
    	cpu_main_hardware_utilization_percent          6.500000    8.600000    8.077000   0.532564
    	cpu_descendants_sum_utilization_percent        0.000000    0.000000    0.000000   0.000000
    	cpu_descendants_hardware_utilization_percent   0.000000    0.000000    0.000000   0.000000
    	cpu_combined_sum_utilization_percent          78.000000  103.200000   96.924000   6.390767
    	cpu_combined_hardware_utilization_percent      6.500000    8.600000    8.077000   0.532564
    Static Data:
    	   ram_unit gpu_ram_unit time_unit ram_system_capacity gpu_ram_system_capacity system_core_count n_expected_cores system_gpu_count n_expected_gpus
    	  gigabytes    gigabytes     hours           67.254166                  16.376                12               12                1               1
    Code Block Results:
    	Name:                my-code-block
    	Num Timepoints:      12
    	Num Calls:           3
    	Num Non Empty Calls: 3
    	Compute Time:
    		       min       max      mean       std     total
    		  2.580433  2.789909  2.651185  0.120147  7.953554
    	Resource Usage:
    		                                                    min         max        mean        std
    		main_ram                                       0.341860    0.936559    0.808736   0.167663
    		descendants_ram                                0.000000    0.000000    0.000000   0.000000
    		combined_ram                                   0.341860    0.936559    0.808736   0.167663
    		system_ram                                     4.859711    5.553644    5.231854   0.191567
    		main_gpu_ram                                   0.000000    0.506000    0.363500   0.225892
    		descendants_gpu_ram                            0.000000    0.000000    0.000000   0.000000
    		combined_gpu_ram                               0.000000    0.506000    0.363500   0.225892
    		system_gpu_ram                                 0.215000    0.727000    0.583250   0.228088
    		gpu_sum_utilization_percent                    0.000000    0.000000    0.000000   0.000000
    		gpu_hardware_utilization_percent               0.000000    0.000000    0.000000   0.000000
    		main_n_threads                                12.000000   15.000000   14.416667   1.164500
    		descendants_n_threads                          0.000000    0.000000    0.000000   0.000000
    		combined_n_threads                            12.000000   15.000000   14.416667   1.164500
    		cpu_system_sum_utilization_percent            11.900000  130.800000  113.641667  32.352363
    		cpu_system_hardware_utilization_percent        0.991667   10.900000    9.470139   2.696030
    		cpu_main_sum_utilization_percent              79.600000  103.100000   96.583333   6.726587
    		cpu_main_hardware_utilization_percent          6.633333    8.591667    8.048611   0.560549
    		cpu_descendants_sum_utilization_percent        0.000000    0.000000    0.000000   0.000000
    		cpu_descendants_hardware_utilization_percent   0.000000    0.000000    0.000000   0.000000
    		cpu_combined_sum_utilization_percent          79.600000  103.100000   96.583333   6.726587
    		cpu_combined_hardware_utilization_percent      6.633333    8.591667    8.048611   0.560549
    
    	Name:                my-function
    	Num Timepoints:      12
    	Num Calls:           3
    	Num Non Empty Calls: 3
    	Compute Time:
    		       min       max      mean       std     total
    		  2.538011  2.577679  2.553176  0.021419  7.659528
    	Resource Usage:
    		                                                     min         max        mean       std
    		main_ram                                        0.864592    0.944374    0.896998  0.034505
    		descendants_ram                                 0.000000    0.000000    0.000000  0.000000
    		combined_ram                                    0.864592    0.944374    0.896998  0.034505
    		system_ram                                      5.203415    5.315219    5.271566  0.038751
    		main_gpu_ram                                    0.314000    0.506000    0.490000  0.055426
    		descendants_gpu_ram                             0.000000    0.000000    0.000000  0.000000
    		combined_gpu_ram                                0.314000    0.506000    0.490000  0.055426
    		system_gpu_ram                                  0.535000    0.727000    0.711000  0.055426
    		gpu_sum_utilization_percent                     0.000000    3.000000    0.250000  0.866025
    		gpu_hardware_utilization_percent                0.000000    3.000000    0.250000  0.866025
    		main_n_threads                                 15.000000   15.000000   15.000000  0.000000
    		descendants_n_threads                           0.000000    0.000000    0.000000  0.000000
    		combined_n_threads                             15.000000   15.000000   15.000000  0.000000
    		cpu_system_sum_utilization_percent            120.300000  133.400000  124.566667  4.001439
    		cpu_system_hardware_utilization_percent        10.025000   11.116667   10.380556  0.333453
    		cpu_main_sum_utilization_percent               94.700000  103.200000   98.841667  2.677332
    		cpu_main_hardware_utilization_percent           7.891667    8.600000    8.236806  0.223111
    		cpu_descendants_sum_utilization_percent         0.000000    0.000000    0.000000  0.000000
    		cpu_descendants_hardware_utilization_percent    0.000000    0.000000    0.000000  0.000000
    		cpu_combined_sum_utilization_percent           94.700000  103.200000   98.841667  2.677332
    		cpu_combined_hardware_utilization_percent       7.891667    8.600000    8.236806  0.223111
    
    


The overall resource usage of the tracking session is provided as well
as its static data. This is followed by the compute time and resource
usage of each code block.

Comparison
^^^^^^^^^^

.. code:: python3

    Storing the results of the sub-tracking analysis in a pickle file allows for one tracking session to be compared to another.

.. code:: none

    $ gpu-tracker sub-track analyze --tf=tracking.csv --stf=sub-tracking.csv --format=pickle --output=my-results.pkl

The ``sub-track compare`` subcommand compares the computational resource
usage of multiple tracking sessions. This is useful when you want to
determine how a change can impact the computational efficiency of your
process, whether it be different input data, an alternative
implementation, etc. The ``-m`` option creates a mapping from the given
name of a tracking session to the file path where its sub-tracking
results are stored in pickle format. Say you wanted to call one tracking
session ‘A’ and then the second tracking session ‘B’ where the results
of tracking session ‘A’ are stored in ‘results.pkl’ and that of session
‘B’ are in ‘results2.pkl’.

.. code:: none

    $ gpu-tracker sub-track compare -m A=results.pkl -m B=results2.pkl


.. code:: none

    Overall Resource Usage:
    	Main Ram:
    		         B         A
    		  0.856037  0.861921
    	Descendants Ram:
    		    A    B
    		  0.0  0.0
    	Combined Ram:
    		         B         A
    		  0.856037  0.861921
    	System Ram:
    		         B         A
    		  5.253445  5.281926
    	Main Gpu Ram:
    		        B         A
    		  0.42992  0.448364
    	Descendants Gpu Ram:
    		    A    B
    		  0.0  0.0
    	Combined Gpu Ram:
    		        B         A
    		  0.42992  0.448364
    	System Gpu Ram:
    		        B         A
    		  0.65032  0.668909
    	Gpu Sum Utilization Percent:
    		    A     B
    		  0.0  0.12
    	Gpu Hardware Utilization Percent:
    		    A     B
    		  0.0  0.12
    	Main N Threads:
    		      B          A
    		  14.72  14.757576
    	Descendants N Threads:
    		    A    B
    		  0.0  0.0
    	Combined N Threads:
    		      B          A
    		  14.72  14.757576
    	Cpu System Sum Utilization Percent:
    		        B           A
    		  119.212  121.918182
    	Cpu System Hardware Utilization Percent:
    		         B          A
    		  9.934333  10.159848
    	Cpu Main Sum Utilization Percent:
    		       B          A
    		  96.924  99.060606
    	Cpu Main Hardware Utilization Percent:
    		      B         A
    		  8.077  8.255051
    	Cpu Descendants Sum Utilization Percent:
    		    A    B
    		  0.0  0.0
    	Cpu Descendants Hardware Utilization Percent:
    		    A    B
    		  0.0  0.0
    	Cpu Combined Sum Utilization Percent:
    		       B          A
    		  96.924  99.060606
    	Cpu Combined Hardware Utilization Percent:
    		      B         A
    		  8.077  8.255051
    Code Block Resource Usage:
    	Main Ram:
    my-code-block:
    			         B         A
    			  0.808736  0.846999
    my-function:
    			         A         B
    			  0.888034  0.896998
    	Descendants Ram:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A    B
    			  0.0  0.0
    	Combined Ram:
    my-code-block:
    			         B         A
    			  0.808736  0.846999
    my-function:
    			         A         B
    			  0.888034  0.896998
    	System Ram:
    my-code-block:
    			         A         B
    			  5.170665  5.231854
    my-function:
    			         B         A
    			  5.271566  5.476632
    	Main Gpu Ram:
    my-code-block:
    			       B         A
    			  0.3635  0.415429
    my-function:
    			     B      A
    			  0.49  0.506
    	Descendants Gpu Ram:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A    B
    			  0.0  0.0
    	Combined Gpu Ram:
    my-code-block:
    			       B         A
    			  0.3635  0.415429
    my-function:
    			     B      A
    			  0.49  0.506
    	System Gpu Ram:
    my-code-block:
    			        B         A
    			  0.58325  0.635714
    my-function:
    			      B      A
    			  0.711  0.727
    	Gpu Sum Utilization Percent:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A     B
    			  0.0  0.25
    	Gpu Hardware Utilization Percent:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A     B
    			  0.0  0.25
    	Main N Threads:
    my-code-block:
    			          B          A
    			  14.416667  14.619048
    my-function:
    			     A     B
    			  15.0  15.0
    	Descendants N Threads:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A    B
    			  0.0  0.0
    	Combined N Threads:
    my-code-block:
    			          B          A
    			  14.416667  14.619048
    my-function:
    			     A     B
    			  15.0  15.0
    	Cpu System Sum Utilization Percent:
    my-code-block:
    			           B           A
    			  113.641667  120.142857
    my-function:
    			           B        A
    			  124.566667  125.025
    	Cpu System Hardware Utilization Percent:
    my-code-block:
    			         B          A
    			  9.470139  10.011905
    my-function:
    			          B         A
    			  10.380556  10.41875
    	Cpu Main Sum Utilization Percent:
    my-code-block:
    			          B          A
    			  96.583333  98.652381
    my-function:
    			          B       A
    			  98.841667  99.775
    	Cpu Main Hardware Utilization Percent:
    my-code-block:
    			         B         A
    			  8.048611  8.221032
    my-function:
    			         B         A
    			  8.236806  8.314583
    	Cpu Descendants Sum Utilization Percent:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A    B
    			  0.0  0.0
    	Cpu Descendants Hardware Utilization Percent:
    my-code-block:
    			    A    B
    			  0.0  0.0
    my-function:
    			    A    B
    			  0.0  0.0
    	Cpu Combined Sum Utilization Percent:
    my-code-block:
    			          B          A
    			  96.583333  98.652381
    my-function:
    			          B       A
    			  98.841667  99.775
    	Cpu Combined Hardware Utilization Percent:
    my-code-block:
    			         B         A
    			  8.048611  8.221032
    my-function:
    			         B         A
    			  8.236806  8.314583
    Code Block Compute Time:
    my-code-block:
    		         B        A
    		  2.651185  2.68558
    my-function:
    		         B         A
    		  2.553176  2.559218
    


Both the overall usage is compared and per code block. The default
format is text and the default output is printing to the console. The
``--format`` and ``--output`` options can be configured similarly to
those in the ``sub-track analyze`` subcommand. By default, the ‘mean’ of
measurements is compared. Alternatively, the ``--stat`` option can be
set to ‘min’, ‘max’, or ‘std’ to compare a different summary statistic.
