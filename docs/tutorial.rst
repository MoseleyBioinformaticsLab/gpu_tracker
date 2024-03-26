Tutorial
========

.. code:: python3

    import gpu_tracker as gput

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
used by the code that’s in between ``start()`` and ``stop()``. The
``Tracker`` class additionally has a ``__str__`` method so it can be
printed as a string that formats the values and units of each
computational resource.

.. code:: python3

    tracker = gput.Tracker()
    tracker.start()
    # Perform expensive operations
    tracker.stop()
    print(tracker)


.. code:: none

    Max RAM: 0.068 gigabytes
    Max GPU RAM: 0.000 gigabytes
    Compute time: 0.000 hours


The equivalent can be accomplished using ``Tracker`` as a context
manager rather than explicitly calling ``start()`` and ``stop()``.

.. code:: python3

    with gput.Tracker() as tracker:
        # Perform expensive operations
        pass
    print(tracker)


.. code:: none

    Max RAM: 0.068 gigabytes
    Max GPU RAM: 0.000 gigabytes
    Compute time: 0.000 hours


The units of the computational resources can be modified as desired. For
example, to measure the RAM in megabytes, the GPU RAM in kilobytes, and
the compute time in seconds:

.. code:: python3

    with gput.Tracker(ram_unit='megabytes', gpu_ram_unit='kilobytes', time_unit='seconds') as tracker:
        # Perform expensive operations
        pass
    print(tracker)


.. code:: none

    Max RAM: 67.949 megabytes
    Max GPU RAM: 0.000 kilobytes
    Compute time: 1.031 seconds


The same information can be obtained in a dictionary via the
``Tracker``\ ’s ``to_json()`` method.

.. code:: python3

    tracker.to_json()




.. code:: none

    {'max_ram': 67.948544,
     'ram_unit': 'megabytes',
     'max_gpu_ram': 0.0,
     'gpu_ram_unit': 'kilobytes',
     'compute_time': 1.0309913158416748,
     'time_unit': 'seconds'}



Additionally, the individual measurements and units are available as
attributes in the ``Tracker`` class.

.. code:: python3

    print(tracker.max_ram, tracker.ram_unit)


.. code:: none

    67.948544 megabytes


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
        gpu-tracker --execute=<command> [--output=<output>] [--format=<format>] [--st=<sleep-time>] [--ic] [--ru=<ram-unit>] [--gru=<gpu-unit>] [--tu=<time-unit>]
    
    Options:
        -h --help               Show this help message.
        -e --execute=<command>  The command to run along with its arguments all within quotes e.g. "ls -l -a".
        -o --output=<output>    File path to store the computational-resource-usage measurements. If not set, prints measurements to the screen.
        -f --format=<format>    File format of the output. Either 'json' or 'text'. Defaults to 'text'.
        --st=<sleep-time>       The number of seconds to sleep in between usage-collection iterations.
        --ic                    Stands for include-children; Whether to add the usage (RAM and GPU RAM) of child processes. Otherwise, only collects usage of the main process.
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
    Max RAM: 0.002 gigabytes
    Max GPU RAM: 0.000 gigabytes
    Compute time: 0.001 hours
    


Like with the API, the units can be modified. For example, –tu stands
for time-unit and –ru stands for ram-unit.

.. code:: none

    $ gpu-tracker -e 'sleep 2' --tu=seconds --ru=megabytes


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    Max RAM: 1.835 megabytes
    Max GPU RAM: 0.000 gigabytes
    Compute time: 2.050 seconds
    


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

    Max RAM: 0.002 gigabytes
    Max GPU RAM: 0.000 gigabytes
    Compute time: 0.001 hours


By default, the format of the output is “text”. The ``-f`` or
``--format`` option can specify the format to be “json” instead.

.. code:: none

    $ gpu-tracker -e 'sleep 2' -f json


.. code:: none

    Resource tracking complete. Process completed with status code: 0
    {
     "max_ram": 0.002097152,
     "ram_unit": "gigabytes",
     "max_gpu_ram": 0.0,
     "gpu_ram_unit": "gigabytes",
     "compute_time": 0.0005690234899520874,
     "time_unit": "hours"
    }


.. code:: none

    $ gpu-tracker -e 'sleep 2' -f json -o out.json


.. code:: none

    Resource tracking complete. Process completed with status code: 0


.. code:: none

    $ cat out.json


.. code:: none

    {
     "max_ram": 0.0018350080000000002,
     "ram_unit": "gigabytes",
     "max_gpu_ram": 0.0,
     "gpu_ram_unit": "gigabytes",
     "compute_time": 0.0005691042873594496,
     "time_unit": "hours"
    }
