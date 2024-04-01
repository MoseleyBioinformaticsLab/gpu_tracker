###########
gpu_tracker
###########
Description
-----------
The ``gpu_tracker`` package provides a ``Tracker`` class and a commandline-interface that tracks (profiles) the usage of compute time, maximum RAM, and maximum GPU RAM.
The compute time is a measurement of the real time taken by the task as opposed to the CPU-utilization time.
The GPU tracking is for Nvidia GPUs and uses the ``nvidia-smi`` command, assuming the Nvidia drivers have been installed.
Computational resources are tracked throughout the duration of a context manager or the duration of explicit calls to the ``start()`` and ``stop()`` methods of the ``Tracker`` class.
The ``gpu-tracker`` command-line interface alternatively tracks the computational-resource-usage of an arbitrary shell command.

Documentation
-------------
The complete documentation for the ``gpu_tracker`` package, including tutorials, can be found `here <https://moseleybioinformaticslab.github.io/gpu_tracker/>`__.

Installation
------------
Requires python 3.10 and above.

Install on Linux, Mac OS X
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. parsed-literal::
   python3 -m pip install gpu-tracker

Install on Windows
~~~~~~~~~~~~~~~~~~
.. parsed-literal::
   py -3 -m pip install gpu-tracker

PyPi
~~~~
See our PyPi page `here <https://pypi.org/project/gpu-tracker/>`__.

Questions, Feature Requests, and Bug Reports
--------------------------------------------
Please submit any questions or feature requests you may have and report any potential bugs/errors you observe on `our GitHub issues page <https://github.com/MoseleyBioinformaticsLab/gpu_tracker/issues>`__.

GitHub Repository
-------------------
Code is available on GitHub: https://github.com/MoseleyBioinformaticsLab/gpu_tracker.
