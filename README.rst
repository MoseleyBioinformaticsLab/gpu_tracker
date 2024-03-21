###########
gpu_tracker
###########
Description
-----------
The ``gpu_tracker`` package provides a ``Tracker`` class that tracks the compute time, maximum RAM, and maximum GPU memory usage.
The GPU tracking is for Nvidia GPUs and uses the ``nvidia-smi`` command, assuming the Nvidia drivers have been installed.
Computational resources are tracked throughout the duration of a context manager or the duration of explicit calls to the ``start()`` and ``stop()`` methods of the ``Tracker`` class.

Documentation
-------------
The complete documentation for the ``gpu_tracker`` package, including tutorials, can be found `here <https://moseleybioinformaticslab.github.io/gpu_tracker/>`__.
