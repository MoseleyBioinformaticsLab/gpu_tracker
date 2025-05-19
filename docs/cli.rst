CLI
===
The ``gpu-tracker`` command-line interface allows tracking computational-resource-usage of an arbitrary shell command.
For example, one may want to profile a command that runs a script or a command ran in a high-performance-computing job.
Below is the help message shown from ``gpu-tracker --help``.
See the CLI section of the :ref:`tutorial-label` for examples of using the CLI.

.. literalinclude:: ../src/gpu_tracker/__main__.py
    :start-at: Usage:
    :end-before: """
    :language: none
