"""
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
"""
import docopt as doc
import subprocess as subp
import json
import logging as log
import sys
from . import Tracker
from . import __version__


def main():
    args = doc.docopt(__doc__, version=__version__)
    command = args['--execute'].split()
    output = args['--output']
    output_format = args['--format'] if args['--format'] is not None else 'text'
    option_map = {
        '--st': 'sleep_time',
        '--ru': 'ram_unit',
        '--gru': 'gpu_ram_unit',
        '--tu': 'time_unit',
        '--nec': 'n_expected_cores',
        '--guuids': 'gpu_uuids',
        '--disable-logs': 'disable_logs'
    }
    kwargs = {
        option_map[option]: value for option, value in args.items() if value is not None and option not in {
            '--execute', '--output', '--format', '--help', '--version'}}
    if 'sleep_time' in kwargs.keys():
        kwargs['sleep_time'] = float(kwargs['sleep_time'])
    if 'n_expected_cores' in kwargs.keys():
        kwargs['n_expected_cores'] = int(kwargs['n_expected_cores'])
    if 'gpu_uuids' in kwargs.keys():
        kwargs['gpu_uuids'] = set(kwargs['gpu_uuids'].split(','))
    if len(command) == 0:
        log.error('Empty command provided.')
        sys.exit(1)
    try:
        process = subp.Popen(command)
    except FileNotFoundError:
        log.error(f'Command not found: "{command[0]}"')
        sys.exit(1)
    except Exception as e:
        log.error(f'The following error occurred when starting the command "{command[0]}":')
        print(e)
        sys.exit(1)
    with Tracker(process_id=process.pid, **kwargs) as tracker:
        process.wait()
    print(f'Resource tracking complete. Process completed with status code: {process.returncode}')
    if output_format == 'json':
        output_str = json.dumps(tracker.to_json(), indent=1)
    elif output_format == 'text':
        output_str = str(tracker)
    else:
        log.error(f'"{output_format}" is not a valid format. Valid values are "json" or "text".')
        sys.exit(1)
    if output is None:
        print(output_str)
    else:
        with open(output, 'w') as file:
            file.write(output_str)
