"""
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
"""
import docopt as doc
import subprocess as subp
import json
import logging as log
import sys
from . import Tracker


def main():
    args = doc.docopt(__doc__)
    command = args['--execute'].split(' ')
    output = args['--output']
    output_format = args['--format'] if args['--format'] is not None else 'text'
    option_map = {
        '--st': 'sleep_time',
        '--ru': 'ram_unit',
        '--gru': 'gpu_ram_unit',
        '--tu': 'time_unit'
    }
    kwargs = {
        option_map[option]: value for option, value in args.items() if value is not None and option not in {
            '--execute', '--output', '--format'}}
    if 'sleep_time' in kwargs.keys():
        kwargs['sleep_time'] = float(kwargs['sleep_time'])
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
        raise ValueError(f'"{output_format} is not a valid format. Valid values are "json" or "text".')
    if output is None:
        print(output_str)
    else:
        with open(output, 'w') as file:
            file.write(output_str)
