"""
Usage:
    gpu-tracker -e=<command>

Options:
    -e  The command to run along with its arguments all within quotes e.g. "ls -l -a".
"""
import docopt as doc
import subprocess as subp
from . import tracker as track


def main():
    doc.docopt()
    with subp.Popen(['ls', '-la']) as process:
        with track.Tracker(process_id=process.pid) as tracker:
            process.wait()
    print(tracker)
