#!/usr/bin/env python3

import subprocess
import argparse
import os
import sys
import importlib.util as ilu
from pathlib import Path
# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', help='Only print the files to run.')
parser.add_argument('-b', '--benchmark', action='store_true', help='Only return `test` versions of the runfiles')
args = parser.parse_args()

# Get the name of the computer.
username = os.getlogin() if args.username == '' else args.username
computername = os.uname() if args.computername == '' else args.computername



###############################################################
os.chdir('/home/'+username)

def run(cmd, test=False):
    if args.test or test:
        print(cmd)
        return
    os.system(cmd)

# Loop through and determine if a job is running.
for projectname,project in projects.items():
    # Loop through each of the sources in a project.
    for source, models in project.items():
        # Check what processes are in the queue for this source.
        for modelname in models:
            modelpath = '/'.join(['/home', username, projectname, source, modelname]) + '/'
            modelname = ('-'.join([projectname, source, modelname])).lower()
            result = subprocess.Popen(['squeue','-u',username,'-o',\
                    '"%.10i %.9P %.16j %.2t %.10M %.5D %R"'], \
                    stdout=subprocess.PIPE)
            jobscript = modelpath+"queue_XXXX_"+computer
            if args.benchmark:
                run(f'sbatch {jobscript.replace("XXXX", "test")}', test=True)
                continue
            try:
                output = subprocess.check_output(('grep','({0:s}-(new)?(resume)?)'.format(modelname)), \
                        stdin=result.stdout).split(b'\n')
            except subprocess.CalledProcessError:
                # If none were found, submit a job.
                # check if job was run before
                if os.path.exists(modelpath + 'results.hdf5'):
                    run(f'sbatch {jobscript.replace("XXXX", "resume")}')
                    continue
                run(f'sbatch {jobscript.replace("XXXX", "new")}')
                continue

            # If multiple jobs were found, pass.

            if len(output) > 2:
                continue

            # If a job has run for less than 24 hours, pass.

            if len(output[0].split()[5].decode("utf-8").split("-")) < 2:
                continue

            # Otherwise, submit a new process that is dependent on the old one
            # finishing.

            jobnumber = output[0].split()[1].decode("utf-8")

            run(f'sbatch --dependency=afterany:{jobscript.replace("XXXX", "resume")} {jobnumber}')