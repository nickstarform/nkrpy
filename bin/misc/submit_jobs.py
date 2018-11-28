#!/usr/bin/env python3

import subprocess
import argparse
import os

# Parse arguments.

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--test', action='store_true')
args = parser.parse_args()

# Objects to submit jobs for.

disk_masses = {"sources":['CRBR12','GSS30-IRS3','LFAM26','WL17'], \
        "jobscript":"queue_{0:s}_fit_schooner"}

hops_transition_disks = {"sources":['HH270MMS2','HOPS-65','HOPS-124', \
        'HOPS-140','HOPS-157','HOPS-163'], \
        "jobscript":"queue_{0:s}_fit_schooner"}

bhr7 = {"sources":['BHR7'], "jobscript":"queue_{0:s}_cont_schooner"}

L1527 = {"sources":['L1527'], "jobscript":"queue_{0:s}_diana_schooner"}

hops370 = {"sources":['HOPS-370'], "jobscript":"queue_{0:s}_cont_schooner"}

# Get the total projects dictionary.

projects = {"DiskMasses":disk_masses, \
        "HOPSTransitionDisks":hops_transition_disks, \
        "HOPS-370":hops370, \
        "L1527":L1527, \
        "BHR7":bhr7, \
        }

# Loop through and determine if a job is running.

for project in projects:
    # Change directories to the project.

    os.chdir("{0:s}/Analysis".format(project))

    # Now get the dictionary for that project.

    project = projects[project]

    # Loop through each of the sources in a project.

    for source in project["sources"]:
        # Check what processes are in the queue for this source.

        result = subprocess.Popen(['squeue','-u','psheehan','-o',\
                '"%.10i %.9P %.15j %.2t %.10M %.6D %R"'], \
                stdout=subprocess.PIPE)
        try:
            output = subprocess.check_output(('grep','{0:s}'.format(source)), \
                    stdin=result.stdout).split(b'\n')
        except subprocess.CalledProcessError:
            # If none were found, submit a job.

            print('sbatch {0:s}'.format(project["jobscript"]).format(source))
            if not args.test:
                os.system('sbatch {0:s}'.format(project["jobscript"]).\
                        format(source))
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

        print('sbatch --dependency=afterany:{1:s} {0:s}'.\
                format(project["jobscript"], jobnumber).format(source))
        if not args.test:
            os.system('sbatch --dependency=afterany:{1:s} {0:s}'.\
                    format(project["jobscript"], jobnumber).format(source))

    # Return to the Projects directory.

    os.chdir("../..")
