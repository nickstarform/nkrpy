#!/usr/bin/env python
'''
JUST FOR RESUBMITTING JOBS FOR SOME TIME
Has a few failsafes built in: Min refresh time 1hr, 
max runtime 240 hrs, max slurm calls 20, slurm batch files must exist
can only have 2 of same job in queue at same time
Must have SBATCH TIME and SBATCH JOB_NAME in slurm file
'''
import subprocess
from os import system,getcwd
from os.path import isfile
from argparse import ArgumentParser
import getpass
from time import sleep
CWD = getcwd()

if __name__ == "__main__":
    description = "For resubmitting jobs on oscer using SLURM"
    parser = ArgumentParser(description=description)
    parser.add_argument('-i','--input',type=str,\
        help="input slurm loader files single or comma separated",dest='inp',required=True)
    parser.add_argument('-r','--refresh',type=float,\
        help="input refresh rate for checking jobs in hours",dest='ref',default=1)
    parser.add_argument('-t','--total',type=float,\
        help="input total time in hours for run to go,max 240",dest='total',default=240)
    parser.add_argument('-v','--verbose',dest='verbose',action='store_true',default=False)
    
    args = parser.parse_args()
    lof = args.inp.split(',')

    # force certain parameters
    assert args.total <= 240 # 5 x 48hour runs max
    assert 1 <= args.ref <= 24 # force checks per hour or less
    for x in lof: # make sure all files exist
        assert isfile(x)

    # determine initials
    user = getpass.getuser()
    timer = 0
    MAXCOUNT = 0 # prevents global spamming
    MASTERID = [] # holds all ending IDS
    while (timer < args.total) and (MAXCOUNT < 20):
        f = open("script.log",'a')
        timer = round(timer,2)
        if args.verbose:
            response = "Current timer: {} hours".format(timer)
            f.write(response + '\n')
            print(response)
        # get all jobs by user at once to reduce overhead
        process = ["squeue","-u",user]
        proc    = subprocess.Popen(process, stdout=subprocess.PIPE)
        output  = proc.stdout.read().decode("utf-8")
        output  = [x.strip().split() for x in output.split('\n')[1:-1]]
        # output is 8*x num jobs

        # loop through all configs
        for x in lof:
            counter = 0
            startrun = False
            # find things in configs
            with open(x,'r') as f:
                al = f.readlines()
            phrase1 = 'time'
            phrase2 = 'job-name'
            for a in al:
                if (phrase1 in a) and ('#SBATCH' in a):
                    fin1 = a.split('=')[1]
                elif (phrase2 in a) and ('#SBATCH' in a):
                    fin2 = a.split('=')[1]

            Truntime = fin1
            jobname  = fin2
            Mruntime = int(Truntime.split(':')[0])

            if len(fin1.split('-')) > 1:
                Mruntime = int(Mruntime.split('-')[0]) * 24 + int(Mruntime.split('-')[1])
            else:
                Mruntime = int(Mruntime)
            Mruntime = Mruntime / 2

            # see if jobs in output are the same ones from configs
            for y in output:
                #print(jobname,y[2],Mruntime,y[5])
                jname = y[2]
                if (jname in jobname) and (y[5] != "0:00"):
                    temp = y[5].split(":")[0]
                    counter +=1
                    if len(temp.split("-")) > 1:
                        atime = int(temp.split('-')[0]) * 24 + int(temp.split('-')[1])
                    else:
                        atime = int(temp)
                elif (jname in jobname) and (y[5] == "0:00"):
                    counter = 99

            if counter == 0:
                # run job
                startrun = True
            elif counter > 1:
                # dont run
                startrun = False
            else:
                bound = args.ref/2
                if (Mruntime-bound) <= atime:
                    startrun = True
                else: 
                    startrun = False
            #print(startrun)

            # start 
            #print(x,'...',counter,'...',startrun)
            if startrun:
                MAXCOUNT += 1
                if args.verbose:
                    print("Started: ",timer,'hours')
                temp = ["sbatch",x]
                temp = subprocess.Popen(temp, stdout=subprocess.PIPE)
                MASTERID.append(temp.stdout.read().decode("utf-8"))
                print(MASTERID[-1])

        f.close()
        timer += args.ref
        sleep(timer*3600)
    response = "Finished all queues, IDs created: {}".format(MASTERID)
    print(response)
    f.write(response + '\n')
    response = "Total time in queue: {}hours, total jobs submitted: {}".format(timer,len(MASTERID))
    print(response)
    f.write(response + '\n')
# eof
