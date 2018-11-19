#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from sys import version

__version__ = float(version[0:3])
__cwd__ = os.getcwd()

def loadCfg(fname):
    fname = __cwd__ + '/' + fname
    try:
        if __version__ >= 3.5:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", fname)
            cf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cf)
        elif __version__ >= 3.3:
            from importlib.machinery import SourceFileLoader
            cf = SourceFileLoader("config",fname).load_module()
        elif __version__ <= 3.0:
            import imp
            cf = imp.load_source('config',fname) 
    except:
        print('Failed. Cannot find file <{}> or the fallback <config.py>'.format(fname))
        print('Or invalid line found in file.')
        exit(1)
    return cf

def loadVariables(mod):
    for k in dir(mod):
        if '__' not in k:
            globals()[k] = getattr(mod,k)


def parseAEI(fname):
    '''parses .aei files intelligently
    returns objectname and data in a tuple
    '''
    def objectName(fname):
        '''discerns name of object from header
        '''
        def notFound(line):
            '''iterating through lines and pulls first non blank line
            as the object name
            '''
            line = line.replace(' ','')
            if line != '':
                return False,line
            else:
                return True,None
        with open(fname,'r') as f:
            for row in f:
                row = row.replace('\n','')
                if not notFound(row)[0]:
                    return row.replace(' ','')
    def getHeader(fname,num=3):
        with open(fname,'r') as f:
            for i,row in enumerate(f):
                if i == num:
                    return [x for x in row.replace(' (','').replace(')','').strip(' ').strip('\n').split(' ') if x != '']
    file = np.loadtxt(fname,skiprows=4,dtype=float)
    return objectName(fname),getHeader(fname),file

def choose2Pairs(head):
    n = len(head)
    numpairs = n*(n-1)/2.
    pairs = []
    for i,x in enumerate(head):
        for j,y in enumerate(head):
            if j > i:
                pairs.append(x+' '+y)
    return numpairs,pairs

def verifyDir(name):
    if not os.path.isdir(name):
        os.mkdir(name)

def plotting(x,y,xl,yl,title=None,fig=None,ax=None):
    if (fig == None) and (ax == None):
        fig,ax = plt.figure(figsize=(10,10))
    ax.scatter(x,y)
    ax.set_xlim(min(x),max(x))
    ax.set_ylim(min(y),max(y))
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)

def flip(orig):
    toReturn = np.ndarray(orig.shape[::-1])
    temp = []

    for i,row in enumerate(orig):
        for j,col in enumerate(row):
            toReturn[j,i] = col

    return toReturn

def main(configFname):
    '''Main calling function for the program
    @input : configFname is the name of the configuration file. Has a default value just incase
    Loads in all of the values found in the configuration file
    '''
    config = loadCfg(configFname)
    loadVariables(config)
    print(dir())

    for i,f in enumerate(files):
        oName,header,odata = parseAEI(f)
        data = flip(odata)#.reshape(len(header),-1)

        for x in choose2Pairs(header)[1]:
            print(x)
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            p1,p2 = x.split(' ')
            outFile = output.replace('-N-',oName).replace('-P-',''.join([p1,p2]))
            direct = outFile.split('/')[0]
            verifyDir(direct)
            x = data[header.index(p1),:]
            y = data[header.index(p2),:]
            plotting(x,y,p1,p2,title=''.join([p1,p2]),fig=fig,ax=ax)
            plt.autoscale(enable=True, axis='both', tight=None)
            plt.tight_layout(pad=1.02,h_pad=None, w_pad=None, rect=None)
            plt.savefig(outFile,dpi=200)
            plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', dest='input', type=str,help='input',required=True)

    args = parser.parse_args()
    
    main(args.input)

