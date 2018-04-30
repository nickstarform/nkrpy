#!/usr/bin/env python
'''
Name  : Functions, function.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : Holds the highly used and general functions
'''

# import standard modules
from collections import Iterable
import math

def typecheck(obj): 
    '''
    Checks if object is iterable (array,list,tuple) and not string
    '''
    return not isinstance(obj, str) and isinstance(obj, Iterable)

def checkconv(coord):
    '''
    will take a coord convert to a list
    return the decimal conversion
    '''
    delimiters = [':',' ',',']
    failed = False
    orig = coord
    # if a string, handle
    if type(coord) == float:
        return coord
    if type(coord) == str:
        # check if float hidden as string
        try:
            return float(coord)
        except:
            # handle string proper by going through all delimiters
            count = 0
            while not typecheck(coord):
                #print(count)
                coord = coord.split(delimiters[count])
                if len(coord) == 1:
                    coord = coord[0]
                if count == len(delimiters) - 1:
                    failed = True
                    break
                count += 1
    # try with different scheme
    if failed == True:
        hold = coord
        sp = 'h'
        t0 = hold.split(sp)[0]
        if t0 == hold:
            sp = 'd'
            t0 = hold.split(sp)[0]


        if len([x for x in hold.split(sp) if x != '']) == 1:
            coord = [t0]
        else:
            t1 = hold.split(sp)[1].split('m')[0]
            if len(t1.split('.')) == 1:
                t2 = hold.split(sp)[1].split('m')[1].strip('s')
            else:
                t2 = ''
            hold = [t0,t1,t2]
            hold = [x for x in hold if ((x != '' )and (type(x) == str))]
            coord = hold

    # if list, make to float, final form
    #print(len(coord),orig)
    if typecheck(coord):
        if len(coord) == 3:
            temp0,temp1,temp2 = map(float,coord)
            total = abs(temp0) + temp1/60. + temp2/3600
        elif len(coord) == 2:
            temp0,temp1 = map(float,coord)
            temp2 = temp1/60. - temp1
            total = abs(temp0) + temp1/60. + temp2/3600
        elif len(coord) == 1:
            # if an iterable of len one with float
            try:
                return float(coord[0])
            except:
                return checkconv(coord[0])
        if temp0 < 0.:
            total = total * -1
    return total

