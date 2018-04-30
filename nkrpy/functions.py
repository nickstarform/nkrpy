# Misc Functions
import numpy as np
from math import cos,sin,acos,ceil

def linear(x,a,b):
    return a*x + b

def binning(data,width=3):
    return data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return [idx,array[idx]]

# function for formatting
def addspace(sstring,spacing=20):
    while True:
        if len(sstring) >= spacing:
            sstring = sstring[:-1]
        elif len(sstring) < (spacing -1):
            sstring = sstring + ' '
        else: 
            break
    return sstring + ' '

def ang_vec(deg):
    rad = deg*pi/180.
    return (cos(rad),sin(rad))

def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]

def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    while np.abs(cosx) >= 1:
        cosx = cosx/(np.abs(cosx)*1.001)
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees

def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det>0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def gen_angles(start,end,resolution=1,direction='+'):
    if direction == "+":# positive direction
        diff = round(angle_clockwise(ang_vec(start),ang_vec(end)),2)
        numd = int(ceil(diff/resolution))+1
        final = [round((start + x*resolution),2)%360 for x in range(numd)]
    elif direction == "-": # negative direction
        diff = round(360.-angle_clockwise(ang_vec(start),ang_vec(end)),2)
        numd = int(ceil(diff/resolution))+1
        final = [round((start - x*resolution),2)%360  for x in range(numd)]
    return final