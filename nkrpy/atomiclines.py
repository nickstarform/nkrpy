#!/usr/bin/env python
'''
Name  : Atomic Lines, atomiclines.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : Houses all useful atomic lines and short program for parsing
        lines held here: atomiclines.lineregion.linename.{val and unit}
        to call parser, at = atomiclines.lines() -> at(*params)  -> return_lines()

'''
# import standard modules
from copy import deepcopy
import numpy as np

atomiclines={\
    'nir':{\
        "Al I":{'val':[1.3115,1.67],'unit':'micron'},\
        "Ar III":{'val':[7137.8,7753.2],'unit':'angstrom'},\
        'brg':{'val':[2.16611],'unit':'micron'},'Br5-4':{'val':[4.05226],'unit':'microns'},'Br6-4':{'val':[2.62587],'unit':'microns'},\
        'Br8-4':{'val':[1.94509],'unit':'microns'},'Br9-4':{'val':[1.81791],'unit':'microns'},'Br10-4':{'val':[1.73669],'unit':'microns'},\
        'Br11-4':{'val':[1.68111],'unit':'microns'},'Br12-4':{'val':[1.64117],'unit':'microns'},'Br13-4':{'val':[1.61137],'unit':'microns'},\
        'Br14-4':{'val':[1.58849],'unit':'microns'},'Br15-4':{'val':[1.57050],'unit':'microns'},'Br16-4':{'val':[1.55607],'unit':'microns'},\
        'Br17-4':{'val':[1.54431],'unit':'microns'},'Br18-4':{'val':[1.53460],'unit':'microns'},'Br19-4':{'val':[1.52647],'unit':'microns'},\
        'Br20-4':{'val':[1.51960],'unit':'microns'},'Br21-4':{'val':[1.51374],'unit':'microns'},\
        "Ca I":{'val':[1.9442,1.9755,2.2605,2.263,2.266],'unit':'micron'},\
        #"Ca II":{'val':[3933.663,3968.468,8500.36, 8544.44,8664.52],'unit':'angstrom'},\
        "CO":{'val':[2.2925,2.3440,2.414,2.322,2.352,2.383],'unit':'micron'},\
        #"Fe I":{'val':[1.1880,1.1970],'unit':'micron'},\
        "Fe II":{'val':[1.688,1.742],'unit':'micron'},\
        "Fe2.a4d7-a6d9":{'val':[1.257],'unit':'micron'},"Fe2.a4d7-a6d7":{'val':[1.321],'unit':'micron'},"Fe2.a4d7-a4d9":{'val':[1.644],'unit':'micron'},\
        "FeH":{'val':[0.9895],'unit':'micron'},\
        #"H2.0-0.S(0)":{'val':[28.221],'unit':'micron'}, "H2.0-0.S(1)":{'val':[17.035],'unit':'micron'}, "H2.0-0.S(2)":{'val':[12.279],'unit':'micron'},\
        #"H2.0-0.S(3)":{'val':[9.6649],'unit':'micron'}, "H2.0-0.S(4)":{'val':[8.0258],'unit':'micron'}, "H2.0-0.S(5)":{'val':[6.9091],'unit':'micron'},\
        #"H2.0-0.S(6)":{'val':[6.1088],'unit':'micron'}, "H2.0-0.S(7)":{'val':[5.5115],'unit':'micron'}, "H2.0-0.S(8)":{'val':[5.0529],'unit':'micron'},\
        #"H2.0-0.S(9)":{'val':[4.6947],'unit':'micron'}, "H2.0-0.S(10)":{'val':[4.4096],'unit':'micron'}, "H2.0-0.S(11)":{'val':[4.1810],'unit':'micron'},\
        #"H2.0-0.S(12)":{'val':[3.9947],'unit':'micron'}, "H2.0-0.S(13)":{'val':[3.8464],'unit':'micron'}, "H2.0-0.S(14)":{'val':[3.724],'unit':'micron'},\
        #"H2.0-0.S(15)":{'val':[3.625],'unit':'micron'}, "H2.0-0.S(16)":{'val':[3.547],'unit':'micron'}, "H2.0-0.S(17)":{'val':[3.485],'unit':'micron'},\
        #"H2.0-0.S(18)":{'val':[3.438],'unit':'micron'}, "H2.0-0.S(19)":{'val':[3.404],'unit':'micron'}, "H2.0-0.S(20)":{'val':[3.380],'unit':'micron'},\
        #"H2.0-0.S(21)":{'val':[3.369],'unit':'micron'}, "H2.0-0.S(22)":{'val':[3.366],'unit':'micron'}, "H2.0-0.S(23)":{'val':[3.372],'unit':'micron'},\
        "H2.1-0.S(0)":{'val':[2.2235],'unit':'micron'}, "H2.1-0.S(1)":{'val':[2.1218],'unit':'micron'}, "H2.1-0.S(2)":{'val':[2.0338],'unit':'micron'},\
        #"H2.1-0.S(3)":{'val':[1.9576],'unit':'micron'}, "H2.1-0.S(4)":{'val':[1.8920],'unit':'micron'}, "H2.1-0.S(5)":{'val':[1.8358],'unit':'micron'},\
        #"H2.1-0.S(6)":{'val':[1.7880],'unit':'micron'}, "H2.1-0.S(7)":{'val':[1.7480],'unit':'micron'}, "H2.1-0.S(8)":{'val':[1.7147],'unit':'micron'},\
        #"H2.1-0.S(10)":{'val':[1.6665],'unit':'micron'}, "H2.1-0.S(11)":{'val':[1.6504],'unit':'micron'},\
        "H2.1-0.Q(1)":{'val':[2.4066],'unit':'micron'}, "H2.1-0.Q(2)":{'val':[2.4134],'unit':'micron'}, "H2.1-0.Q(3)":{'val':[2.4237],'unit':'micron'},\
        "H2.1-0.Q(4)":{'val':[2.4375],'unit':'micron'}, "H2.1-0.Q(5)":{'val':[2.4548],'unit':'micron'}, "H2.1-0.Q(6)":{'val':[2.4756],'unit':'micron'},\
        #"H2.1-0.Q(7)":{'val':[2.5001],'unit':'micron'}, "H2.1-0.O(2)":{'val':[2.6269],'unit':'micron'}, "H2.1-0.O(3)":{'val':[2.8025],'unit':'micron'},\
        #"H2.1-0.O(4)":{'val':[3.0039],'unit':'micron'}, "H2.1-0.O(5)":{'val':[3.2350],'unit':'micron'}, "H2.1-0.O(6)":{'val':[3.5007],'unit':'micron'},\
        #"H2.1-0.O(7)":{'val':[3.8075],'unit':'micron'}, "H2.1-0.O(8)":{'val':[4.1625],'unit':'micron'}, "H2.2-1.S(0)":{'val':[2.3556],'unit':'micron'},\
        #"H2.2-1.S(1)":{'val':[2.2477],'unit':'micron'}, "H2.2-1.S(2)":{'val':[2.1542],'unit':'micron'}, "H2.2-1.S(3)":{'val':[2.0735],'unit':'micron'},\
        #"H2.2-1.S(4)":{'val':[2.0041],'unit':'micron'}, "H2.2-1.S(5)":{'val':[1.9449],'unit':'micron'}, "H2.2-1.O(2)":{'val':[2.7862],'unit':'micron'},\
        #"H2.2-1.O(3)":{'val':[2.9741],'unit':'micron'}, "H2.2-1.O(4)":{'val':[3.1899],'unit':'micron'}, "H2.2-1.O(5)":{'val':[3.4379],'unit':'micron'},\
        #"H2.2-1.O(6)":{'val':[3.7236],'unit':'micron'}, "H2.2-1.O(7)":{'val':[4.0540],'unit':'micron'}, "H2.3-2.S(0)":{'val':[2.5014],'unit':'micron'},\
        #"H2.3-2.S(1)":{'val':[2.3864],'unit':'micron'}, "H2.3-2.S(2)":{'val':[2.2870],'unit':'micron'},\
        "H2.3-2.S(4)":{'val':[2.1280],'unit':'micron'}, "H2.3-2.S(5)":{'val':[2.0656],'unit':'micron'}, "H2.3-2.S(6)":{'val':[2.0130],'unit':'micron'},\
        #"H2.3-2.S(7)":{'val':[1.9692],'unit':'micron'}, "H2.3-2.O(2)":{'val':[2.9620],'unit':'micron'}, "H2.3-2.O(3)":{'val':[3.1637],'unit':'micron'},\
        #"H2.3-2.O(4)":{'val':[3.3958],'unit':'micron'}, "H2.3-2.O(5)":{'val':[3.6630],'unit':'micron'}, "H2.3-2.O(6)":{'val':[3.9721],'unit':'micron'},\
        #"H2.2-0.S(1)":{'val':[1.1622],'unit':'micron'}, "H2.2-0.S(2)":{'val':[1.1382],'unit':'micron'},\
        #"H2.2-0.S(3)":{'val':[1.1175],'unit':'micron'}, "H2.2-0.S(4)":{'val':[1.0998],'unit':'micron'}, "H2.2-0.S(5)":{'val':[1.0851],'unit':'micron'},\
        #"H2.2-9.Q(1)":{'val':[1.2383],'unit':'micron'}, "H2.2-0.Q(2)":{'val':[1.2419],'unit':'micron'}, "H2.2-0.Q(3)":{'val':[1.2473],'unit':'micron'},\
        #"H2.2-0.Q(4)":{'val':[1.2545],'unit':'micron'}, "H2.2-0.Q(5)":{'val':[1.2636],'unit':'micron'}, "H2.2-0.O(2)":{'val':[1.2932],'unit':'micron'},\
        #"H2.2-0.O(3)":{'val':[1.3354],'unit':'micron'}, "H2.2-0.O(4)":{'val':[1.3817],'unit':'micron'}, "H2.2-0.O(5)":{'val':[1.4322],'unit':'micron'},\
        #"H2.4-3.S(3)":{'val':[2.3446],'unit':'micron'}, "H2.3-2.S(3)":{'val':[2.2014],'unit':'micron'},"H2.1-0.S(9)":{'val':[1.6877],'unit':'micron'},\
        "H12":{'val':[3751.22],'unit':'angstrom'},"H11":{'val':[3771.70],'unit':'angstrom'},"H10":{'val':[3798.98],'unit':'angstrom'},"H9":{'val':[3836.48],'unit':'angstrom'},\
        "H8":{'val':[3890.15],'unit':'angstrom'},"Hep":{'val':[3971.19],'unit':'angstrom'},"Hdel":{'val':[4102.92],'unit':'angstrom'},"Hgam":{'val':[4341.69],'unit':'angstrom'},\
        "Hbeta":{'val':[4862.69],'unit':'angstrom'},"Halpha":{'val':[6564.61],'unit':'angstrom'},\
        "He I":{'val':[0.388975,0.587730,0.6679996,1.0833],'unit':'micron'},\
        'Humphreys10-6':{'val':[5.12865,4.67251,4.17080,4.02087,3.90755,3.81945,3.74940,3.69264,3.64593,\
                                3.60697,3.57410,3.54610,3.52203,3.50116,3.48296],'unit':'microns'},\
        "K I":{'val':[1.1682,1.1765,1.2518,1.2425,1.5152],'unit':'micron'},\
        "O I":{'val':[1.129,0.630204,0.5578887],'unit':'micron'},\
        "O II":{'val':[3726,3727.09,3729,3729.88],'unit':'micron'},\
        "O III":{'val':[5008.24,4960.30,4364.44],'unit':'angstrom'},\
        "N II":{'val':[6549.84, 6585.23, 5756.24],'unit':'angstrom'},\
        "Mg":{'val':[5167.321,5172.684, 5183.604],'unit':'angstrom'},\
        "Mg I":{'val':[1.1820,1.4872,1.5020,1.5740,1.7095],'unit':'micron'},\
        "Na I":{'val':[0.589158,0.589755,0.818,1.1370,2.2040,2.206,2.208],'unit':'micron'},\
        "Ne II":{'val':[6585.23,6549.84,5756.24],'unit':'angstrom'},\
        "Ne III":{'val':[0.386981,0.396853],'unit':'micron'},\
        "S II":{'val':[6718.32,6732.71],'unit':'angstrom'},\
        "S III":{'val':[6313.8],'unit':'angstrom'},\
        "S III":{'val':[9071.1,9533.2],'unit':'angstrom'},\
        "Si I":{'val':[1.5875],'unit':'micron'},\
        "P11":{'val':[8865.217],'unit':'angstrom'},\
        "P10":{'val':[9017.385],'unit':'angstrom'},\
        "P9":{'val':[9231.547],'unit':'angstrom'},\
        "P8":{'val':[9548.590],'unit':'angstrom'},\
        "P7":{'val':[10052.128],'unit':'angstrom'},\
        "paa":{'val':[1.875613],'unit':'micron'},"pab":{'val':[1.28216],'unit':'micron'},"pag":{'val':[1.0941091],'unit':'micron'},\
        'Pa7-3':{'val':[1.00521],'unit':'microns'},'Pa8-3':{'val':[0.95486],'unit':'microns'},'Pa9-3':{'val':[0.92315],'unit':'microns'},\
        'Pa10-3':{'val':[0.90174],'unit':'microns'},\
        'Pfund':{'val':[4.65378,3.74056,3.29699,3.03920,2.87300,2.87300,2.75827,2.67513,2.61265,2.56433,\
                        2.52609,2.49525,2.46999,2.44900,2.43136,2.41639,2.40355,2.39248,2.38282,2.37438,\
                        2.36694,2.36035,2.35448,2.34924,2.34453,2.34028,2.33644,2.33296,2.32979],'unit':'microns'}\
    },\
    'radio':{\
    }
}

class lines(object):
    '''
    supported units all to anstroms and hz
    to add new units have to correct self.units and resolve_units
    '''
    def __init__(self):
        '''
        Setup the class with loading copy
        '''
        self.c     = 2.99792458e18       # speed of light AGS
        self.units = {\
                      'bananas'    : 2.032*10**9,\
                      'angstroms'  : 1.,\
                      'micrometers': 10**4,\
                      'millimeters': 10**7,\
                      'centimeters': 10**8,\
                      'meters'     : 10**10,\
                      'kilometers' : 10**13,\
                      'hz'         : 1.,\
                      'khz'        : 10**3,\
                      'mhz'        : 10**6,\
                      'ghz'        : 10**9,\
                      'thz'        : 10**12,\
                      }
        self.types = ['nir']

    def __call__(self,wtype='nir',bu='meters',x1=-1,x2=-1):
        '''
        Allows repeat calls and inline calling of function.
        Main process to gather all files
        This returns object of itself. Use return functions to get needed items
        '''
        try:
            if (wtype == self.type) and ((bu == self.bu[1]) or (bu == self.bu[2])) and (self.fulllines):
                regen  = False
                failed = False
                pass
            else:
                failed = True
                pass
        except:
            failed = True

        if failed:
            self.alllines  = deepcopy(atomiclines)
            self.bu   = self.resolve_units(bu.lower())
            self.type = self.resolve_type(wtype.lower())
            self.find_lines()
            regen  = True
            failed = False

        try:
            if ((x1 == self.x1) and (x2 == self.x2) and (self.region)) and not regen:
                regen = False
                pass    
            else:
                regen = True
                pass       
        except:
            regen = True

        if regen:
            self.find_regions(x1,x2)
            self.x1,self.x2 = x1,x2
            regen = False

        return self

    def return_lines(self):
        '''
        Returns all lines
        '''
        return self.fulllines

    def return_regions(self):
        '''
        Returns all lines within region <= all lines
        '''
        return self.region

    def get_params(self):
        '''
        Return parameters
        '''
        return dir(self)

    def get_args(self):
        '''
        Return arguments given
        '''
        return vars(self)

    def get_types(self):
        '''
        Returns the types of line regions that have been defined
        '''
        return [x for x in self.types]

    def get_units(self):
        '''
        Returns the units possible in the current setup
        '''
        return [x for x in self.units]

    def clear(self):
        '''
        Clears major memory hogs for reset
        '''
        self.alllines   = None
        self.fulllines  = None
        self.region     = None
        self.type       = None
        self.bu         = None
        self.x1,self.x2 = None,None

    def resolve_units(self,bu):
        '''
        Resolves the units and conversion factor
        '''
        tmp = self.resolve_name(bu)
        if tmp[0]:
            return tmp
        else:
            self.exit('Unit: <{}> was not found in list of units: {}'.format(bu,self.get_units()))

    def resolve_name(self,bu):
        '''
        Will resolve the name of the 
        unit from known types
        '''
        resolve_units = {\
                         'bananas'    : {'vals':['b','banana'],'type':'wave'},\
                         'angstroms'  : {'vals':['ang','a','angs','angstrom'],'type':'wave'},\
                         'micrometers': {'vals':['microns','micron','mu','micrometres','micrometre','micrometer'],'type':'wave'},\
                         'millimeters': {'vals':['mm','milli','millimetres','millimetre','millimeter'],'type':'wave'},\
                         'centimeters': {'vals':['cm','centi','centimetres','centimetre','centimeter'],'type':'wave'},\
                         'meters'     : {'vals':['m','metres','meter','metre'],'type':'wave'},\
                         'kilometers' : {'vals':['km','kilo','kilometres','kilometre','kilometer'],'type':'wave'},\
                         'hz'         : {'vals':['hertz','h'],'type':'freq'},\
                         'khz'        : {'vals':['kilohertz','kilo-hertz','kh'],'type':'freq'},\
                         'mhz'        : {'vals':['megahertz','mega-hertz','mh'],'type':'freq'},\
                         'ghz'        : {'vals':['gigahertz','giga-hertz','gh'],'type':'freq'},\
                         'thz'        : {'vals':['terahertz','tera-hertz','th'],'type':'freq'},\
                         }

        if bu not in self.get_units():
            for i in resolve_units:
                for k in resolve_units[i]['vals']:
                    if bu == k:
                        return True,bu,i,resolve_units[i]['type']
            return False,bu
        else:
            return True,bu,bu,resolve_units[bu]['type']

    def resolve_type(self,typel):    
        '''
        Resolves Type
        '''
        if typel not in self.get_types():
            self.exit('Type: <{}> was not found in list of types: {}'.format(typel,self.get_types()))
        else:
            return typel

    def conversion(self,init,ctype,fin,ftype):
        '''
        returns conversion factor needed
        '''
        if ctype == ftype:
            return self.units[init]/self.units[fin]
        elif ctype == 'freq':
            return self.units['angstroms']/self.units[fin] * self.c * self.units[init]/self.units['hz']
        elif ctype == 'wave':
            return self.units['hz']/self.units[fin] * self.c * self.units[init]/self.units['angstroms']


    def find_lines(self):    
        '''
        Returns the dictionary of all converted types
        '''
        self.fulllines = self.alllines[self.type]
        for i in self.alllines[self.type]:
            initialun   = self.resolve_name(self.alllines[self.type][i]['unit'])[2:4]
            finalun     = self.bu[2:4]
            initialconv = self.conversion(*initialun,*finalun)
            temp = []
            for k,j in enumerate(self.alllines[self.type][i]['val']):
                if ('hz' not in finalun[0]) or ('hz' not in initialun[0]):
                    temp.append(j * initialconv)
                else:
                    temp.append(initialconv/j)
            self.fulllines[i] = temp

    def find_regions(self,x1,x2):
        '''
        returns line names within a region
        if you modify the line type, you will want to regen the region
        '''
        self.region = {}
        for key in self.fulllines:
            a = np.array(self.fulllines[key])
            if (x1 != -1) and (x2 != -1):
                ind = np.where(np.logical_and(a>=x1,a<=x2))
            elif (x1 == -1) and (x2 == -1):
                ind = (np.arange(0,a.shape[0],1),)
            elif (x1 == -1):
                ind = np.where(a<=x2)
            elif (x2 == -1):
                ind = np.where(a>=x1)
            ind = np.array([x for y in ind for x in y])
            if ind.shape[0] != 0:
                self.region[key] = a[ind].tolist()

    def find_aperture(self,data,ap=1):
        '''
        returns a new key value pair dictionary
        where the new data is suppressed 
        within some ap so less overlap
        '''
        return True
        tmp = {}
        # reverses data dictionary keeps unique vals
        for key,vals in data.items():
            for x in vals:
                if tmp[str(x)]:
                    tmp[str(x)] = '{},{}'.format(tmp[str(x)],key)
                else:
                    tmp[str(x)] = key
        # reduce tmp to its final form
        ite = 0
        keys = tmp.keys()
        key0 = float(keys[ite])
        nkey = []
        while ite < (len(tmp.items()) - 1):
            key1 = float(keys[ite+1])
            if (key1-key0) > ap:
                nkey.append(key0)
                key0 = key1
            if ite == (len(tmp.items()) - 2):
                nkey.append(key0)
                key0 = key1
            # need to work on edge case
            ite += 1


    def exit(self,exitcode,exitparam=0):
        '''
        Handles error codes and exits nicely
        '''
        print(exitcode)
        print('v--------Ignore exit codes below--------v')
        self.clear()
        if exitparam == 0:
            return None
        else:
            from sys import exit
            exit(0)
