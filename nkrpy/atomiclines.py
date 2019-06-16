#!/usr/bin/env python
"""
Name  : Atomic Lines, atomiclines.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : Houses all useful atomic lines and short program for parsing
        Suggest using the "call" wrapper for first calling the lines class
        After initial call, then use the functions defined within the class

"""
__filename__ = __file__.split('/')[-1].strip('.py')

# import standard modules
from copy import deepcopy
import numpy as np
from .miscmath import binning

def call(*args,**kwargs):
    """
    Better wrapper for quickly calling 
    the lines class
    Can specify the arguments within a single line without
    having to self initialize and recall
    returns the lines class pre initialized
    """
    final = None
    try: # if key-word arguments are supplied
        final = lines()
        final(**kwargs)
    except: # if arguments are supplied (less specified)
        final = lines()
        final(*args)
    # if nothing is specified or if if the above two break
    if final == None:
        final = lines()
        final()
    return final

class lines(object):
    """
    supported units all to anstroms and hz
    to add new units have to correct self.units and resolve_units
    """

    def __init__(self):
        """
        Setup the class with loading copy

        units{} defines all units that can be used
        Dictionary is as follows:
        key  = master name
        vals = possible aliases to resolve
        type = specifies either wavelength or frequency
        fac  = the conversion factor to get to Angstrom(Hertz) for wavelength(frequency)
        """
        self.c     = 2.99792458e18       # speed of light AGS

        self.units = {\
                         'bananas'    : {'vals':['b','banana'],'type':'wave','fac':2.032*10**9},\
                         'angstroms'  : {'vals':['ang','a','angs','angstrom'],'type':'wave','fac':1.},\
                         'micrometers': {'vals':['microns','micron','mu','micrometres','micrometre','micrometer'],'type':'wave','fac':10**4},\
                         'millimeters': {'vals':['mm','milli','millimetres','millimetre','millimeter'],'type':'wave','fac':10**7},\
                         'centimeters': {'vals':['cm','centi','centimetres','centimetre','centimeter'],'type':'wave','fac':10**8},\
                         'meters'     : {'vals':['m','metres','meter','metre'],'type':'wave','fac':10**10},\
                         'kilometers' : {'vals':['km','kilo','kilometres','kilometre','kilometer'],'type':'wave','fac':10**13},\
                         'hz'         : {'vals':['hertz','h'],'type':'freq','fac':1.},\
                         'khz'        : {'vals':['kilohertz','kilo-hertz','kh'],'type':'freq','fac':10**3},\
                         'mhz'        : {'vals':['megahertz','mega-hertz','mh'],'type':'freq','fac':10**6},\
                         'ghz'        : {'vals':['gigahertz','giga-hertz','gh'],'type':'freq','fac':10**9},\
                         'thz'        : {'vals':['terahertz','tera-hertz','th'],'type':'freq','fac':10**12},\
                         }

        self.types = atomiclines.keys()

    def __call__(self,wtype='nir',bu='meters',x1=-1,x2=-1):
        """
        Allows repeat calls and inline calling of function.
        Main process to gather all files
        This returns object of itself. Use return functions to get needed items
        """
        assert wtype in self.types
        try:
            if (wtype == self.type) and ((bu == self.bu[1]) or (bu == self.bu[2])) and (self.fulllines):
                # if current  wavelength region is same, current conversion is same/already done, and 
                #     linelist has been created 
                regen  = False # dont regenerate regions
                failed = False # dont regenerate full set
                pass
            else:
                failed = True # regenerate full set
                pass
        except:
            failed = True

        if failed:
            self.bu   = self.resolve_units(bu.lower())
            self.type = self.resolve_type(wtype.lower())
            self.alllines  = deepcopy(atomiclines[self.type])
            self.find_lines()
            regen  = True # regenerate regions
            failed = False 

        try:
            if ((x1 == self.x1) and (x2 == self.x2) and (self.region)) and not regen:
                # checking if regions have already been set but flag misfired
                regen = False
                pass    
            else:
                regen = True
                pass       
        except:
            regen = True

        if regen: # regenerate regions
            self.find_regions(x1,x2)
            self.x1,self.x2 = x1,x2
            regen = False
        return self

    def return_lines(self):
        """
        Returns all lines
        """
        return self.fulllines

    def return_regions(self):
        """
        Returns all lines within region <= all lines
        """
        return self.region

    def get_functions(self):
        """
        Return parameters
        """
        return dir(self)

    def get_args(self):
        """
        Return arguments given
        """
        return vars(self)

    def get_types(self):
        """
        Returns the types of line regions that have been defined
        """
        return self.types

    def get_units(self):
        """
        Returns the units possible in the current setup
        """
        return self.units.keys()

    def clear(self):
        """
        Clears major memory hogs for reset
        """
        self.alllines   = None
        self.fulllines  = None
        self.region     = None
        self.type       = None
        self.bu         = None
        self.x1,self.x2 = None,None

    def resolve_units(self,bu):
        """
        Resolves the units and conversion factor
        """
        tmp = self.resolve_name(bu)
        if tmp[0]:
            return tmp
        else:
            self.exit('Unit: <{}> was not found in list of units: {}'.format(bu,self.get_units()))

    def resolve_name(self,bu):
        """
        Will resolve the name of the 
        unit from known types
        """

        if bu not in self.get_units():
            for i in self.units:
                for k in self.units[i]['vals']:
                    if bu == k:
                        return True,bu,i,self.units[i]['type']
            return False,bu
        else:
            return True,bu,bu,self.units[bu]['type']

    def resolve_type(self,typel):    
        """
        Resolves Type
        """
        if typel not in self.get_types():
            self.exit('Type: <{}> was not found in list of types: {}'.format(typel,self.get_types()))
        else:
            return typel

    def conversion(self,init,ctype,fin,ftype):
        """
        returns conversion factor needed
        """
        if ctype == ftype: # converting between common types (wavelength->wavelength)
            return self.units[init]['fac']/self.units[fin]['fac']
        elif ctype == 'freq': # converting from freq to wavelength
            return self.units['angstroms']['fac']/self.units[fin]['fac'] * self.c * self.units[init]['fac']/self.units['hz']['fac']
        elif ctype == 'wave': # converting from wavelength to freq
            return self.units['hz']['fac']/self.units[fin]['fac'] * self.c * self.units[init]['fac']/self.units['angstroms']['fac']


    def find_lines(self):    
        """
        Returns the dictionary of all converted types
        """
        self.fulllines = self.alllines
        # iterating through all lines in a type
        for i in self.alllines:
            # Type of line from atomicline list     unit,type
            initialun   = self.resolve_name(self.alllines[i]['unit'])[2:4]
            # Type of output desired   unit,type
            finalun     = self.bu[2:4]
            # general conversion between the two above
            initialconv = self.conversion(*initialun,*finalun)
            temp = []
            # iterating through values of the given line to create final list
            for k,j in enumerate(self.alllines[i]['val']): 
                if ('hz' not in finalun[0]) or ('hz' not in initialun[0]):
                    temp.append(j * initialconv) # converting in wavelength
                else:
                    temp.append(initialconv/j) # converting in frequency
            self.fulllines[i] = temp

    def find_regions(self,x1,x2):
        """
        returns line names within a region
        if you modify the line type, you will want to regen the region
        """
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

    def aperture(self):
        """
        returns a new key value pair dictionary
        where the new data is suppressed 
        psuedo kmeans cluster
        First we calculate an average difference between sequential elements 
        and then group together elements whose difference is less than average.
        """
        tmp = {}
        for linenam in self.fulllines:
            d = sorted(self.fulllines[linenam])
            if len(d)>1:
                diff = [d[i+1]-d[i] for i in range(len(d)-1)]#[y - x for x, y in zip(*[iter(d)] * 2)]
                avg = sum(diff) / len(diff)

                m = [[d[0]]]

                for x in d[1:]:
                    if x - m[-1][-1] < avg:#x - m[-1][0] < avg:
                        m[-1].append(x)
                    else:
                        m.append([x])
            else:
                m = d
            tmp[linenam] = [np.mean(x) for x in m]  
        self.fulllines = tmp
        return self


    def exit(self,exitcode,exitparam=0):
        """
        Handles error codes and exits nicely
        """
        print(exitcode)
        print('v--------Ignore exit codes below--------v')
        self.clear()
        if exitparam == 0:
            return None
        else:
            from sys import exit
            exit(0)

# v-----------------------------------------------------------v
#  These are a list of lines that can be used with above code
# ^-----------------------------------------------------------^

atomiclines={\
    'nir':{\
        "Al I":{'val':[1.3115,1.67],'unit':'micrometers'},\
        "Ar III":{'val':[7137.8,7753.2],'unit':'angstroms'},\
        r'Br $\gamma$':{'val':[2.16611],'unit':'micrometers'},'Br5-4':{'val':[4.05226],'unit':'micrometers'},'Br6-4':{'val':[2.62587],'unit':'micrometers'},\
        'Br8-4':{'val':[1.94509],'unit':'micrometers'},'Br9-4':{'val':[1.81791],'unit':'micrometers'},'Br10-4':{'val':[1.73669],'unit':'micrometers'},\
        #'Br11-4':{'val':[1.68111],'unit':'micrometers'},'Br12-4':{'val':[1.64117],'unit':'micrometers'},'Br13-4':{'val':[1.61137],'unit':'micrometers'},\
        #'Br14-4':{'val':[1.58849],'unit':'micrometers'},'Br15-4':{'val':[1.57050],'unit':'micrometers'},'Br16-4':{'val':[1.55607],'unit':'micrometers'},\
        'Br17-4':{'val':[1.54431],'unit':'micrometers'},'Br18-4':{'val':[1.53460],'unit':'micrometers'},'Br19-4':{'val':[1.52647],'unit':'micrometers'},\
        #'Br20-4':{'val':[1.51960],'unit':'micrometers'},'Br21-4':{'val':[1.51374],'unit':'micrometers'},\
        "Ca I":{'val':[1.9442,1.9755,2.2605,2.263,2.266],'unit':'micrometers'},\
        #"Ca II":{'val':[3933.663,3968.468,8500.36, 8544.44,8664.52],'unit':'angstroms'},\
        "CO":{'val':[2.2925,2.3440,2.414,2.322,2.352,2.383],'unit':'micrometers'},\
        #"Fe I":{'val':[1.1880,1.1970],'unit':'micrometers'},\
        "Fe II":{'val':[1.688,1.742],'unit':'micrometers'},\
        "Fe2.a4d7-a6d9":{'val':[1.257],'unit':'micrometers'},"Fe2.a4d7-a6d7":{'val':[1.321],'unit':'micrometers'},"Fe2.a4d7-a4d9":{'val':[1.644],'unit':'micrometers'},\
        "FeH":{'val':[0.9895],'unit':'micrometers'},\
        #"H2.0-0.S(0)":{'val':[28.221],'unit':'micrometers'}, "H2.0-0.S(1)":{'val':[17.035],'unit':'micrometers'}, "H2.0-0.S(2)":{'val':[12.279],'unit':'micrometers'},\
        #"H2.0-0.S(3)":{'val':[9.6649],'unit':'micrometers'}, "H2.0-0.S(4)":{'val':[8.0258],'unit':'micrometers'}, "H2.0-0.S(5)":{'val':[6.9091],'unit':'micrometers'},\
        #"H2.0-0.S(6)":{'val':[6.1088],'unit':'micrometers'}, "H2.0-0.S(7)":{'val':[5.5115],'unit':'micrometers'}, "H2.0-0.S(8)":{'val':[5.0529],'unit':'micrometers'},\
        #"H2.0-0.S(9)":{'val':[4.6947],'unit':'micrometers'}, "H2.0-0.S(10)":{'val':[4.4096],'unit':'micrometers'}, "H2.0-0.S(11)":{'val':[4.1810],'unit':'micrometers'},\
        #"H2.0-0.S(12)":{'val':[3.9947],'unit':'micrometers'}, "H2.0-0.S(13)":{'val':[3.8464],'unit':'micrometers'}, "H2.0-0.S(14)":{'val':[3.724],'unit':'micrometers'},\
        #"H2.0-0.S(15)":{'val':[3.625],'unit':'micrometers'}, "H2.0-0.S(16)":{'val':[3.547],'unit':'micrometers'}, "H2.0-0.S(17)":{'val':[3.485],'unit':'micrometers'},\
        #"H2.0-0.S(18)":{'val':[3.438],'unit':'micrometers'}, "H2.0-0.S(19)":{'val':[3.404],'unit':'micrometers'}, "H2.0-0.S(20)":{'val':[3.380],'unit':'micrometers'},\
        #"H2.0-0.S(21)":{'val':[3.369],'unit':'micrometers'}, "H2.0-0.S(22)":{'val':[3.366],'unit':'micrometers'}, "H2.0-0.S(23)":{'val':[3.372],'unit':'micrometers'},\
        "H2.1-0.S(0)":{'val':[2.2235],'unit':'micrometers'}, "H2.1-0.S(1)":{'val':[2.1218],'unit':'micrometers'}, "H2.1-0.S(2)":{'val':[2.0338],'unit':'micrometers'},\
        #"H2.1-0.S(3)":{'val':[1.9576],'unit':'micrometers'}, "H2.1-0.S(4)":{'val':[1.8920],'unit':'micrometers'}, "H2.1-0.S(5)":{'val':[1.8358],'unit':'micrometers'},\
        #"H2.1-0.S(6)":{'val':[1.7880],'unit':'micrometers'}, "H2.1-0.S(7)":{'val':[1.7480],'unit':'micrometers'}, "H2.1-0.S(8)":{'val':[1.7147],'unit':'micrometers'},\
        #"H2.1-0.S(10)":{'val':[1.6665],'unit':'micrometers'}, "H2.1-0.S(11)":{'val':[1.6504],'unit':'micrometers'},\
        "H2.1-0.Q(1)":{'val':[2.4066],'unit':'micrometers'}, "H2.1-0.Q(2)":{'val':[2.4134],'unit':'micrometers'}, "H2.1-0.Q(3)":{'val':[2.4237],'unit':'micrometers'},\
        "H2.1-0.Q(4)":{'val':[2.4375],'unit':'micrometers'}, "H2.1-0.Q(5)":{'val':[2.4548],'unit':'micrometers'}, "H2.1-0.Q(6)":{'val':[2.4756],'unit':'micrometers'},\
        #"H2.1-0.Q(7)":{'val':[2.5001],'unit':'micrometers'}, "H2.1-0.O(2)":{'val':[2.6269],'unit':'micrometers'}, "H2.1-0.O(3)":{'val':[2.8025],'unit':'micrometers'},\
        #"H2.1-0.O(4)":{'val':[3.0039],'unit':'micrometers'}, "H2.1-0.O(5)":{'val':[3.2350],'unit':'micrometers'}, "H2.1-0.O(6)":{'val':[3.5007],'unit':'micrometers'},\
        #"H2.1-0.O(7)":{'val':[3.8075],'unit':'micrometers'}, "H2.1-0.O(8)":{'val':[4.1625],'unit':'micrometers'}, "H2.2-1.S(0)":{'val':[2.3556],'unit':'micrometers'},\
        #"H2.2-1.S(1)":{'val':[2.2477],'unit':'micrometers'}, "H2.2-1.S(2)":{'val':[2.1542],'unit':'micrometers'}, "H2.2-1.S(3)":{'val':[2.0735],'unit':'micrometers'},\
        #"H2.2-1.S(4)":{'val':[2.0041],'unit':'micrometers'}, "H2.2-1.S(5)":{'val':[1.9449],'unit':'micrometers'}, "H2.2-1.O(2)":{'val':[2.7862],'unit':'micrometers'},\
        #"H2.2-1.O(3)":{'val':[2.9741],'unit':'micrometers'}, "H2.2-1.O(4)":{'val':[3.1899],'unit':'micrometers'}, "H2.2-1.O(5)":{'val':[3.4379],'unit':'micrometers'},\
        #"H2.2-1.O(6)":{'val':[3.7236],'unit':'micrometers'}, "H2.2-1.O(7)":{'val':[4.0540],'unit':'micrometers'}, "H2.3-2.S(0)":{'val':[2.5014],'unit':'micrometers'},\
        #"H2.3-2.S(1)":{'val':[2.3864],'unit':'micrometers'}, "H2.3-2.S(2)":{'val':[2.2870],'unit':'micrometers'},\
        "H2.3-2.S(4)":{'val':[2.1280],'unit':'micrometers'}, "H2.3-2.S(5)":{'val':[2.0656],'unit':'micrometers'}, "H2.3-2.S(6)":{'val':[2.0130],'unit':'micrometers'},\
        #"H2.3-2.S(7)":{'val':[1.9692],'unit':'micrometers'}, "H2.3-2.O(2)":{'val':[2.9620],'unit':'micrometers'}, "H2.3-2.O(3)":{'val':[3.1637],'unit':'micrometers'},\
        #"H2.3-2.O(4)":{'val':[3.3958],'unit':'micrometers'}, "H2.3-2.O(5)":{'val':[3.6630],'unit':'micrometers'}, "H2.3-2.O(6)":{'val':[3.9721],'unit':'micrometers'},\
        #"H2.2-0.S(1)":{'val':[1.1622],'unit':'micrometers'}, "H2.2-0.S(2)":{'val':[1.1382],'unit':'micrometers'},\
        #"H2.2-0.S(3)":{'val':[1.1175],'unit':'micrometers'}, "H2.2-0.S(4)":{'val':[1.0998],'unit':'micrometers'}, "H2.2-0.S(5)":{'val':[1.0851],'unit':'micrometers'},\
        #"H2.2-9.Q(1)":{'val':[1.2383],'unit':'micrometers'}, "H2.2-0.Q(2)":{'val':[1.2419],'unit':'micrometers'}, "H2.2-0.Q(3)":{'val':[1.2473],'unit':'micrometers'},\
        "H2.2-0.Q(4)":{'val':[1.2545],'unit':'micrometers'}, "H2.2-0.Q(5)":{'val':[1.2636],'unit':'micrometers'}, "H2.2-0.O(2)":{'val':[1.2932],'unit':'micrometers'},\
        #"H2.2-0.O(3)":{'val':[1.3354],'unit':'micrometers'}, "H2.2-0.O(4)":{'val':[1.3817],'unit':'micrometers'}, "H2.2-0.O(5)":{'val':[1.4322],'unit':'micrometers'},\
        #"H2.4-3.S(3)":{'val':[2.3446],'unit':'micrometers'}, "H2.3-2.S(3)":{'val':[2.2014],'unit':'micrometers'},"H2.1-0.S(9)":{'val':[1.6877],'unit':'micrometers'},\
        "H12":{'val':[3751.22],'unit':'angstroms'},"H11":{'val':[3771.70],'unit':'angstroms'},"H10":{'val':[3798.98],'unit':'angstroms'},"H9":{'val':[3836.48],'unit':'angstroms'},\
        "H8":{'val':[3890.15],'unit':'angstroms'},"Hep":{'val':[3971.19],'unit':'angstroms'},"Hdel":{'val':[4102.92],'unit':'angstroms'},"Hgam":{'val':[4341.69],'unit':'angstroms'},\
        r'H $\beta$':{'val':[4862.69],'unit':'angstroms'},r'H $\alpha$':{'val':[6564.61],'unit':'angstroms'},\
        "He I":{'val':[0.388975,0.587730,0.6679996,1.0833],'unit':'micrometers'},\
        'Humphreys10-6':{'val':[5.12865,4.67251,4.17080,4.02087,3.90755,3.81945,3.74940,3.69264,3.64593,\
                                3.60697,3.57410,3.54610,3.52203,3.50116,3.48296],'unit':'micrometers'},\
        "K I":{'val':[1.1682,1.1765,1.2518,1.2425,1.5152],'unit':'micrometers'},\
        "O I":{'val':[1.129,0.630204,0.5578887],'unit':'micrometers'},\
        "O II":{'val':[3726,3727.09,3729,3729.88],'unit':'micrometers'},\
        "O III":{'val':[5008.24,4960.30,4364.44],'unit':'angstroms'},\
        "N II":{'val':[6549.84, 6585.23, 5756.24],'unit':'angstroms'},\
        "Mg":{'val':[5167.321,5172.684, 5183.604],'unit':'angstroms'},\
        "Mg I":{'val':[1.1820,1.4872,1.5020,1.5740,1.7095],'unit':'micrometers'},\
        "Na I":{'val':[0.589158,0.589755,0.818,1.1370,2.2040,2.206,2.208],'unit':'micrometers'},\
        "Ne II":{'val':[6585.23,6549.84,5756.24],'unit':'angstroms'},\
        "Ne III":{'val':[0.386981,0.396853],'unit':'micrometers'},\
        "S II":{'val':[6718.32,6732.71],'unit':'angstroms'},\
        "S III":{'val':[6313.8],'unit':'angstroms'},\
        "S III":{'val':[9071.1,9533.2],'unit':'angstroms'},\
        "Si I":{'val':[1.5875],'unit':'micrometers'},\
        "P11":{'val':[8865.217],'unit':'angstroms'},\
        "P10":{'val':[9017.385],'unit':'angstroms'},\
        "P9":{'val':[9231.547],'unit':'angstroms'},\
        "P8":{'val':[9548.590],'unit':'angstroms'},\
        "P7":{'val':[10052.128],'unit':'angstroms'},\
        r'Pa $\alpha$':{'val':[1.875613],'unit':'micrometers'},r'Pa $\beta$':{'val':[1.28216],'unit':'micrometers'},r'Pa $\gamma$':{'val':[1.0941091],'unit':'micrometers'},\
        'Pa7-3':{'val':[1.00521],'unit':'micrometers'},'Pa8-3':{'val':[0.95486],'unit':'micrometers'},'Pa9-3':{'val':[0.92315],'unit':'micrometers'},\
        'Pa10-3':{'val':[0.90174],'unit':'micrometers'},\
        'Pfund':{'val':[4.65378,3.74056,3.29699,3.03920,2.87300,2.87300,2.75827,2.67513,2.61265,2.56433,\
                        2.52609,2.49525,2.46999,2.44900,2.43136,2.41639,2.40355,2.39248,2.38282,2.37438,\
                        2.36694,2.36035,2.35448,2.34924,2.34453,2.34028,2.33644,2.33296,2.32979],'unit':'micrometers'}\
    },\
    'radio':{\
    },\
    'xray':{\
    },\
    'gamma':{\
    }
}
