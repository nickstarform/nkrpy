"""Handle AtomicLine searching and configuration."""

# standard modules
from copy import deepcopy

# external modules
import numpy as np

# relative modules
from .miscmath import binning
from .constants import c
from .astro import Units
from .astro.linelist import atomiclines

__doc__ = "Houses all useful atomic lines and short program for parsing\
        Suggest using the 'call' wrapper for first calling the lines class\
        After initial call, then use the functions defined within the class"

__filename__ = __file__.split('/')[-1].strip('.py')

__all__ = ('Lines',)

# setting to angstroms/s
c = c * 1E8


class Lines(Units):
    """Support Unit Parent class and extend."""

    """
    Tries to handle all astronomical line conversions.
    """

    def __init__(self, *args, **kwargs):
        """Initilization Magic Method."""
        super().__init__(*args, **kwargs)
        self.types = atomiclines.keys()

    def __call__(self, wtype='nir', bu='meters', x1=-1, x2=-1):
        """Calling Magic Method."""
        """
        Allows repeat calls and inline calling of function.
        Main process to gather all files
        This returns object of itself. Use return functions to get needed items
        """
        assert wtype in self.types
        try:
            if (wtype == self.type) and \
               ((bu == self.bu[1]) or (bu == self.bu[2])) and \
               (self.fulllines):
                # if wavelength, conversion, linelist has been created
                regen = False  # dont regenerate regions
                failed = False  # dont regenerate full set
                pass
            else:
                failed = True  # regenerate full set
                pass
        except:
            failed = True

        # regen all lines and regions
        if failed:
            self.type = self.resolve_type(wtype)
            self.bu = self._resolve_units(bu)
            self.alllines = deepcopy(atomiclines[self.type])
            self.find_lines()
            regen = True  # regenerate regions
            failed = False

        try:
            if ((x1 == self.x1) and
                (x2 == self.x2) and
                (self.region)) and not regen:  # noqa
                regen = False
                pass
            else:
                regen = True
                pass
        except:
            regen = True

        if regen:  # regenerate regions
            self.find_regions(x1, x2)
            self.x1, self.x2 = x1, x2
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

    def get_types(self):
        """
        Returns the types of line regions that have been defined
        """
        return self.types

    def resolve_type(self, typel):    
        """
        Resolves Type
        """
        typel = typel.lower()
        if typel not in self.get_types():
            self.exit(f'Type: <{typel}> was not found in list' +\
                f' of types: {self.get_types()}')
        else:
            return typel

    def find_lines(self):    
        """
        Returns the dictionary of all converted types
        """
        self.fulllines = self.alllines
        # iterating through all lines in a type
        for i in self.alllines:
            # Type of line from atomicline list     unit,type
            initialun = self.resolve_name(self.alllines[i]['unit'])[2:4]
            # Type of output desired   unit,type
            finalun = self.bu[2:4]
            # general conversion between the two above
            initialconv = conversion(*initialun,*finalun)
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

# end of code
