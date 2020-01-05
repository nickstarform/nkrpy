"""Handle AtomicLine searching and configuration."""

# standard modules

# external modules
import numpy as np

# relative modules
from ..constants import c as nkrpy__c
from ..unit.unit import Unit, BaseUnit
from . import atomiclines

__doc__ = """
Houses all useful atomic lines and short program for parsing.

The overview of how this module operates on the backend,
 * User defines a band (NIR, etc), a unit (meters, Hz, etc),
    and possibly limits.
 * The program will try to resolve the band to the known bands
    and the unit to the known units.
 * Then it will populate a preliminary configuration dict
 * Then this dict is fed into the nkrpy.unit resolver and
    this is saved to a new dict
 * If regions were specified, then they are regenerated.
"""

__filename__ = __file__.split('/')[-1].strip('.py')

__all__ = ('Lines',)


class Lines(object):
    """."""

    c = nkrpy__c * 1E8  # setting to angstroms/s
    types = atomiclines.keys()
    atomiclines = BaseUnit(**atomiclines)
    baselines = {}
    fulllines = {}
    config = {'band': 'nir', 'unit': 'meters',
              'xlower': -1, 'xupper': -1,
              'resolved': None}

    def __init__(self, band: str = 'nir', unit: str = 'meters',
                 xlower: float = -1, xupper: float = -1):
        """Initilization Magic Method."""
        self.config['band'] = self.resolve_band(band)
        self.config['unit'] = Unit.resolve_unit(unit)
        self.config.update({'xlower': xlower, 'xupper': xupper})
        self.__format_atomiclines()
        self.__populate_lines()
        self.__generate_regions(xlower, xupper)

    def __reset__(self):
        """Reset the class attributes."""
        self.config = {'band': 'nir', 'unit': 'meters',
                       'xlower': -1, 'xupper': -1}
        self.__format_atomiclines()
        self.__populate_lines()

    def __call__(self, band: str, unit: str,
                 xlower: float = -1, xupper: float = -1):
        """Magic Method.

        Allows repeat calls and inline calling of function.
        Main process to gather all files
        This returns object of itself. Use return functions to get needed items
        """
        band = self.resolve_band(band)
        if not band:
            band = 'nir'
        self.config['band'] = band
        resolve_units = True
        unit = Unit.resolve_unit(unit)
        if not unit:
            unit = 'meters'
        if unit == self.config['unit']:
            resolve_units = False
        if resolve_units:
            self.config['unit'] = unit
            self.__format_atomiclines()
        regen = False
        if (xlower, xupper) != (self.config['xlower'], self.config['xupper']):
            self.config.update({'xlower': xlower, 'xupper': xupper})
            regen = True

        if regen:  # regenerate regions
            self.__generate_regions(xlower, xupper)
        self.__populate_lines()
        return self

    @property
    def return_types(self):
        """Return the types of line regions that have been defined."""
        return self.types

    @property
    def return_atomiclines(self):
        """Return all lines."""
        return self.atomiclines

    @property
    def return_baselines(self):
        """Return all lines."""
        return self.baselines

    @property
    def return_lines(self):
        """Return all lines."""
        return self.fulllines

    @property
    def return_regions(self):
        """Return lines within the region."""
        return self.region

    def __populate_lines(self):
        for linename, lineinfo in self.baselines:
            if linename in self.regions:
                vals = self.regions[linename]
                self.baselines[linename]['vals'] = vals
            unit = Unit(**self.baselines[linename])
            self.fulllines[linename] = {
                'resolved': unit.get_vals,
                'rank': lineinfo['rank'],
            }

    @staticmethod
    def __format_atomiclines(self):
        for linename, lineinfo in self.atomiclines.items():
            if linename in self.baselines:
                self.baselines[linename].update({'baseunit': lineinfo['unit'],
                                                 'covunit': self.config['unit'],  # noqa
                                                 'vals': lineinfo['val']})
                continue
            self.baselines[linename] = BaseUnit({'baseunit': lineinfo['unit'],
                                                 'covunit': self.config['unit'],  # noqa
                                                 'vals': lineinfo['val']})

    @classmethod
    def resolve_band(__cls__, band: str):
        """Resolve Band."""
        band = band.lower()
        if band not in __cls__.types:
            return 'nir'
        else:
            return band

    def __generate_regions(self, xlower: float, xupper: float):
        """Resolve regions.

        if you modify the line type, you will want to regen the region
        """
        self.region = {}
        for key in self.fulllines:
            if (xlower == -1) and (xupper == -1):
                continue
            a = np.array(self.fulllines[key]['resolved'], dtype=np.float)
            if (xlower != -1) and (xupper != -1):
                ind = np.where(np.logical_and(a >= xlower, a <= xupper))
            elif (xlower == -1):
                ind = np.where(a <= xupper)
            elif (xupper == -1):
                ind = np.where(a >= xlower)
            ind = ind.ravel()
            self.region[key] = a[ind]

    def aperture(self):
        """Apply a window.

        returns a new key value pair dictionary
        where the new data is suppressed
        psuedo kmeans cluster
        First we calculate an average difference
            between sequential elements
        and then group together elements whose
            difference is less than average.
        """
        tmp = {}
        for linenam in self.fulllines:
            d = sorted(self.fulllines[linenam])
            if len(d) > 1:
                diff = [d[i+1]-d[i] for i in range(len(d)-1)]
                # [y - x for x, y in zip(*[iter(d)] * 2)]
                avg = sum(diff) / len(diff)

                m = [[d[0]]]

                for x in d[1:]:
                    if x - m[-1][-1] < avg:
                        # x - m[-1][0] < avg:
                        m[-1].append(x)
                    else:
                        m.append([x])
            else:
                m = d
            tmp[linenam] = [np.mean(x) for x in m]
        self.fulllines = tmp

# end of code
