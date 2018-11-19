"""Various Astronomy functions."""

# external modules
import numpy as np

# relative modules
from .constants import h, c, kb, jy, msun
from .functions import typecheck
from .dustmodels.kappa import *

c = c * 1E8 # A/s
h = h * 1E-7 # SI
kb = kb * 1E-7 #SI

__all__ = ('Units', 'planck', 'dustmass', 'planck_nu', 'planck_wav')


class Units(object):
    """Convert between major unit types."""

    __doc__ = """
    supported units all to anstroms and hz
    to add new units have to correct self.units and resolve_units
    To setup, just initialize and call with units /  values to convert
    Holds values to quick accessing later

    Use Units.converting for info on how you are converting
    """

    def __init__(self, unit=None, vals=None):
        """Setup the class with loading copy."""
        """
        units{} defines all units that can be used
        Dictionary is as follows:
        key = master name
        vals = possible aliases to resolve
        type = specifies either wavelength or frequency
        fac = the conversion factor to Ang(Hz) for wavelen(freq)
        """

        self.units = {
            'J': {'vals': ('j', 'joules', 'joule'),
                   'type': 'energy',
                   'fac': 1.},
            'eV': {'vals': ('ev', 'electronvolt', 'eVs', 'electronvolts'),
                   'type': 'energy',
                   'fac': 1.6021766208E-19},
            'keV': {'vals': ('kev', 'kiloelectronvolt','keVs','kiloelectronvolts', 'kiloeV'),
                   'type': 'energy',
                   'fac': 1.E3 * 1.6021766208E-19},
            'MeV': {'vals': ('mev', 'megaelectronvolt','mevs','megaelectronvolts'),
                   'type': 'energy',
                   'fac': 1.E6 * 1.6021766208E-19},
            'GeV': {'vals': ('gev', 'gigaaelectronvolt','gevs','gigaelectronvolts'),
                   'type': 'energy',
                   'fac': 1.E9 * 1.6021766208E-19},
            'bananas': {'vals': ('b', 'banana'),
                        'type': 'wave',
                        'fac': 2.032 * 10 ** 9},
            'angstroms': {'vals': ('ang', 'a', 'angs', 'angstrom'),
                          'type': 'wave',
                          'fac': 1.},
            'micrometers': {'vals': ('microns', 'micron', 'mu', 'micrometres',
                                     'micrometre', 'micrometer'),
                            'type': 'wave',
                            'fac': 10 ** 4},
            'millimeters': {'vals': ('mm', 'milli', 'millimetres',
                                     'millimetre', 'millimeter'),
                            'type': 'wave',
                            'fac': 10 ** 7},
            'centimeters': {'vals': ('cm', 'centi', 'centimetres',
                                     'centimetre', 'centimeter'),
                            'type': 'wave',
                            'fac': 10 ** 8},
            'meters': {'vals': ('m', 'metres', 'meter', 'metre'),
                       'type': 'wave',
                       'fac': 10 ** 10},
            'kilometers': {'vals': ('km', 'kilo', 'kilometres', 'kilometre',
                                    'kilometer'),
                           'type': 'wave',
                           'fac': 10 ** 13},
            'lightyear': {'vals': ('lyr', 'lightyears'),
                           'type': 'wave',
                           'fac': 9.461E25},
            'parcsec': {'vals': ('pc', 'parsecs'),
                           'type': 'wave',
                           'fac': 3.086E26},
            'hz': {'vals': ('hertz', 'h'),
                   'type': 'freq',
                   'fac': 1.},
            'khz': {'vals': ('kilohertz', 'kilo-hertz', 'kh'),
                    'type': 'freq',
                    'fac': 10 ** 3},
            'mhz': {'vals': ('megahertz', 'mega-hertz', 'mh'),
                    'type': 'freq',
                    'fac': 10 ** 6},
            'ghz': {'vals': ('gigahertz', 'giga-hertz', 'gh'),
                    'type': 'freq',
                    'fac': 10 ** 9},
            'thz': {'vals': ('terahertz', 'tera-hertz', 'th'),
                    'type': 'freq',
                    'fac': 10 ** 12}}
        self.reset()
        if unit:
            self.current_unit = self._resolve_units(unit)
        else:
            self.current_unit = None
        self.vals = vals

    def __call__(self, unit=None, vals=None):
        """Standard call for resolving various conditions."""
        """Allows repeat calls and inline calling of function.
        Main process to gather all files
        This returns object of itself. Use return functions to get needed items
        """
        if (type(unit) != str):
            vals = unit
            unit = None
        if unit and vals:
            # set new unit and new vals
            # print(1)
            self.current_unit = self._resolve_units(unit)
            self.vals = vals
            self.conv = (self.current_unit[2], self.current_unit[2])
        elif unit and self.vals:
            # set new unit and regen vals
            # print(2)
            _tmp = self._resolve_units(unit)
            if self.current_unit[2] != _tmp[2]:
                self.vals = self._return_vals(unit=_tmp)
            else:
                self._return_vals()
        elif vals and self.current_unit:
            # set new values for current unit
            # print(3)
            self.vals = self._return_vals(vals=vals)
        elif unit:
            # only set the new unit
            # print(4)
            self.current_unit = self._resolve_units(unit)
        elif vals:
            # print(5)
            self.vals = vals
        return self._return_vals()

    def reset(self):
        self.current_unit = None
        self.vals = None
        self.conv = (None, None)

    def set_base_unit(self, unit):
        _tmp = self._resolve_units(unit)
        self.current_unit = _tmp

    def converting(self):
        print(f'Converting from {self.conv[0]} to {self.conv[1]}')

    def get_units(self):
        """Return the units possible in the current setup."""
        return self.units.keys()

    def _resolve_units(self, bu):
        """Resolve the units and conversion factor."""
        tmp = self._resolve_name(bu)
        if tmp[0]:
            return tmp
        else:
            _u = self.get_units()
            self.exit(f'Unit: <{bu}> was not found in list of units: {_u}')

    def _resolve_name(self, bu):
        """Will resolve the name of the unit from known types."""
        if bu not in self.get_units():
            for i in self.units:
                for k in self.units[i]['vals']:
                    if bu == k:
                        return True, bu, i, self.units[i]['type']
            return False, False
        else:
            return True, bu, bu, self.units[bu]['type']

    def _conversion(self, init, ctype, fin, ftype, val):
        """Return conversion factor needed."""
        """ctype = current type (wavelength, frequency, energy)
        ftype = final type to resolve to (wavelength, frequency, energy)
        init is the initial unit
        fin is the final unit
        This assumes everything has already been resolved with units
        """
        # converting between common types (wavelength->wavelength)
        self.current_unit = self._resolve_name(fin)
        self.conv = (init,fin)
        if ctype == ftype:
            scaled = val * self.units[init]['fac']
        # converting from freq to wavelength
        elif ((ctype == 'freq') and (ftype == 'wave') or
              (ctype == 'wave') and (ftype == 'freq')):
            # print(c,self.units[init]['fac'], self.units[fin]['fac'])
            scaled = c / (val * self.units[init]['fac'])
        elif (ctype == 'energy') and (ftype == 'freq'):
            scaled = val * self.units[init]['fac'] / h
        elif (ctype == 'freq') and (ftype == 'energy'):
            scaled = h * val * self.units[init]['fac']
        elif ((ctype == 'energy') and (ftype == 'wave') or
              (ctype == 'wave') and (ftype == 'energy')):
            scaled = h * c / (val * self.units[init]['fac'])

        return scaled / self.units[fin]['fac']

    def _return_vals(self, vals=None, unit=None):
        """Convert the values appropriately."""
        """unit is the type _resolve_name output
        vals can be either single values or iterable."""
        # print(vals, unit)
        # print(self.scaler)
        if (unit and self.current_unit) or vals:
            # print('first')
            # convert list of self.vals to new unit
            if vals:
                self.vals = vals
            if typecheck(self.vals) and not isinstance(self.vals, np.ndarray):
                for i in range(len(self.vals)):
                    _tmp = self.vals[i]
                    self.vals[i] = self._conversion(*self.current_unit[2:
                                                              len(self.current_unit)],
                                                    *unit[2:len(unit)], _tmp)
            else:
                _tmp = self.vals
                self.vals = self._conversion(*self.current_unit[2:
                                                          len(self.current_unit)],
                                             *unit[2:len(unit)], _tmp)
        return self.vals

    def exit(self, exitcode, exitparam=0):
        """Handles error codes and exits nicely."""
        print(exitcode)
        print('v--------Ignore exit codes below--------v')
        if exitparam == 0:
            return None
        else:
            from sys import exit
            exit(0)

def planck_wav(temp=None, val=None, unit=None):
    """Plank Function in wavelength."""
    _c = Units(unit='angstroms', vals=c)('meters')
    _h = h

    wav = Units(unit=unit, vals=val)('meters')
    a = 2.0 * _h * _c ** 2
    b = _h * _c / (wav * kb * temp)
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
    # returns in units of watts/ m^2 / steradian / inputunit
    return intensity * Units(unit='meters', vals=1)(unit)


def planck_nu(temp=None, val=None, unit=None):
    """Plank Function in frequency."""
    _c = Units(unit='angstroms', vals=c)('meters')
    _h = h

    nu = Units(unit=unit, vals=val)('hz')
    a = 2.0 * _h / _c ** 2
    b = _h * nu / (kb * temp)
    intensity = a * (nu ** 3) / (np.exp(b) - 1.0)
    # returns in units of watts/ m^2 / steradian / inputunit
    return intensity * Units(unit='hz', vals=1)(unit)


def planck():
    pass

def dustmass(dist, dist_unit, val, val_unit, flux, temp, model_name, beta):
    """Calculate dust mass."""
    """Assuming temp in Kelvin, flux in Janskys"""
    dist = Units(unit=dist_unit, vals=dist)('cm') # to match the opacity units
    wav = Units(unit=val_unit, vals=val)('microns') # to search opcaity models
    intensity = planck_nu(temp, Units(unit=val_unit, vals=val)('hz'), 'hz') *\
        1.E26 # in jansky
    opacity = kappa(wav, model_name='oh1994', beta=beta) # cm^2/g
    toret = 'For the various opacities:\n'
    _ret = []
    for x in opacity:
        _tmp = dist**2 * flux / x / intensity / msun # in msun units (no ISM assump.)
        # print(x, dist, flux, intensity, _tmp)
        toret += '{}...{}\n'.format(x, _tmp)
        _ret.append(np.array([x, _tmp]))
    return toret, np.array(_ret)




# end of file

