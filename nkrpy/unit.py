"""Unit conversion."""

# internal modules
from collections.abc import (MutableSequence, MutableSet, MutableMapping)
from inspect import isclass

# external modules
from numpy import ndarray

# relative modules
from .functions import typecheck
from .constants import h, c, kb  # imported as cgs
from ._unit import units

# global attributes
__all__ = ('Unit',)
__doc__ = """Convert supported units all to angstroms and hz
    to add new units have to correct self.__units and resolve_units
    To setup, just initialize and call with units /  values to convert
    Holds values to quick accessing later

    Use Units.converting for info on how you are converting
    """
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

c = c * 1E8  # A/s
h = h * 1E-7  # SI
kb = kb * 1E-7  # SI


def checknum(num):
    try:
        _ = float(num)
        return True
    except Exception:
        return False


class BaseUnit(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def keys(self):
        return self.__dict__.keys()
    def __iter__(self):
        return self.__dict__.items().__iter__()
    def __getitem__(self, key):
        return getattr(self, key)
    def __next__(self):
        pass
    def __setattr__(self):
        pass
    def __delattr__(self):
        pass


class Unit(object):
    """Convert between major unit types."""

    def __init__(self, baseunit: str=None, convunit: str=None, vals=None):
        """Main unit building.

        Parameters
        ----------
        baseunit: str
            The base unit to convert from.
        convunit: str
            The unit to convert to. Not Required.
        vals: number | numpy.ndarray
            The numbers to convert. Not Required.
        """

        self.__units = BaseUnit(**units)
        self.reset()
        if baseunit is not None:
            self.__current_unit = self.__resolve_units(baseunit)
        if convunit is not None:
            self.__final_unit = self.__resolve_units(convunit)
        if baseunit is None and convunit is None:
            return
        self.__current_vals = vals
        self.__final_vals = self.__return_vals(vals=vals)

    def __call__(self, convunit: str=None, vals=None):
        """Standard call for resolving various conditions.

        Parameters
        ----------
        unit: str
        vals: float | ndarray
        Allows repeat calls and inline calling of function.
        Main process to gather all files
        This returns object of itself. Use return functions to get needed items
        """
        if not isinstance(vals, ndarray) and not checknum(vals) and (vals is not None):
            return

        if isinstance(convunit, ndarray) or checknum(convunit):
            vals = convunit
            convunit = None

        if vals is None and convunit is None:
            unit = self.__current_unit
        elif convunit is not None and vals is not None:
            unit = self.__resolve_units(convunit)
            self.__current_unit = unit
            self.__final_unit = unit
            self.__current_vals = vals
        elif convunit is not None:
            unit = self.__resolve_units(convunit)
            vals = self.__current_vals
        elif convunit is None:
            unit = self.__final_unit
        vals = self.__return_vals(unit=unit, vals=vals)
        return vals

    def __repr__(self):
        """Representative Magic Method for calling."""
        self.__return_vals()
        _t = self.__final_vals
        if typecheck(_t):
            return ', '.join(map(str,_t))
        else:
            return str(f'{_t}')

    def __abs__(self):
        return abs(self.__final_vals)

    def __add__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals + value

    def __radd__(self, *args, **kwargs):
        return self.__add__(*args, **kwargs)

    def __sub__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals - value

    def __rsub__(self, *args, **kwargs):
        return self.__sub__(*args, **kwargs)

    def __divmod__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals / value

    def __rdivmod__(self, *args, **kwargs):
        return self.__divmod__(*args, **kwargs)

    def __truediv__(self, *args, **kwargs):
        return self.__divmod__(*args, **kwargs)

    def __rtruediv__(self, *args, **kwargs):
        return self.__truediv__(*args, **kwargs)

    def __mul__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals * value

    def __rmul__(self, *args, **kwargs):
        return self.__mul__(*args, **kwargs)

    def __pow__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals ** value

    def __rpow__(self, *args, **kwargs):
        return self.__pow__(*args, **kwargs)

    def __mod__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals % value

    def __rmod__(self, *args, **kwargs):
        return self.__mod__(*args, **kwargs)

    def __floordiv__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals // value

    def __rfloordiv__(self, *args, **kwargs):
        return self.__floordiv__(*args, **kwargs)

    def __int__(self):
        return int(self.__final_vals)

    def __float__(self):
        return float(self.__final_vals)

    def __gt__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals > value

    def __lt__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals < value

    def __le__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals <= value

    def __eq__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals == value

    def __ge__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals >= value

    def __ne__(self, value):
        if isinstance(value, Unit):
            value = value.__final_vals
        return self.__final_vals != value

    @property
    def debug(self):
        print(f"""
            Current Unit: {self.__current_unit}\n
            Current Values: {self.__current_vals}\n
            Final Unit: {self.__final_unit}\n
            Final Values: {self.__final_vals}""")

    def reset(self):
        self.__current_unit = None
        self.__current_vals = None
        self.__final_vals = None
        self.__final_unit = None
        self.__units = BaseUnit(**units)

    def set_base_unit(self, unit):
        """Set the base unit, don't reset vals."""
        _tmp = self.__resolve_units(unit)
        self.__current_unit = _tmp
        self.__return_vals()

    @property
    def get_base_val(self):
        """Get the base val."""
        return self.__current_vals

    @property
    def get_units(self):
        """Return the units possible in the current setup."""
        return self.__units.keys()

    def __resolve_units(self, bu):
        """Will resolve the name of the unit from known types."""
        bu = str(bu).lower()
        if bu not in self.get_units:
            for i in self.get_units:
                if bu in self.__units[i]['vals']:
                    return self.__units[i]
            return None
        else:
            return self.__units[bu]

    def __conversion(self, vals = None):
        """Return conversion factor needed.

        Parameters
        ----------
        ctype = current type (wavelength, frequency, energy)
        ftype = final type to resolve to (wavelength, frequency, energy)
        init is the initial unit
        fin is the final unit
        This assumes everything has already been resolved with units
        """
        if vals is None:
            vals = self.__current_vals
        # converting between common types (wavelength->wavelength)
        if self.__current_unit is None or self.__final_unit is None:
            return None
        ctype, ftype = self.__current_unit['type'], self.__final_unit['type']
        if ctype == ftype:
            scaled = vals * self.__current_unit['fac']
        # converting from freq to wavelength
        elif ((ctype == 'freq') and (ftype == 'wave') or
              (ctype == 'wave') and (ftype == 'freq')):
            # print(c,self.__units[init]['fac'], self.__units[fin]['fac'])
            scaled = c / (vals * self.__current_unit['fac'])
        elif (ctype == 'energy') and (ftype == 'freq'):
            scaled = vals * self.__current_unit['fac'] / h
        elif (ctype == 'freq') and (ftype == 'energy'):
            scaled = h * vals * self.__current_unit['fac']
        elif ((ctype == 'energy') and (ftype == 'wave') or
              (ctype == 'wave') and (ftype == 'energy')):
            scaled = h * c / (vals * self.__current_unit['fac'])

        return scaled / self.__final_unit['fac']

    def __return_vals(self, unit=None, vals=None):
        """Convert the values appropriately.

        unit is the type _resolve_name output
        vals can be either single values or iterable."""
        # print(vals, unit)
        if unit is None and vals is None:
            if self.__final_unit is self.__current_unit:
                self.__final_vals = self.__current_vals
        if unit is not None:
            self.__final_unit = unit
        else:
            if self.__final_unit is None:
                self.__final_unit = self.__current_unit
        if vals is not None:
            # convert from cu to fu with vals
            return self.__conversion(vals)
        else:
            # convert from cu to fu with self.vals
            self.__final_vals = self.__conversion()

# end of code
