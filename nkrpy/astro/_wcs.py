'''.'''
# cython modules

# internal modules
from copy import deepcopy

# external modules
import numpy as np

# relative modules
from ..misc.functions import typecheck
from ..io import _fits as fits
from .._types import WCSClass
from ..misc.decorators import argCase
from ..misc.frozendict import FrozenDict as frozendict

# global attributes
__all__ = ('WCS',)
__doc__ = '''.'''
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class _base_wcs_class(object):
    '''Base Class object for WCS solving.

    This is a backend class, don't use.
    '''

    @argCase(case='lower')
    def __init__(self, delt: float = 0, rpix: float = 0, unit: str = '', rval: float = 0, axis: int = 0, axisnum: int = 0, **kwargs):
        self.__kwargs = {'rval': rval, 'rpix': rpix, 'delt': delt, 'unit': unit, 'axis': axis, 'axisnum': axisnum}
        if kwargs:
            kwargs.update(self.__kwargs)
            self.__kwargs = kwargs

    @argCase(case='lower')
    def __call__(self, *args, **kwargs):
        return self.__grab_values(*args, **kwargs)

    def to_fits(self):
        result = {}
        axis = self.__kwargs['axisnum']
        for k, v in self.__kwargs.items():
            if k in ['rval', 'rpix', 'delt']:
                result[f'c{k}{axis}'] = v
            if k.lower() == 'axis':
                result[f'n{k}{axis}'] = v
        return result

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        return result

    def copy(self):
        return self.__copy__()

    def __deepcopy__(self, memo):
        hsh = id(self)
        result = self.__copy__()
        memo[hsh] = result
        return result

    @argCase(case='lower')
    def __grab_values(self, val: float = np.nan, return_type: str = 'pix', find_nearest: bool = False):
        '''Intelligently return desired values.

        Usage
        -----
        > If nothing is specified, returns all axes
        > If val and return_type is specified, returns the converted values from either pix or wcs
        '''
        assert return_type in {'pix', 'wcs'}
        if not typecheck(val) and val == np.nan:
            return self.__kwargs
        if return_type == 'pix':
            pix = (val - self.__kwargs['rval']) / self.__kwargs['delt'] + \
                self.__kwargs['rpix']
            if isinstance(pix, np.ndarray):
                if find_nearest:
                    np.around(pix, 0, pix)
                pix.astype(int)
            elif find_nearest:
                pix = int(np.around(pix, 0))
            return pix
        val = (val - self.__kwargs['rpix']) * self.__kwargs['delt'] + \
            self.__kwargs['rval']
        return val

    def get(self):
        return self.__kwargs

_base_wcs_class.__call__.__doc__ = _base_wcs_class._base_wcs_class__grab_values.__doc__

class WCS(WCSClass):
    '''Generalized WCS object.

    To access the available axis after the header is loaded

    Parameters
    ----------
    wcs: dict
        A dictionary containing the common header items.
        Will search for ctype# to determine the axes and then look
            for other common header names.
    file: [optional]
        Must be a fits file with a proper header

    Example
    -------
    from nkrpy.io import fits
    a = fits.read('oussid.s15_0.Per33_L1448_IRS3B_sci.spw25.cube.I.iter1.image.fits')
    a = dict(a[0][0])
    b = WCS(a)
    b(500, 'wcs')
    b(51.28827040497872, 'pix')
    b(500, 'wcs', 'dec--sin')
    b(30.7503474380, 'pix', 'dec--sin')
    b(50, 'wcs', 'spectral')
    '''

    def __init__(self, wcs=None):
        if wcs is not None:
            if isinstance(wcs, str):
                h, d = fits.read(wcs)
                wcs = h
            elif isinstance(wcs, WCSClass):
                wcs = wcs.header
            if str(type(wcs)) == "<class 'astropy.io.fits.header.Header'>":
                self.header = wcs
            else:
                self.header = None
            # wcs is now a dictionary of header items
            self.__initialize_from_dict(wcs)

    def __initialize_from_dict(self, header: dict):
        head_lower = dict([[t.lower().replace(' ', ''), t]
                           for t in header.keys()])
        self.__header_lower = head_lower
        # define some common switches
        keys = [['restfreq', 'restfrq']]
        for k in keys:
            self.__switch_keys(self.__header_lower, *k)
        self.__header = dict(header)
        naxis = [t for t in self.__header_lower if 'ctype' in t]
        self.__axis = {}
        #from IPython import embed; embed() 
        for t in naxis:
            axis_heads = [[x.replace('naxis', 'axis').replace('c', '')[:4], x]
                          for x in self.__header_lower if (x.startswith('c') or x.startswith('naxis')) and
                          x.endswith(f'{t[-1]}')]
            vals = dict([[h[0], self.__header[self.__header_lower[h[1]]]] for h in axis_heads])
            aname = vals['type'].lower()
            vals['type'] = aname
            vals['axisnum'] = int(t[-1])
            self.__axis[aname] = _base_wcs_class(**vals)
            setattr(self, f'axis{t[-1]}', self.__axis[aname].get())

    @argCase(case='lower')
    def drop_axis(self, axis: str = 'freq'):
        v = self.get(axis)
        if v is None:
            return
        num = v['axisnum']
        # delete from header
        for k in v.keys():
            rk = (f'C{k}{num}').lower()
            if rk not in self.__header_lower:
                rk = (f'n{k}{num}').lower()
            if rk not in self.__header_lower:
                continue
            rk = self.__header_lower[rk]
            del self.__header[rk]
            if self.header is not None:
                del self.header[rk]
        # delete from axis dict
        del self.__axis[axis]
        # delete from WCS
        delattr(self, f'axis{num}')
        # delete from header

    @argCase(case='lower')
    def add_axis(self, axis: str, crval: float, crpix: float, cdelt: float, unit: str, size: int = 0):
        """Add axis to current WCS.

        On conflict do nothing.
        """
        v = self.get(axis)
        if v is not None:
            return
        unit = Unit.resolve_unit(unit)
        if unit is None:
            return
        unit = unit['name']
        axisnum = [axis['axisnum'] for axis in self.get_axes()]
        axisnum = max(axisnum) + 1
        vals = {'delt': cdelt, 'rpix': crpix, 'unit': unit,  'rval': crval, 'axis': size, 'axisnum': axisnum}
        self.__axis[axis] = _base_wcs_class(**vals)
        setattr(self, f'axis{axisnum}', self.__axis[axis].get())

    def get_header(self):
        """Get an immutable header."""
        return frozendict(self.__header)

    @argCase(case='lower')
    def set_head(self, key: str, val):
        """Get an immutable header."""
        if key not in self.__header_lower:
            return
        rk = self.__header_lower[key]
        try:
            self.__header[rk] = val
        except ValueError:
            return

    @staticmethod
    @argCase(case='lower')
    def __switch_keys(d: dict, k1: str, k2: str):
        k = None
        if k1 in d:
            k = (k1, k2)
        elif k2 in d:
            k = (k2, k1)
        if k is not None:
            d[k[1]] = d[k[0]]

    @argCase(case='lower')
    def get_axis_number(self, axis: str = 'freq'):
        v = self.get(axis)
        return v['axis']

    @argCase(case='lower')
    def __call__(self, *args, **kwargs):
        return self.__grab_values(*args, **kwargs)

    @argCase(case='lower')
    def __grab_values(self, val = None, return_type: str = None, axis: str = 'ra---sin', find_nearest: bool = False):
        '''Intelligently return desired values.

        Usage
        -----
        > If nothing is specified, returns all axes
        > If val and return_type is specified, returns the converted values from either pix or wcs. The axis must be specified
        '''
        if val is None:
            return self.get_axes()
        if isinstance(axis, str):
            axis = self.get_axis_base_object(axis)
        return axis(val=val, return_type=return_type, find_nearest=find_nearest)

    @argCase(case='lower')
    def __str__(self):
        ret = []
        for name, axis in self.__axis.items():
            ret.append(f'''{name}: {axis.get()}''')
        return ' | '.join(ret)

    @argCase(case='lower')
    def __repr__(self):
        return self.__str__()

    @argCase(case='lower')
    def array(self, start: int = None, stop: int = None, axis: str = 'freq', size: int = None, return_type: str = 'pix', find_nearest: bool = False):
        '''Generate an array of the specified axis.

        Parameters
        ----------
        size: int
            Default the size of the original axis, otherwise it is the new size, evenly separated from the start/end of the old values
        '''
        axis = self.__get_axis(axis)
        if size is None:
            size = axis.get()['axis']
        if start is None:
            array = np.arange(0, axis.get()['axis'], float(axis.get()['axis']/ size))
        elif start is not None or stop is not None:
            if start is None or start <= -1:
                start = 0
            if stop is None or stop <= -1 or stop == np.inf:
                stop = size
            array = np.arange(start, stop + 1, 1)
        if return_type == 'wcs':
            return self.__grab_values(val=array, return_type=return_type, axis=axis, find_nearest=find_nearest)
        return array

    def add_history(self, history: str):
        if 'history' not in self.__header_lower:
            self.__header['HISTORY'] = ''
            self.__header_lower['history'] = 'HISTORY'
        self.__header['HISTORY'] = str(self.__header['HISTORY']) + history

    def update_header(self):
        """Update the header with the information from the axis."""
        header = dict(self.get_header())
        head_lower = dict([[k.lower(), k] for k in header.keys()])
        for k in head_lower:
            if k.startswith('ctype'):
                rkey = head_lower[k]
                v = header[rkey]
                axis = self.get_axis_base_object(v)
                if axis is None:
                    continue
                newaxis = axis.to_fits()
                newaxis[k] = v
                for k, v in newaxis.items():
                    if k not in self.__header_lower:
                        continue
                    rk = self.__header_lower[k]
                    self.__header[rk] = v

    def create_fits_header(self, override: bool = True):
        if override:
            if self.header is not None:
                header = self.header
            else:
                header = {}
        else:
            header = {}
        for k, v in self.get_header().items():
            try:
                header[k] = v
            except ValueError:
                pass
        return header

    def refresh(self):
        """Refresh the axis objects."""
        self.__initialize_from_dict(self.__header)

    @argCase(case='lower')
    def get(self, axis: str = 'freq'):
        '''Nice wrapper to quickly get axis params.'''
        axis = self.__get_axis(axis=axis)
        if axis is not None:
            return axis.get()

    @argCase(case='lower')
    def get_head(self, key: str):
        return None if key not in self.__header_lower else self.__header[self.__header_lower[key]]

    def get_axes(self):
        '''Nice wrapper to quickly get axis params.'''
        axes = []
        for axis in self.__axis.values():
            axes.append(axis.get())
        return axes

    @argCase(case='lower')
    def __get_axis(self, axis: str = 'freq'):
        '''Return a single axis based on the unit type (spectral, ra dec).'''
        axis = axis.lower()
        if axis in self.__axis:
            return self.__axis[axis]

    @argCase(case='lower')
    def get_axis_base_object(self, axis: str = 'freq'):
        '''Return all axes available.'''
        return self.__get_axis(axis)

    def get_axes_base_object(self):
        '''Return all axes available.'''
        return [axis for axis in self.__axis.values()]

    @argCase(case='lower')
    def shift_axis(self, axis: str = 'ra---sin', unit: str = 'wcs', val: float = 0):
        '''Shift the axis.'''
        if val == 0:
            return
        unit = unit.lower()
        if isinstance(axis, str):
            axis = self.get_axis_base_object(axis)
        assert unit in ['wcs', 'pix']
        if unit == 'wcs':
            axis.get()['rval'] += val
        else:
            axis.get()['rpix'] += val

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def copy(self):
        '''Shallow Copy.'''
        return self.__copy__()

    def deepcopy(self):
        '''Full deep copy.'''
        return self.__deepcopy__(memo={})


def test():
    '''Testing function for module.'''
    pass


if __name__ == "__main__":
    '''Directly Called.'''

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
