'''.'''
# cython modules

# internal modules
from copy import deepcopy

# external modules
import numpy as np
from astropy.wcs import WCS as astropy_WCS
from astropy.io import fits as astropy__fits

# relative modules
from ..misc.functions import typecheck
from ..io import fits
from .._types import WCSClass
from ..misc.decorators import argCase
from ..misc.frozendict import FrozenDict as frozendict
from .._unit import Unit

# global attributes
__all__ = ['WCS',]
__doc__ = '''.'''
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class _base_wcs_class(object):
    '''Base Class object for WCS solving.

    This is a backend class, don't use.
    '''

    __pi = np.pi
    __ln2 = np.log(2)
    def __init__(self, delt: float = 0, rpix: float = 0, unit: str = '', rval: float = 0, axis: int = 0, axisnum: int = 0, dtype: str='', **kwargs):
        delt, rpix, rval = map(lambda x: round(x, 10), [delt, rpix, rval])
        self.kwargs = {'rval': rval, 'rpix': rpix, 'delt': delt, 'unit': unit, 'axis': axis, 'axisnum': axisnum, 'dtype': dtype}
        if kwargs:
            kwargs.update(self.kwargs)
            self.kwargs = kwargs

    def set(self, delt: float = 0, rpix: float = 0, unit: str = '', rval: float = 0, axis: int = 0, axisnum: int = 0, dtype: str='', **kwargs):
        delt, rpix, rval = map(lambda x: round(x, 10), [delt, rpix, rval])
        self.kwargs = {'rval': rval, 'rpix': rpix, 'delt': delt, 'unit': unit, 'axis': round(axis,0), 'axisnum': axisnum, 'dtype': dtype}
        if kwargs:
            kwargs.update(self.kwargs)
            self.kwargs = kwargs
    def __call__(self, *args, **kwargs):
        return self.__grab_values(*args, **kwargs)

    def to_fits(self):
        result = {}
        axis = self.kwargs['axisnum']
        for k, v in self.kwargs.items():
            if k in ['rval', 'rpix', 'delt', 'unit', 'dtype']:
                k = k.replace('dtype', 'type')
                result[f'c{k}{axis}'.upper()] = v
            if k.lower() == 'axis':
                result[f'n{k}{axis}'.upper()] = v
        return result

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.kwargs = deepcopy(self.kwargs)
        return result

    def copy(self):
        return self.__copy__()

    def __deepcopy__(self, memo):
        hsh = id(self)
        result = self.__copy__()
        memo[hsh] = result
        return result
    def __grab_values(self, val: float = None, return_type: str = 'pix',declination_degrees: float = 0):
        '''Intelligently return desired values.

        Usage
        -----
        > If nothing is specified, returns all axes
        > If val and return_type is specified, returns the converted values from either pix or wcs

        > If return type is 'pix' then val must be wcs and vice versa
        '''
        return_type = 'pix' if return_type.startswith('pix') else 'wcs'
        if val is None:
            return self.kwargs
        delt = self.kwargs['delt'] / np.cos(declination_degrees*np.pi / 180.)
        if return_type == 'pix':
            pix = (val - self.kwargs['rval']) / delt + (self.kwargs['rpix'])
            return pix
        val = (val - self.kwargs['rpix']) * delt + \
            self.kwargs['rval']
        return np.around(val, 10)

    def __dict__(self):
        return self.kwargs

    def baseget(self):
        return self.kwargs

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

    __pi = np.pi
    __ln2 = np.log(2)
    def __init__(self, wcs=None, beamtable=None, **kwargs):
        self.__axis = {}
        if wcs is not None:
            if isinstance(wcs, str):
                h, _ = fits.read(wcs)
                wcs = dict(h)
            elif isinstance(wcs, WCSClass):
                self.resolve(wcs)
                return
            if str(type(wcs)) == "<class 'astropy.io.fits.header.Header'>":
                self.header = wcs
                wcs = dict(wcs)
            if isinstance(wcs, list):# assuming two astropy headers in alist
                if wcs[1]['EXTNAME'].lower().replace(' ', '') == 'beams':
                    self.beamtable = beamtable if beamtable is not None else wcs[1]
                self.header = wcs
                wcs = wcs[0]
            # wcs is now a dictionary of header items
            self.__initialize_from_dict(wcs)
        elif kwargs:
            self.__initialize_from_dict(kwargs)

    def resolve(self, wcs):
        self.header = wcs.get_header()
        self.__initialize_from_dict(self.header)
        self.__axis = wcs.get_axes_base_object()
        self.refresh_axes()

    def __bool__(self):
        if self.__axis:
            return True
        return False

    def __eq__(self, comp):
        if not isinstance(comp, WCSClass):
            raise ValueError(f'Not a valid comparison, must compare with another WCSClass-like object.')
            return False
        # first make sure same axis
        if self.get_axes() == comp.get_axes():
            return True

        for axis in self.__axis:
            baseaxis = self.__axis[axis].baseget()
            selfstart = self(0, 'wcs', axis=axis)
            compstart = comp(0, 'wcs', axis=axis)
            selfend = self(baseaxis['axis'], 'wcs', axis=axis)
            compend = comp(baseaxis['axis'], 'wcs', axis=axis)
            if (selfstart != compstart) or (selfend != compend):
                return False
        return True

    @staticmethod
    def remove_digit(s):
        return ''.join([i for i in s if not i.isdigit()])

    def __initialize_from_dict(self, header: dict):
        for x in list(header.keys()):
            if (x.lower().replace(' ', '') == 'cd1_2') or\
               (x.lower().replace(' ', '') == 'cd2_1'):
               del header[x]
            if (x.lower().replace(' ', '') == 'cd1_1') or\
               (x.lower().replace(' ', '') == 'cd2_2'):
               header['CDELT' + x.split('_')[-1]] = header[x]
               del header[x]

        head_lower = dict([[t.lower().replace(' ', ''), t]
                           for t in header.keys()])
        self.__header_lower = head_lower
        # define some common switches
        keys = [['restfreq', 'restfrq'], ['bmaj', 'bma'], ['bmin', 'bmi']]
        for k in keys:
            self.__switch_keys(self.__header_lower, *k)
        self.__header = dict(header)
        naxis = [t for t in self.__header_lower if 'ctype' in t]
        for ti, t in enumerate(naxis):
            axis_heads = [[self.remove_digit(x.replace('naxis', 'axis').replace('c', '').replace('type', 'dtype').replace('cd1_1', 'cdelt1').replace('cd2_2', 'cdelt2')), x]
                          for x in self.__header_lower if (x.startswith('c') or x.startswith('naxis')) and
                          x.endswith(f'{t[-1]}')]
            vals = dict([[h[0], self.__header[self.__header_lower[h[1]]]] for h in axis_heads])
            aname = vals['dtype'].lower()
            vals['dtype'] = aname
            vals['axisnum'] = int(t[-1])
            #self.refresh_axes()
            self.__axis[aname] = _base_wcs_class(**vals)
            self.refresh_axes()
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
        # delete from axis dict
        del self.__axis[axis]
        # delete from WCS
        delattr(self, f'axis{num}')
        # delete from header
    def add_axis(self, axis_name: str, crval: float, crpix: float, cdelt: float, unit: str, size: int = 0):
        """Add axis to current WCS.

        On conflict do nothing.
        """
        if self.get(axis_name) is not None:
            return
        axisnum_all = [axis['axisnum'] for _, axis in self.get_axes().items()]
        axisnum = set(np.arange(1, max(axisnum_all) + 1)) - set(axisnum_all)
        if len(axisnum) == 0:
            axisnum = max(axisnum_all) + 1
        else:
            axisnum = np.min(list(axisnum))
        vals = {'delt': cdelt, 'rpix': crpix, 'unit': unit,  'rval': crval, 'axis': size, 'axisnum': axisnum, 'dtype': axis_name}
        self.__axis[axis_name] = _base_wcs_class(**vals)
        setattr(self, f'axis{axisnum}', self.__axis[axis_name].baseget())
        self.refresh_axes()
        #self.update_fits_header()
        self.update_WCS_header()

    def get_header(self):
        """Get an immutable header."""
        return frozendict(self.__header)
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
    def __switch_keys(d: dict, k1: str, k2: str):
        k = None
        if k1 in d:
            k = (k1, k2)
        elif k2 in d:
            k = (k2, k1)
        if k is not None:
            d[k[1]] = d[k[0]]
    def get_axis_number(self, axis: str = 'freq'):
        v = self.get(axis)
        return v['axis']
    def __call__(self, *args, **kwargs):
        return self.__grab_values(*args, **kwargs)
    def __grab_values(self, val = None, return_type: str = 'wcs', axis: str = 'ra---sin', declination_degrees = 0):
        '''Intelligently return desired values.

        Usage
        -----
        > If nothing is specified, returns all axes
        > If val and return_type is specified, returns the converted values from either pix or wcs. The axis must be specified
        > If you are selecting values from the ra axis, need to adjust by factor of cos(declination)
        > forcing axis to be string, otherwise slows programs.
        '''
        if val is None:
            return self.get_axes()
        axis = self.get_axis_base_object(axis)
        return axis(val=val, declination_degrees=declination_degrees, return_type=return_type)
    def __str__(self):
        ret = []
        for name, axis in self.__axis.items():
            ret.append(f'''{name}: {axis.baseget()}''')
        return ' | '.join(ret)
    def __repr__(self):
        return self.__str__()

    def beam2(self):
        # convert the beamarea to square pixels
        beam =  self.get_beam()[:-1]
        return beam[0] * beam[1]

    def beam2as(self):
        # convert the beamarea to square pixels
        beam =  self.get_beam()[:-1]
        return beam[0] * beam[1] * 3600 ** 2

    def beam2pix(self):
        # convert the beamarea to square pixels
        return abs(self.__beam2pix(*self.get_beam()[:-1], abs(self.axis1['delt'])))


    @classmethod
    def __beam2pix(cls, bma, bmi, cellsize):
        # assume all in same coordinates
        return abs(cls.__pi * bma * bmi / (4* cls.__ln2) / (cellsize ** 2))

    def array(self, start: float = None, stop: float = None, axis: str = 'freq', startstop_type: str = 'pix', size: int = None, return_type: str = 'pix', declination_degrees=0):
        '''Generate an array of the specified axis.

        Parameters
        ----------
        size: int
            Default the size of the original axis, otherwise it is the new size, evenly separated from the start/end of the old values
        '''
        axis = self.__get_axis(axis)
        startstop_type = 'pix' if startstop_type.startswith('pix') else 'wcs'
        if size is None:
            size = axis.baseget()['axis']
        if start is None:
            start = 0
        elif startstop_type == 'wcs':
            start = self.__grab_values(val=start, return_type='pix', axis=axis, declination_degrees=declination_degrees)
        if stop is None:
            stop = axis.baseget()['axis'] - 1
        elif startstop_type == 'wcs':
            stop = self.__grab_values(val=stop, return_type='pix', axis=axis, declination_degrees=declination_degrees)

        ite = -1 if axis.baseget()['delt'] < 0 else 1
        start, stop = sorted([start, stop])
        array = np.linspace(start=start, stop = stop, num=int(size), dtype=float) + 1
        if return_type == 'wcs':
            return self.__grab_values(val=array, return_type=return_type, axis=axis, declination_degrees=declination_degrees)[::ite]
        return array[::ite]

    def add_history(self, history: str):
        if 'history' not in self.__header_lower:
            self.__header['HISTORY'] = ''
            self.__header_lower['history'] = 'HISTORY'
        self.__header['HISTORY'] = str(self.__header['HISTORY']) + history

    def update_WCS_header(self):
        """Update the header current with the information from the axis."""
        header = self.create_fitsheader_from_axes()
        self.__header = header

    def create_fitsheader_from_axes(self):
        """Create a fits header from the wcs axis only."""
        header = {}
        for _, axis in self.__axis.items():
            axisfits = axis.to_fits()
            for k, v in axisfits.items():
                if isinstance(v, str):
                    v = v.upper()
                if len(k.lower().replace('axis', '')) != len(k):
                    v = int(v)
                header[k.upper()] = v
        header['NAXIS'] = len(self.__axis)
        for h in self.__header:
            if h not in header:
                header[h] = self.__header[h]
        print(header)
        return header

    def update_fits_header(self):
        """Refresh the header using astropy to make it writable to fits"""
        self.header = astropy__fits.header.Header(self.__header)

    def refresh_axes(self):
        for k in dir(self):
            if k.startswith('axis'):
                delattr(self, f'{k}')
        for i, k in enumerate(self.__axis.keys()):
            self.__axis[k] = _base_wcs_class(**{**self.__axis[k].baseget(), 'axisnum': i + 1, 'dtype': k})
            setattr(self, f'axis{i+1}', self.__axis[k].baseget())
        pass

    def refresh_from_header(self):
        """Refresh the axis objects."""
        self.__initialize_from_dict(dict(self.__header))

    def get(self, axis: str = 'freq'):
        '''Nice wrapper to quickly get axis params.'''
        axis = self.__get_axis(axis=axis)
        if axis is not None:
            return axis.baseget()

    def get_head(self, key: str):
        return None if key not in self.__header_lower else self.__header[self.__header_lower[key]]

    def get_axes(self):
        '''Nice wrapper to quickly get axis params.'''
        axes = {}
        for aname, axis in self.__axis.items():
            axes[aname] = axis.baseget()
        return axes

    def __get_axis(self, axis: str = 'freq'):
        '''Return a single axis based on the unit type (spectral, ra dec).'''
        if not isinstance(axis, str):
            axis = axis.kwargs['dtype']
        if axis in self.__axis:
            return self.__axis[axis]

    def get_beam(self):
        if hasattr(self, 'beamtable'):
            bma = self.beamtable['BMAJ']
            bmi = self.beamtable['BMIN']
            bpa = self.beamtable['BPA']

            bma, bmi, bpa = map(np.median, (bma, bmi, bpa))
        else:
            bma = self.get_head('bmaj')
            bmi = self.get_head('bmin')
            bpa = self.get_head('bpa')
        return bma, bmi, bpa

    def get_beam_pix(self):
        bma, bmi, bpa = self.get_beam()
        return abs(bma/self.axis1['delt']), abs(bmi/self.axis1['delt']), bpa

    def set_beam(self, bmaj, bmin, bpa):
        self.set_head('bmaj', bmaj)
        self.set_head('bmin', bmin)
        self.set_head('bpa', bpa)
        self.refresh_from_header()
        self.update_WCS_header()
        self.update_fits_header()
    def get_axis_base_object(self, axis: str = 'freq'):
        '''Return all axes available.'''
        return self.__get_axis(axis)

    def get_axes_base_object(self):
        '''Return all axes available.'''
        return dict([[aname, axis] for aname, axis in self.__axis.items()])
    def shift_axis(self, val: float = 0, axis: str = 'ra---sin', unit: str = 'wcs'):
        '''Shift the axis.'''
        if val == 0:
            return
        unit = unit.lower()
        axis = self.get_axis_base_object(axis)
        unit = 'pix' if unit.startswith('pix') else 'wcs'
        if unit == 'wcs':
            axis.baseget()['rval'] += val
        else:
            axis.baseget()['rpix'] += val
    def center_axis_pix(self, pix: int, width:int, axis='ra---sin'):
        '''Center the axis.

        val is the value in wcs coords
        width is the width in wcs coords
        Center the current axis onto val with a certain width
        nearest will center to the closest pixel
        '''
        axis_baseobj = self.get_axis_base_object(axis)
        axis = axis_baseobj.baseget()
        axis_baseobj.set(delt=axis['delt'], rpix=pix, unit=axis['unit'], rval=axis['rval'], axis=width, axisnum=axis['axisnum'], dtype=axis['dtype'])
        self.refresh_axes()
    def center_axis_wcs(self, val: float, width: float = None, axis='ra---sin', declination_degrees=0):
        '''Center the axis.

        val is the value in wcs coords
        width is the width in wcs coords
        Center the current axis onto val with a certain width
        nearest will center to the closest pixel
        '''
        axis_baseobj = self.get_axis_base_object(axis)
        axis = axis_baseobj.baseget()
        if val == 0:
            return
        if width is None:
            return
        width = abs(width / (axis['delt']))
        axis_baseobj.set(delt=axis['delt'], rpix=width/2, unit=axis['unit'], rval=val, axis=round(width,0), axisnum=axis['axisnum'], dtype=axis['dtype'])
        self.refresh_axes()

    def export(self):
        return dict([[h.upper(), self.__header[v]] for h, v in self.__header_lower.items()])


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
