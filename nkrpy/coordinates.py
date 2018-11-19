"""Coordinate transformations."""

# import standard modules
import math
import re

# relative modules
from .functions import typecheck


def checkconv(coord, conv='deg'):
    """Convert coord to a list."""
    """Handles RA, dec conversions.
    Possible conv: deg, icrs.
    Workflow: icrs->deg
    > convert input to two lists of the deg, mm, ss
    > convert and return a float
    Workflow: deg->icrs
    > convert input to two floats
    > return two strings with : delimiters"""
    conversion = ('deg', 'icrs')
    oregex = r""
    if conv == 'deg':
        # convert from icrs to deg
        # first check if string and convert to list
        assert not (isinstance(coord, float) or isinstance(coord, int))
        case = 1
        while isinstance(coord, str) and (case != -1):
            coord = coord.lower()
            if case == 1:
                # try just splitting
                delimiters = (':', ' ', ',')
                for delim in delimiters:
                    coord = coord.split(delim)
                    if len(coord) == 1:
                        coord = coord[0]
                    if typecheck(coord):
                        toret = coord
                        case = -1
                        break
            if case == 2:
                # now try via regex
                for delim in (('h', 'm', 's'), ('d', 'm', 's'),
                              ('h', 'm'), ('d', 'm'),
                              ('h', ), ('d', )):
                    if case != -1:
                        regex = oregex
                        _tmp = []
                        for i, j in enumerate(delim):
                            regex += r'(-?[0-9]+\.?[0-9]+?)' + f'{j}'
                        regex += r'(.*(.*))$'
                        # print(regex)
                        matches = re.finditer(regex, coord, re.MULTILINE)
                        matched = False
                        for match in matches:
                            for groupNum in range(0, len(match.groups())):
                                    groupNum = groupNum + 1
                                    _tmp.append(match.group(groupNum))
                                    matched = True
                        if matched:
                            toret = _tmp
                            case = -1
                            break
            if case == 3:
                print('Failed')
                exit()
            if case == -1:
                break
            case += 1
        # print(toret, regex, delim)
        # toret should be the returned list of len 3
        if len(toret) == 1:
            toret = [*toret, 0, 0]
        if len(toret) == 2:
            toret = [*toret, 0]
        if len(toret) > 3:
            toret = toret[0:3]
        for x in toret:
            if x == '':
                toret[toret.index(x)] = 0
        temp0, temp1, temp2 = tuple(map(float, toret))
        total = abs(temp0) + temp1 / 60. + temp2 / 3600
        if temp0 < 0:
            total = -1. * total
        return total

    elif conv == 'icrs':
        # convert from deg to icrs
        toret = abs(float(coord))
        _t1 = int(toret)
        _t2 = (toret - _t1) * 60.
        _t3 = (_t2 - int(_t2)) * 60.
        _toret = ':'.join(map(str, [int(_t1), int(_t2), _t3]))
        if float(coord) < 0:
            _toret = '-' + _toret
        return _toret
    else:
        return False


def rad_2_deg(rad):
    """Convert radian to degrees."""
    return 180. * rad / math.pi


def deg_2_rad(deg):
    """Convert degrees to radians."""
    return deg / 180. * math.pi


class coord(object):
    """Class holding coordinate objects."""

    """Can handle cartesian, spherical, cylindrical.
    Degrees are assumed. Also all coords are
    same units it possible.
    Cyl: assume theta is from x to y
    sph: assume phi from z.
    Usage:
    > a = coord(0,0,0,'cart')
    > a()
    > a."""

    def __init__(self, c1=0, c2=0, c3=0, sys='cart'):
        """Init."""
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.sys = sys
        self._asys = ('cart', 'sph', 'cyl')
        _i = self._asys.index(sys)
        self._hold = [None, None, None]
        self._hold[_i] = (self.c1, self.c2, self.c3)

    def __call__(self, sys=None, c1=None, c2=None, c3=None):
        """calling."""
        if c1 == None:
            if sys:
                i1 = self._asys.index(self.sys)
                i2 = self._asys.index(sys)
                if not self._hold[i2]:
                    while i1 != i2:
                        i1 = (i1 + 1) % len(self._asys)
                        if i1 == 0:
                            self._cyl_2_cart()
                        elif i1 == 1:
                            self._cart_2_sph()
                        elif i1 == 2:
                            self._sph_2_cyl()
                    return self
                else:
                    self.c1, self.c2, self.c3 = self._hold[i2]
                    self.sys = sys
        else:
            self.c1 = c1
            self.c2 = c2
            self.c3 = c3
            self.sys = sys
            _i = self._asys.index(self.sys)
            for i in range(len(self._hold)):
                if _i == i:
                    self._hold[_i] = (self.c1, self.c2, self.c3)
                else:
                    self._hold[i] = None
        return self

    def round(self, precision=16, hold=False):
        """Allow superficial (or True) rounding."""
        if not hold:
            c1 = round(self.c1, precision)
            c2 = round(self.c2, precision)
            c3 = round(self.c3, precision)
            return ','.join(map(str, [c1, c2, c3, self.sys]))
        else:
            self.c1 = round(self.c1, precision)
            self.c2 = round(self.c2, precision)
            self.c3 = round(self.c3, precision)
            _i = self._asys.index(self.sys)
            self._hold[_i] = (self.c1, self.c2, self.c3)
            return self

    def __repr__(self):
        """Representative string."""
        return ','.join(map(str, [self.c1, self.c2, self.c3, self.sys]))

    def _cart_2_sph(self):
        _r = (self.c1 ** 2 + self.c2 ** 2 + self.c3 ** 2) ** 0.5

        if self.c3 != 0:
            _phi = rad_2_deg(math.acos(self.c3 / _r))
        else:
            _phi = 90.

        if self.c1 != 0:
            _theta = rad_2_deg(math.asin(self.c2 /
                                         (_r * math.sin(deg_2_rad(_phi)))))
            if self.c1 < 0:
                _theta += 90
        else:
            _theta = 90.
        print(_theta)

        if self.c2 == 0:
            _theta = 0

        if (self.c1 == 0) and (self.c2 == 0) and (self.c3 == 0):
            self.c1 = 0
            self.c2 = 0
            self.c3 = 0
        else:
            self.c1 = _r
            self.c2 = _theta
            self.c3 = _phi
        self.sys = 'sph'
        _i = self._asys.index(self.sys)
        self._hold[_i] = (self.c1, self.c2, self.c3)

    def _sph_2_cyl(self):
        _p = self.c1 * math.sin(deg_2_rad(self.c3))
        _z = self.c1 * math.cos(deg_2_rad(self.c3))
        self.c1 = _p
        self.c3 = _z
        self.sys = 'cyl'
        _i = self._asys.index(self.sys)
        self._hold[_i] = (self.c1, self.c2, self.c3)

    def _cyl_2_cart(self):
        _x = self.c1 * math.cos(deg_2_rad(self.c2))
        _y = self.c1 * math.sin(deg_2_rad(self.c2))
        self.c1 = _x
        self.c2 = _y
        self.sys = 'cart'
        _i = self._asys.index(self.sys)
        self._hold[_i] = (self.c1, self.c2, self.c3)

# end of file
