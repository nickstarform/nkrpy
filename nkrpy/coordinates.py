"""Coordinate transformations."""

# standard modules
import math
import re

# relative modules
from .functions import typecheck
from .math import rad, deg

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
            _phi = deg(math.acos(self.c3 / _r))
        else:
            _phi = 90.

        if self.c1 != 0:
            _theta = deg(math.asin(self.c2 /
                                         (_r * math.sin(radians(_phi)))))
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
        _p = self.c1 * math.sin(radians(self.c3))
        _z = self.c1 * math.cos(radians(self.c3))
        self.c1 = _p
        self.c3 = _z
        self.sys = 'cyl'
        _i = self._asys.index(self.sys)
        self._hold[_i] = (self.c1, self.c2, self.c3)

    def _cyl_2_cart(self):
        _x = self.c1 * math.cos(radians(self.c2))
        _y = self.c1 * math.sin(radians(self.c2))
        self.c1 = _x
        self.c2 = _y
        self.sys = 'cart'
        _i = self._asys.index(self.sys)
        self._hold[_i] = (self.c1, self.c2, self.c3)

# end of file
