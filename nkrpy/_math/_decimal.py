"""Various math functions."""
# cython: binding=True
# cython modules

# standard modules
import decimal as dml

# external modules
import numpy as np
from scipy import special
from decimal import Decimal
from mpmath import mp

# relative modules
from ..misc import functions as n_f

typecheck = n_f.typecheck
# global attributes
__all__ = ['Decimal']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class __base_decimal_obj(object):
    pass

class Decimal(__base_decimal_obj):

    def __init__(self, value):
        if typecheck(value):
            vals = []
            for i, v in enumerate(value):
                vals.append(Decimal(v))
            self.__val = vals
        elif typecheck(value, __base_decimal_obj):
            self.__val = value
        else:
            self.__val = Decimal(value)

    def __







