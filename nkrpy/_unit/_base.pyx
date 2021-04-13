"""Unit conversion."""
# cython modules

# internal modules
from copy import deepcopy
import operator

# external modules
import numpy as np

# relative modules
from ..misc.functions import typecheck
from .._types import UnitClass

# global attributes
__all__ = ('BaseUnit', 'BaseVals',)
__doc__ = """Base Units for the unit conversion module.
    """
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class BaseVals(UnitClass):
    """Generalized value holder.

    Normalizes numbers into iterables for (not)inplace operations."""

    __slots__ = ['vals', '__inplace']

    def __init__(self, vals, inplace: bool = False):
        self.__inplace = inplace
        if not inplace:
            vals = deepcopy(vals)
        self.vals = vals

    def __len__(self):
        if not typecheck(self.vals):
            return 1
        return len(self.vals)

    def __repr__(self):
        return f'{self.vals}'

    def __str__(self):
        return f'{self.vals}'

    def _inplace(self, func, val=None, override_inplace = False):
        if isinstance(func, str):
            func = getattr(operator, func)
        if val is not None:
            if override_inplace:
                if not typecheck(self.vals):
                    self.vals = func(self.vals, val)
                    return self.vals
                elif isinstance(self.vals, np.ndarray):
                    """
                    if func.__name__ in dir(np):
                        func = getattr(np, func.__name__)
                        return func(val, out=np.vals)
                    """
                    return func(self.vals, val)
                for i in range(len(self.vals)):
                    self.vals[i] = func(self.vals[i], val)
            else:
                if not typecheck(self.vals):
                    return func(self.vals, val)
                if isinstance(self.vals, np.ndarray):
                    return func(self.vals, val)
                t = []
                for i in range(len(self.vals)):
                    t.append(func(self.vals[i], val))
                return type(self.vals)(t)
        else:
            if override_inplace:
                if not typecheck(self.vals):
                    self.vals = func(self.vals)
                    return self.vals
                if isinstance(self.vals, np.ndarray):
                    """
                    if func.__name__ in dir(np):
                        func = getattr(np, func.__name__)
                        return func(val, out=np.vals)
                    """
                    return func(self.vals)
                for i in range(len(self.vals)):
                    self.vals[i] = func(self.vals[i])
            else:
                if not typecheck(self.vals):
                    return func(self.vals)
                if isinstance(self.vals, np.ndarray):
                    return func(self.vals)
                t = []
                for i in range(len(self.vals)):
                    t.append(func(self.vals[i]))
                return type(self.vals)(t)

    def reciprocal(self, override_inplace = False):
        inplace = (self.__inplace and override_inplace) or override_inplace
        recip = lambda x: 1. / x if not isinstance(self.vals, np.ndarray) else np.reciprocal
        return self._inplace(recip, override_inplace = inplace)

    def __abs__(self):
        """Dunder."""
        return self._inplace(abs)

    def __iadd__(self, value: float):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.iadd, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __add__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.add, value, False)
        return value

    def __radd__(self, *args, **kwargs):
        """Dunder."""
        return self.__add__(*args, **kwargs)

    def __isub__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.isub, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __sub__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.sub, value)
        return value

    def __rsub__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        return value - self.vals

    def __divmod__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.truediv, value)
        return value

    def __idivmod__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.itruediv, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __rdivmod__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        return value - self.vals

    def __truediv__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.truediv, value)
        return value

    def __itruediv__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.itruediv, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __rtruediv__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        return value - self.vals

    def __mul__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.mul, value)
        return value

    def __imul__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        self._inplace(operator.imul, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __rmul__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.mul, value)
        return value

    def __irmul__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.imul, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __pow__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.pow, value)
        return value

    def __ipow__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.ipow, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __rpow__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = operator.pow(value, self.vals)
        if not typecheck(self.vals):
            return value
        return self

    def __mod__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.mod, value)
        return value

    def __imod__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.imod, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __rmod__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = operator.mod(value, self.vals)
        return value

    def __floordiv__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.floordiv, value)
        return value

    def __ifloordiv__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = self._inplace(operator.ifloordiv, value, True)
        if not typecheck(self.vals):
            return self.vals
        return self

    def __rfloordiv__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        value = operator.ifloordiv(value, self.vals)
        return value

    def __iter__(self):
        """Dunder."""
        if not typecheck(self.vals):
            for i in [self.vals]:
                yield i
        else:
            for i in self.vals:
                yield i

    def __getitem__(self, key):
        """Dunder."""
        if not typecheck(self.vals):
            return self.vals
        else:
            return self.vals[key]

    def __setitem__(self, key, val):
        """Dunder."""
        if not typecheck(self.vals):
            self.vals = val
        else:
            self.vals[key] = val

    def __next__(self):
        """Dunder."""
        if not typecheck(self.vals):
            raise TypeError('Not an iterable')
        else:
            for i in self.vals:
                yield i

    def __gt__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        return self.vals > value

    def __lt__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        return self.vals < value

    def __eq__(self, value):
        """Dunder."""
        if isinstance(value, UnitClass):
            value = value.vals
        return self.vals == value

class BaseUnit(object):
    """Override typical dict classing."""

    def __init__(self, **entries):
        """Dunder."""
        self.__dict__.update(entries)

    def values(self):
        """Dunder."""
        return self.__dict__.values()

    def items(self):
        """Dunder."""
        return self.__dict__.items()

    def keys(self):
        """Dunder."""
        return self.__dict__.keys()

    def __iter__(self):
        """Dunder."""
        return self.__dict__.items().__iter__()

    def __getitem__(self, key):
        """Dunder."""
        return self.__dict__[key]

    def __next__(self):
        """Dunder."""
        pass

    def __setattr__(self):
        """Dunder."""
        pass

    def __delattr__(self):
        """Dunder."""
        pass

# end of code

# end of file
