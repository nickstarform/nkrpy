"""."""

# internal modules
import collections
from collections.abc import Iterable
from copy import deepcopy

# external modules

# relative modules

# global attributes
__all__ = ('FrozenOrderedDict', 'FrozenDict')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class FrozenDict(collections.Mapping):
    """Mutable parent class for frozendicts."""
    _cls = dict

    def __init__(self, *args, **kwargs):
        """Dunder."""
        self.__built = 0
        _d = self._cls()
        for x in args:
            _d.update(x)
        _d_kwargs = self._cls(**kwargs)
        _d.update(_d_kwargs)
        self.__dict = _d
        self._hash = None
        self.__built = 1

    def __getitem__(self, key):
        """Dunder."""
        iterable = self.__dict[key]
        if isinstance(iterable, Iterable) and not isinstance(iterable, str):
            return deepcopy(iterable)
        return self.__dict[key]

    def __contains__(self, key):
        """Dunder."""
        return key in self._keys

    def __iter__(self):
        """Dunder."""
        return iter(self.__dict)

    def __len__(self):
        """Dunder."""
        return len(self._keys)

    def __repr__(self):
        """Dunder."""
        return f'<{self.__class__.__name__!s} {self.__dict!r}>'

    def __hash__(self):
        """Dunder."""
        if self._hash is None:
            h = 0
            for key, value in iteritems(self.__dict):
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def __copy__(self):
        """Dunder."""
        return self.__class__(self)

    def __deepcopy__(self):
        """Dunder."""
        return self.__class__(self)

    def copy(self):
        """Dunder."""
        return self.__copy__()

    def append(self, *add_or_replace_args, **add_or_replace_kwargs):
        """Dunder."""
        return self.__class__(self, *add_or_replace_args, **add_or_replace_kwargs)


class FrozenOrderedDict(FrozenDict):
    """'Frozen' Dictionary that preserves insert order."""
    _cls = collections.OrderedDict
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# end of code

# end of file
