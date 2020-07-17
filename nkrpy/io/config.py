"""."""
# internal modules
from sys import version
from os import getcwd
import importlib
from datetime import datetime
import pickle
import warnings

# external modules

# relative modules

# global attributes
__all__ = ('Config', )
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = float(version[0:3])
example_typedict = {
    'keys': {'dtype': str, 'opt': True},
}


class _base_config_object(object):
    """Base configuration object.

    Turn dictionaries -> modules and allow backconversion.
    """

    def __init__(self, inp):
        self.base = dir(self)
        if isinstance(inp, dict):
            for key, value in inp.items():
                setattr(self, key, value)
        else:
            for key in dir(inp):
                if '__' not in key and key not in self.base:
                    setattr(self, key, getattr(inp, key))

    def to_dict(self):
        ret = {}
        for key in dir(self):
            if '__' not in key and key not in self.base:
                ret[key] = getattr(self, key)
        return ret


# main class object for manipulating configuration files
class Config(object):
    """Python Configuration generator/verifier.

    Currently this only handles python and pickle files.

    Methods
    =======
    refresh:
        refreshes the configuration only if a file is supplied
    config:
        returns dictionary of iterator
    items:
        returns dict_item iterator of keys of configuration
    vals:
        returns dict_list of keys of configuration
    keys:
        returns dict_list of keys of configuration
    save:
        saves the config to a file given a filename
    __getitem__:
        allows for requesting of parameters dictionary style; config[key]
    __setitem__:
        allows for setting of parameters dictionary style; config[key] = value

    As Intended
    ===========
    Typedict is the master holder.
    * If key in typedict and not optional, it must be in config and target and pass type check.
    * If key in typedict and optional, if it appears in config or target, it must pass type check.
    * If key not in typedict, then auto pass.
    * If key in default but not in target, then default values are inherited, unless above conditions hold.
    * Getting/Setting can be granted by either indexing or attribute getters config[key] or config.key
    * Getting/Setting are non yielding
    * config is iterable and yields dict_iterable

    Example
    =======
    # test standard
    config(target={'a': 5, 'b': 'True'}, default={'a': 1, 'b': 'test'}, typedict={'a': {'dtype': int, 'opt': False}, 'b': {'dtype': str, 'opt': False}})
    # test optional
    config(target={'a': 5, 'b': 'True'}, default={'a': 1, 'b': 'test'}, typedict={'a': {'dtype': int, 'opt': False}, 'b': {'dtype': str, 'opt': True}})
    config(target={'a': 5}, default={'a': 1, 'b': 'test'}, typedict={'a': {'dtype': int, 'opt': False}, 'b': {'dtype': str, 'opt': True}})
    # test in default not in typecheck
    config(target={'a': 5}, default={'a': 1, 'b': 'test'}, typedict={'a': {'dtype': int, 'opt': False}})
    # test type error
    config(target={'a': 5, 'b': True}, default={'a': 1, 'b': 'test'}, typedict={'a': {'dtype': int, 'opt': False}, 'b': {'dtype': str, 'opt': True}})
    # test key in default not in target
    config(target={'a': 5, 'b': 'True'}, default={'a': 1, 'b': 'test'}, typedict={'a': {'dtype': int, 'opt': False}, 'b': {'dtype': str, 'opt': False}})
    """
    supported_files = {'py', 'pkl', 'pickle', 'pyx'}

    def __init__(self,
                 default_config_file: str = '',
                 target_config_file: str = '',
                 cwd: str = '',
                 default: dict = {},
                 typedict: dict = {},
                 target: dict = {}):
        """Dunder."""
        self.__time = datetime.now()
        self.__cwd = getcwd() if not cwd else cwd
        self.__target_config_file = target_config_file
        self.__default_config_file = default_config_file

        # read in default typed dictionary
        if not typedict:
            raise RuntimeError("""Default typed configuration unspecified...
        Please specify either an input file or a default configuration dictionary.""")  # noqa
        else:
            self.__typedict = typedict

        self.__verify_has_keys_inner(self.__typedict, example_typedict['keys'])  # noqa
        self.__required = set([key for key in self.__typedict if not self.__typedict[key]['opt']])

        # read in default configuration
        if not default and not default_config_file:
            raise RuntimeError("""Default configuration unspecified...
        Please specify either an input file or a default configuration dictionary.""")  # noqa
        elif default_config_file:
            # read in file
            default = self.__read(default_config_file)

        if self.__typedict.keys() != default.keys():
            raise KeyError("""Default configuration and type checker must match keys.""")

        for key, value in self.__typedict.items():
            if not self.__verify_has_val_type(key, default, self.__typedict):
                raise TypeError("""Default configuration doesn't meet type requirements specified.""")
        self.__default = {**default}

        # read in target configuration
        if not target and target_config_file:
            # read in file
            target = {**self.__read(target_config_file)}
        elif not target:
            target = {}
        else:
            target = {**target}

        for key, value in target.items():
            if not self.__verify_has_val_type(key, target, self.__typedict):
                raise TypeError("""Target configuration doesn't meet type requirements specified.""")
                del target[key]
        self.__target = {**self.__default, **target}

    def __call__(self, **config):
        for key, value in config.items():
            if not self.__verify_has_val_type(key, config, self.__typedict, val=value):
                warnings.warn(str(TypeError("""Target configuration doesn't meet type requirements specified. Using fallback.""")))
                del config[key]
        self.__target = {**self.__target, **config}
        return self.__target.items()

    def refresh(self):
        if not self.__target_config_file:
            warnings.warn(str(ConfigFileError('Configuration file not specified. Unable to refresh')))
            return
        untested = self.__read(self.__target_config_file)
        self.__verify_has_keys_outer(untested, self.__required)
        for key, value in untested.items():
            if not self.__verify_has_val_type(key, untested, self.__typedict):
                continue
            else:
                self.__target[key] = value

    @classmethod
    def __read(cls, inputfile: str):
        ext = inputfile.split('.')[-1]
        if ext not in cls.supported_files:
            raise ConfigFileError('Configuration file not supported. Must be: ' + ', '.join(cls.supported_files))  # noqa

        if ext in {'py', 'pyx'}:
            try:
                if __version__ >= 3.5:
                    spec = importlib.util.spec_from_file_location("config", inputfile)
                    cf = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(cf)
            except Exception as e:
                raise ConfigFileError(e)
        elif ext in {'pkl', 'pickle'}:
            with open(inputfile, 'rb') as f:
                cf = pickle.load(f)
        return _base_config_object(cf).to_dict()

    @staticmethod
    def __verify_has_val_type(key: str, target: dict, check: dict, val = None,):
        if key not in check:
            return True
        if val is not None:
            if not isinstance(val, check[key]['dtype']):
                warnings.warn(str(TypeError("""Value doesn't have correct type for key: """ + key)))
                return False
        if key not in target:
            raise KeyError('Key not found in target dictionary: ' + key)
        elif not isinstance(target[key], check[key]['dtype']):
            warnings.warn(str(TypeError("""Target doesn't have correct type for key: """ + key)))
            return False
        return True

    @staticmethod
    def __verify_has_keys_outer(default: dict, keys: set):
        if not all([key in default for key in keys]):
            warnings.warn(str(KeyError('Key is not found in target')))
            return False
        return True

    @classmethod
    def __verify_has_keys_inner(cls, default: dict, keys: set):
        for inner in default:
            return cls.__verify_has_keys_outer(default[inner], keys)
        return True

    def config(self):
        return self.__target

    def __getitem__(self, key: str):
        return self.__target[key]

    def __setitem__(self, key: str, val):
        if self.__verify_has_val_type(key, self.__target, self.__typedict, val=val):
            self.__target[key] = val

    def __len__(self):
        return len(self.__target)

    def __iter__(self):
        """Return item iterator."""
        return self.__target.items()

    def items(self):
        """Return item iterator."""
        return self.__target.items()

    def keys(self):
        """Return dict_key."""
        return self.__target.keys()

    def vals(self):
        """Return value iterator."""
        return self.__target.values()

    def save(self, filename: str):
        """Return value iterator."""
        ext = filename.split('.')[-1]
        filename = '.'.join(filename.split('.'[:-1]))
        filename = filename + '.' + ext if ext in ['pkl', 'pickle'] else filename + '.pickle'
        try:
            with open(filename, 'rb') as f:
                pickle.dump(f, self.__target)
        except Exception as e:
            print('Unable to save file.' + e)


class ImproperParameterType(Exception):
    pass


class NonOptionalParameter(Exception):
    pass


class ConfigFileError(Exception):
    pass


class InstantiationError(Exception):
    pass

# end of code

# end of file
