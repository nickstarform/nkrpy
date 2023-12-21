"""."""
# internal modules
import sys
from os import getcwd
import importlib
from datetime import datetime
import pickle
import warnings
exit = sys.exit

# external modules

# relative modules
from nkrpy.misc.errors import ConfigError
from nkrpy.misc.decorators import deprecated
from nkrpy.misc import Format
FAIL, RESET, HEADER = Format('FAIL'), Format('RESET'), Format('HEADER')
from nkrpy.misc.functions import typecheck, flatten_dict, deep_get_single, deep_set

# global attributes
__all__ = ['Config', 'verify_param', 'verify_dir', 'load_cfg']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = float(sys.version[:3])
__cwd__ = getcwd()
example_typedict = {
    '__CFG_keys': {'type': str, 'opt': True, 'requires': '', 'default': None, 'doc': ''},
}


def verify_dir(name, create=False):
    """Verify/create a directory."""
    if not os.path.isdir(name):
        if create:
            os.makedirs(name)
        else:
            return False
    return True


def verify_param(target, comparison):
    """Verify parameters within file against a template."""
    """All of comparison must be in target, not vice versa."""
    if isinstance(target, str):
        target = load_cfg(target)
    if isinstance(comparison, str):
        comparison = load_cfg(comparison)
    if isinstance(target, dict):
        _t = [x for x in target.keys() if '__' not in x]
    elif typecheck(target):
        _t = [x for x in target if '__' not in x]
    else:
        _t = [x for x in dir(target) if '__' not in x]
    if isinstance(comparison, dict):
        _c = [x for x in comparison.keys() if '__' not in x]
    elif typecheck(comparison):
        _c = [x for x in comparison if '__' not in x]
    else:
        _c = [x for x in dir(comparison) if '__' not in x]

    for x in _c:
        if x not in _t:
            raise ConfigError(f'{FAIL}Parameters not found.{RESET}',  # noqa
                              (target, comparison))
    else:
        return True


def load_cfg(fname):
    """Load configuration file as module."""
    if '/' not in fname:
        fname = __cwd__ + '/' + fname
    try:
        if __version__ >= 3.5:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", fname)
            cf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cf)
        elif __version__ >= 3.3:
            from importlib.machinery import SourceFileLoader
            cf = SourceFileLoader("config", fname).load_module()
        elif __version__ <= 3.0:
            import imp
            cf = imp.load_source('config', fname)
    except Exception:
        print(f'{FAIL}Failed.{RESET}Cannot find file <{HEADER}{fname}{RESET}> or the fallback config.py>')  # noqa
        print(f'Or invalid line found in file. Try using import <{HEADER}{fname[:-3]}{RESET}> yourself')  # noqa
        exit(1)
    return cf


@deprecated
def load_variables(mod):
    """Given a module, load attributes directly to global."""
    for k in dir(mod):
        if '__' not in k:
            globals()[k] = getattr(mod, k)


@deprecated
def load_func(my_func, *args, **kwargs):
    """Access functions."""
    return locals()[my_func](*args, **kwargs)


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


class ConfigClass(object):
    pass


# main class object for manipulating configuration files
class Config(ConfigClass):
    """Python Configuration generator/verifier.

    Currently this only handles dictionaries, python, pickle files.

    Periods ('.') are NOT supported in the keys of the configuration file.

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
    Default is the master holder.
    * If key in Default and not optional, it must be in target and pass type check.
    * If key in typedict and optional, if it appears in config or target, it must pass type check.
    * If key not in typedict, then auto pass.
    * If key in default but not in target, then default values are inherited, unless above conditions hold.
    * Getting/Setting can be granted by either indexing or attribute getters config[key] or config.key
    * Getting/Setting are non yielding
    * config is iterable and yields dict_iterable
    * Ignored all keys that begin with '__CFG'

    Example
    =======
    # test standard
    config(target={'a': 5, 'b': 'True'}, default={'a': {'type': int, 'opt': False, 'default': 1, 'requires': None}, 'b': {'type': str, 'opt': False, 'default': 'test', 'requires': None}})

    Can Pass a previous ConfigClass object into 'cfg' argument if already made to check.
    """

    __supported_files = {'py', 'pkl', 'pickle', 'pyx'}

    def __init__(self, cfg: ConfigClass = None,
                 default_config_file: str = '',
                 target_config_file: str = '',
                 cwd: str = '',
                 default: dict = {},
                 target: dict = {},
                 strict: bool=False,
                 level: int = 0,
                 template: bool = True):
        """Dunder.

        Parameters
        ==========
        cfg: ConfigClass
            Default: None, If you want to create a new ConfigClass by passing another
        default_config_file: str
            Read from a configuration file
        target_config_file: str
            Read from a target configuration file
        cwd: str
            The working directory if file reading/writing is needed
        default: dict
            Use an input config dict as the default
        target: dict
            Use an input dict as the dict to check against
        strict: bool
            To strictly check type.
        template: bool
            If the given default configuration is the same for all parameters.

        """
        if cfg is not None and isinstance(cfg, ConfigClass):
            target_config_file = cfg['target_config_file'] if cfg['target_config_file'] else target_config_file
            default_config_file = cfg['default_config_file'] if cfg['default_config_file'] else default_config_file
            cwd = cfg['cwd'] if cfg['cwd'] else cwd
            default = cfg['default'] if cfg['default'] else default
            target = cfg['target'] if cfg['target'] else target
            strict = cfg['strict'] + strict
        self.__time = datetime.now()
        self.__cwd = getcwd() if not cwd else cwd
        self.__target_config_file = target_config_file
        self.__default_config_file = default_config_file
        self.__strict = strict
        self.__level = 0

        assert level >= 0

        # read in default configuration
        if not default and not default_config_file:
            # no default provided
            default = {**example_typedict}
        elif not default and default_config_file:
            # read in file
            default = self.__read(default_config_file)
        self.__default = {**default}

        _ = self.__verify_has_keys_level(self.__default, set(example_typedict['__CFG_keys'].keys()))  # noqa
        self.__required = set([key for key in self.__default if not self.__default[key]['opt']])
        #print('Required: ', self.__required)

        self.__level = level
        # read in target configuration
        if not target and target_config_file:
            # read in file
            target = {**self.__read(target_config_file)}
        elif not target:
            target = {}
        else:
            target = {**target}

        if self.__default != example_typedict:
            self.__flattened_target = self.__verify_has_keys_level(target, self.__required, level=self.__level)
            #print('Flattened:', self.__flattened_target)
            for keys in self.__flattened_target:
                if not self.__verify_has_val_type(keys, target, self.__default, strict=self.__strict, level=self.__level):
                    raise TypeError("""Target configuration doesn't meet type requirements specified.""")
                    del target[key]
        self.__target = {**self.__get_default_vals(), **target}

    def __call__(self, level: int = None, **config):
        """Override the previous configuration with a new one."""
        self.__level = level if level is not None else self.__level
        flat = flatten_dict(config, lift=lambda x: (x,), only_keys=True)
        for keys in flat:
            if not self.__verify_has_val_type(keys, config, self.__default, val=value, strict=self.__strict, level=self.__level):
                warnings.warn(str(TypeError("""Target configuration doesn't meet type requirements specified. Using fallback.""")), stacklevel=1)
                del config[key]
        self.__flattened_target = flat
        self.__target = {**self.__target, **config}

    def refresh(self):
        """Refresh current configuration.

        Order of operations is either re-reads from target file and applies
            default configuration or re-applies default configuration to the
            current configuration (usefule for if values/defaults are changed
            OTF.
        """
        if self.__target_config_file:
            untested = self.__read(self.__target_config_file)
            inplace = False
        else:
            untested = self.__target
            inplace = True
        flattened_untested = self.__verify_has_keys_level(untested, self.__required, level=self.__level) # check
        for keys in flattened_untested:
            if not self.__verify_has_val_type(key, untested, self.__default, strict=self.__strict, level=self.__level):
                continue
            elif not inplace:
                self.__target[key] = value

    @classmethod
    def __read(cls, inputfile: str):
        """Generalized backend reader.

        This only supports a few datatypes that are hard coded. Will attempt
            to read from these files, assign to an object and then convert
            to a dictionary

        Parameters
        ----------
        inputfile: str
            The file to read from

        Returns
        -------
        dict
            dictionary of the input file module
        """
        ext = inputfile.split('.')[-1]
        if ext not in cls.__supported_files:
            raise ConfigFileError('Configuration file not supported. Must be: ' + ', '.join(cls.__supported_files))  # noqa

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

    def __get_default_vals(self):
        """Backend for retreiving default values."""
        ret = {}
        for k, v in self.__default.items():
            if k.startswith('__CFG'):
                continue
            ret[k] = v['default']
        return ret

    @staticmethod
    def __verify_has_val_type(keys: tuple, target: dict, check: dict, val=None, strict: bool = False, level: int=0):
        """Backend for enforcing value typing."""
        key = keys[-1]
        if key not in check:
            # if an extra key is provided we don't check
            return True
        if val is not None:
            # if a key, value is provided, lets check it
            if not isinstance(val, check[key]['type']):
                if not strict:
                    try:
                        _ = check[key]['type'](val)
                        return True
                    except:
                        pass
                warnings.warn(str(TypeError("""Value doesn't have correct type for key: """ + key)), stacklevel=1)
                return False
        if len(keys) > 1:
            _target = deep_get_single(target, '.'.join(keys[:-1]))
        else:
            _target = target
        if _target is None or key not in _target:
            # if you gave a key not in the target dictionary (nothing to check)
            raise KeyError('Key not found in target dictionary: ' + key)
            return False
        elif strict and not isinstance(_target[key], check[key]['type']):
            warnings.warn(str(TypeError("""Target doesn't have correct type for key: """ + key)), stacklevel=1)
            return False
        if check[key]['requires'] is not None and deep_get_single(target, check[key]['requires']) is None:
            warnings.warn(str(KeyRequirementFailed(f"""The key '{key}' has a requirement {check[key]['requires']} be specified.""")), stacklevel=1)
            return False
        return True

    @staticmethod
    def __verify_has_keys_level(target: dict, required_keys: set, level: int = 0):
        """Backend to verify that every key in keys is found in target.

        Parameters
        ----------
        default: dict
            The dictionary to check against.
        keys: set
            The list of unique values to check from

        Returns
        -------
        bool
            True if passed else False
        """
        __flattened_target = flatten_dict(target, lift=lambda x: (x,), only_keys=True)
        for __target_key in __flattened_target:
            if len(__target_key) - 1 != level:
                continue
            __target_key = __target_key[:level]
            target = deep_get_single(target, __target_key)
            if target is None:
                continue
            target_keys = set(target)
            if len(required_keys - target_keys) > 0:
                warnings.warn(str(KeyError(f'''Keys: `{required_keys}` are not found in target: {target.keys()}

                    These keys are missing: {required_keys - target_keys}''')), stacklevel=2)
                print(f'''Keys: `{required_keys}` are not found in target: {target.keys()}

                    These keys are missing: {required_keys - target_keys}''')
                exit()
        return __flattened_target

    def dict(self):
        """Return dict of target."""
        return self.__target

    def __getitem__(self, key: str):
        """Dunder."""
        return deep_get_single(self.__target, key)

    def __setitem__(self, key: str, val):
        """Dunder."""
        if self.__verify_has_val_type(key, self.__target, self.__default, val=val, level=self.__level):
            deep_set(self.__target, key, val)

    def __len__(self):
        """Dunder."""
        return len(self.__target)

    def __iter__(self):
        """Dunder."""
        return iter(self.__target)

    def set_default_type(self, key: str, dtype):
        """Set new default type for a given key.

        DOESN'T FORCE A REFRESH.
        """
        if key not in self.__default:
            raise KeyNotInConfig(f'Key not found: {key}')
            return
        self.__default[key]['type'] = dtype

    def set_default_value(self, key: str, val):
        """Set new default value for a given key.

        DOESN'T RORCE A REFRESH.
        """
        if key not in self.__default:
            raise KeyNotInConfig(f'Key not found: {key}')
            return
        self.__default[key]['default'] = dtype

    def set_default_doc(self, key: str, *, doc: str):
        """Set new docstring for a given key."""
        if key not in self.__default:
            raise KeyNotInConfig(f'Key not found: {key}')
            return
        self.__default[key]['doc'] = doc

    def doc(self, key):
        """Generate doc of a given key."""
        if key not in self.__default:
            raise KeyNotInConfig(f'Key not found: {key}')
            return
        return f"""{self.__default[key]['doc']}. Requires: <{self.__default[key]['requires']}> Typed: <{self.__default[key]['type'].__name__}> Default: <{self.__default[key]['default']}>"""

    def docs(self):
        """Generate full docs."""
        ret = []
        for k in self.__default:
            if k.startswith('__CFG'):
                continue
            v = self.doc(k)
            ret.append(f'{k}: {v}')
        return ret

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

    @classmethod
    def supported_files(cls):
        """Return supported files."""
        return cls.__supported_files


class KeyRequirementFailed(Exception):
    pass


class KeyNotInConfig(Exception):
    pass


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
