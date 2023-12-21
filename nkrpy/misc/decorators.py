"""Random decorators for fancy functions."""

# internal modules
import warnings
import functools
import sys
import cProfile
import pstats
import io
import time
from functools import wraps

# external modules

# relative modules
from .errors import ArgumentError

# global attributes
__all__ = ['deprecated', 'call_counter', 'timeit', 'checker',
           'ignore_deprecation_warnings', 'alias', 'aliasClass', 'argCase']
__doc__ = """Generalized decorators for common usage."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def argCase(case: str):
    def _case(s: str, ul: str):
        ul = ul.lower()
        if ul == 'lower':
            return s.lower()
        return s.upper()

    def decorator(function):
        """Force all arguments to lower/upper case."""
        def wrapper(*args, **kwargs):
            args = list(args)
            for i, a in enumerate(args):
                if isinstance(a, str):
                    args[i] = _case(a, case)
            for k, v in kwargs.items():
                if isinstance(v, str):
                    kwargs[k] = _case(v, case)
            return function(*args, **kwargs)
        return wrapper
    return decorator


class _MethodDecoratorAdaptor(object):
    def __init__(self, decorator, func):
        self.decorator = decorator
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.decorator(self.func)(*args, **kwargs)
    def __get__(self, instance, owner):
        return self.decorator(self.func.__get__(instance, owner))

def generaldecorator(decorator):
    """Allows you to use the same decorator on methods and functions,
    hiding the self argument from the decorator."""
    def adapt(func):
        return _MethodDecoratorAdaptor(decorator, func)
    return adapt


def allowed(*allowed_methods):
    @generaldecorator
    def wrapper(func):
        def wrapped(request):
            if request not in allowed_methods:
                raise ValueError("Invalid method %s" % request)
            return func(request)
        return wrapped
    return wrapper



def _construct_master_group():
    gl = globals()
    name = ''
    if '__filename__' in gl:
        name = f'_{gl["__filename__"]}'
    name = f'__nkrpy_groups{name}'
    if name not in gl:
        gl[name] = {False: {}, True: {}}
    return gl[name]


def _get_group_num(exclusive: bool = False):
    gl_o = _construct_master_group()
    gl = gl_o[exclusive]
    if len(gl) == 0:
        gl[0] = ()
        return 0, gl_o, exclusive
    latest = max(gl.keys())
    gl[latest + 1] = ()
    return latest + 1, gl_o, exclusive


def normalize_args_kwargs(function, *args, **kwargs):
    nk = {**kwargs}
    nk.update(zip(function.__code__.co_varnames, args))
    return nk

_groupnums = []
def group(*grouped_args: list, exclusive: bool=False):
    """Decorator factory forms groups of arguments.

    This can take positional or kwargs. However, the decorator input is the variable names.

    If not invoked in a file, this places into global variables __nkrpy_groups[0...]
    If invoked in a file, this places into global variables __nkrpy_groups_filename[0...]
    
    Usage
    -----
    @group(['argument1_group0', 'argument2_group0'], ['argument1_group1', 'argument2_group1'])
    def func(argument1_group0, argument2_group0, argument1_group1 = None, argument2_group1=None):
        pass
    """
    for groups in grouped_args:
        num, gl, e = _get_group_num(exclusive)
        gl[e][num] = tuple(groups)
        _groupnums.append([e, num])
    def decorator(function):
        def inner(*args, **kwargs):
            if function.__name__ not in ('group', 'inner'):
                nk = normalize_args_kwargs(function, *args, **kwargs)
                for e, num in _groupnums:
                    groups = gl[e][num]
                    set_args = set(groups)
                    _t = set_args - nk.keys()
                    _t = set_args - _t
                    gl[e][num] = tuple(_t)
                    if e and any([nk[k] is not None for k in gl[e][num]]):
                        raise ArgumentError()
            return function(*args, **kwargs)
        return inner
    return decorator

def exclusive(*grouped_args: list):
    return 
""" for testing
del gl['__nkrpy_groups_notafile']
print('Before Decorate')
@group(['pos1', 'kpos1', 'notinkwargs'])
@group(['pos2', 'kpos2'])
def cube(pos1, pos2, kpos1 = None, kpos2 = None, ungrouped = None):
    pass

print('After Decorate')
cube(1,2,3,4,5)
print('After call')
"""


def _counter():
    i = 0
    while True:
        yield i
        i += 1


def counter():
    return next(_counter())



class aliasClass(object):
    """Aliasing methods.

    Alias class that can be used as a decorator for making methods callable
    through other names (or "aliases").
    Note: This decorator must be used inside an @aliased -decorated class.
    For example, if you want to make the method shout() be also callable as
    yell() and scream(), you can use alias like this:

        @alias('yell', 'scream')
        def shout(message):
            # ....
    """

    def __init__(self, *aliases):
        """Constructor."""
        self.aliases = set(aliases)

    def __call__(self, f):
        """Dunder.

        Method call wrapper. As this decorator has arguments, this method will
        only be called once as a part of the decoration process, receiving only
        one argument: the decorated function ('f'). As a result of this kind of
        decorator, this method must return the callable that will wrap the
        decorated function.
        """
        f._aliases = self.aliases
        return f


def alias(aliased_class):
    """Actual alias setter.

    Decorator function that *must* be used in combination with @alias
    decorator. This class will make the magic happen!
    @aliased classes will have their aliased method (via @alias) actually
    aliased.
    This method simply iterates over the member attributes of 'aliased_class'
    seeking for those which have an '_aliases' attribute and then defines new
    members in the class using those aliases as mere pointer functions to the
    original ones.

    Usage:
        @aliasclass
        class MyClass(object):
            @alias('coolMethod', 'myKinkyMethod')
            def boring_method():
                # ...

        i = MyClass()
        i.coolMethod() # equivalent to i.myKinkyMethod() and i.boring_method()
    """
    original_methods = aliased_class.__dict__.copy()
    for name, method in original_methods.iteritems():
        if hasattr(method, '_aliases'):
            # Add the aliases for 'method', but don't override any
            # previously-defined attribute of 'aliased_class'
            for alias in method._aliases - set(original_methods):
                setattr(aliased_class, alias, method)
    return aliased_class


def call_counter(func):
    """Count number of function calls."""
    def helper(x):
        helper.calls += 1
        return func(x)
    helper.calls = 0

    return helper


def timeit(method):
    """Time and profile functions."""
    def timed(*args, **kw):
        pr = cProfile.Profile()
        # ... do something ...
        ts = time.time()
        pr.enable()
        result = method(*args, **kw)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print('Stats on:', method.__name__.upper(), s.getvalue())
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def checker(f):
    """Output all inputs."""
    def wrap(*args, **kwargs):
        print(f'args: {args}')
        print(f'kwargs: {kwargs}')
        return f(*args, **kwargs)
    return wrap


def debug(f):
    """Embeds IPython console for debug."""
    from IPython import embed
    def wrap(*args, **kwargs):
        embed()
        _t = f(*args, **kwargs)
        embed()
        return _t
    return wrap


#
# -------------------/Smart deprecation warnings\-------------------#
#
def discontinued(func):
    """Mark discontinued methods.

    as fully discontinued/jetisoned. It will result in a warning being emitted
    when the function is used.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to dicontinued function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return
    return new_func


def deprecated(func):
    """Mark deprecated methods.

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Usage
    -----
    @deprecated
    def my_func(): pass

    @other_decorators_must_be_upper
    @deprecated
    def my_func(): pass

    @deprecated
    def my_func2(): pass
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


#
# -------------------/Ignoredeprecation warnings\-------------------#
#
def ignore_deprecation_warnings(func):
    """Ignore depreciation.

    This is a decorator which can be used to ignore deprecation warnings
    occurring in a function.

    Usage
    -----
    @ignore_deprecation_warnings
    def some_function_raising_deprecation_warning():
        warnings.warn("This is a deprecationg warning.",
                    category=DeprecationWarning)

    class SomeClass:
        @ignore_deprecation_warnings
        def some_method_raising_deprecation_warning():
            warnings.warn("This is a deprecationg warning.",
                        category=DeprecationWarning)
    """
    def new_func(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


#
# -------------------/Type Enforcement\-------------------#
#
def accepts(*types, **kw):
    """Check decorated arguments are of the expected types.

    Parameters
    ----------
    types -- The expected types of the inputs to the decorated function.
             Must specify type for each parameter.
    kw    -- Optional specification of 'debug' level (this is the only valid
             keyword argument, no other should be given).
             debug = ( 0 | 1 | 2 )

    """
    if not kw:
        # default level: MEDIUM
        debug = 1
    else:
        debug = kw['debug']
    try:
        def decorator(f):
            def newf(*args):
                if debug == 0:
                    return f(*args)
                assert len(args) == len(types)
                argtypes = tuple(map(type, args))
                if argtypes != types:
                    msg = info(f.__name__, types, argtypes, 0)
                    if debug == 1:
                        print(sys.stderr, 'TypeWarning: ', msg)
                    elif debug == 2:
                        raise TypeError(msg)
                return f(*args)
            newf.__name__ = f.__name__
            return newf
        return decorator
    except KeyError as key:
        raise KeyError(key + "is not a valid keyword argument")
    except TypeError as msg:
        raise TypeError(msg)


def returns(ret_type, **kw):
    """Check decorated function's return is of the expected type.

    Parameters
    ----------
    ret_type -- The expected type of the decorated function's return value.
                Must specify type for each parameter.
    kw       -- Optional specification of 'debug' level (this is the only valid
                keyword argument, no other should be given).
                debug=(0 | 1 | 2)

    """
    try:
        if not kw:
            # default level: MEDIUM
            debug = 1
        else:
            debug = kw['debug']

        def decorator(f):
            def newf(*args):
                result = f(*args)
                if debug == 0:
                    return result
                res_type = type(result)
                if res_type != ret_type:
                    msg = info(f.__name__, (ret_type,), (res_type,), 1)
                    if debug == 1:
                        print(sys.stderr, 'TypeWarning: ', msg)
                    elif debug == 2:
                        raise TypeError(msg)
                return result
            newf.__name__ = f.__name__
            return newf
        return decorator
    except KeyError as key:
        raise KeyError(key + "is not a valid keyword argument")
    except TypeError as msg:
        raise TypeError(msg)


_t = """
Usage
-----
    >>> NONE, MEDIUM, STRONG = 0, 1, 2
    >>>
    >>> @accepts(int, int, int)
    ... @returns(float)
    ... def average(x, y, z):
    ...     return (x + y + z) / 2
    ...
    >>> average(5.5, 10, 15.0)
    TypeWarning:  'average' method accepts (int, int, int), but was given
    (float, int, float)
    15.25
    >>> average(5, 10, 15)
    TypeWarning:  'average' method returns (float), but result is (int)
    15

Needed to cast params as floats in function def (or simply divide by 2.0).

    >>> TYPE_CHECK = STRONG
    >>> @accepts(int, debug=TYPE_CHECK)
    ... @returns(int, debug=TYPE_CHECK)
    ... def fib(n):
    ...     if n in (0, 1): return n
    ...     return fib(n-1) + fib(n-2)
    ...
    >>> fib(5.3)
    Traceback (most recent call last):
      ...
    TypeError: 'fib' method accepts (int), but was given (float)
"""

returns.__doc__ += _t
accepts.__doc__ += _t


def info(fname, expected, actual, flag):
    """Return nicely formatted error/warning msg."""
    def format(types):
        return ', '.join([str(t).split("'")[1] for t in types])

    expected, actual = format(expected), format(actual)
    msg = "'{}' method ".format(fname)\
          + ("accepts", "returns")[flag] + " ({}), but ".format(expected)\
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg


def validate(func):
    """Validate the inputs that only a single flag is used."""
    def wrapper(*args, **kwargs):
        total = {}
        for x in args:
            total.update({counter(): x})
        if kwargs:
            total.update(**kwargs)
        res = [x for x in total.values() if isinstance(x, bool)]
        if (res.count(True) != 1):
            raise Exception('Incorrect flag configuration.' +
                            ' Please only toggle a single flag.')
        else:
            return func(*args, **kwargs)
    return wrapper


#
# -------------------/CGI Method wrapper\-------------------#
#
class CGImethod(object):
    """A CGI wrapper for a givenfunction."""

    """Usage examples
    @CGImethod("Hello with Decorator")
    def say_hello():
        print('<h1>Hello from CGI-Land</h1>')
    """
    def __init__(self, title):
        """Dunder."""
        self.title = title

    def __call__(self, fn):
        """Caller magic method."""
        def wrapped_fn(*args):
            print("Content-Type: text/html\n\n")
            print("<HTML>")
            print("<HEAD><TITLE>{}</TITLE></HEAD>".format(self.title))
            print("<BODY>")
            try:
                fn(*args)
            except Exception as e:
                print('\n')
                print(e)
            print()
            print("</BODY></HTML>")

        return wrapped_fn

# end of file
