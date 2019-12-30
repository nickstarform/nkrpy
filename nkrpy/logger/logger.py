# internal modules
from datetime import datetime as datetime__datetime
from time import sleep as time__sleep
from time import time as time__time
from atexit import register as atexit__register
from signal import signal as signal__signal
from signal import SIGTERM as signal__SIGTERM
from sys import exit as sys__exit

# external modules

# relative modules
from .. import colours


# global parameters
datetime__today = datetime__datetime.today
__all__ = (  # noqa
    "Logger",
    "decorator",
    "waiting",
    "pyinput",
    "debug",
    "message",
    "header1",
    "header2",
    "success",
    "failure",
    "warn",
    "enable_colour",
    "disable_colour",
    "setup",
    "set_verbosity",
    "get_verbosity",
    "set_logfile_base",
    "switch_logfile",
    "new_logfile",
    "get_logfile",
    "teardown",
)
__doc__ = """
    Description
    -----------
    A very (read overly) generalized logger. Instantiated as a singleton,
    this module can handle nearly any case of logging output desired. It
    is codified to try and be threadsafe and cleanly deconstruct/close the
    necessary files as terminated.

    Definitions
    -----------
    verbosity
        An integer from 1 - 5 that detail the amount of output to have. With
        5 being a higher verbosity.
    structure
        A string that will handle deliniation between various logger outputs.
        A string of `.>` with standard verbosity set, will output
            ```
            SUCCESS
            FAILURE
            .> HEADER1
            .>.> HEADER2
            .>.>.> MESSAGE
            .>.> DEBUG
            ```

    Usage
    -----
    ```
    from nkrpy import logger

    logger.setup()  # inputs described below

    logger.warn('This is a test message')

    @logger.decorator(style = 'debug', verbosity=5)
    def custom_function(*args, **kwargs):
        return args, kwargs

    print(custom_function(1, ten = 10))
    # .>.> Calling <{function.__name}> with  params: {args}, {kwargs}'
    # 1, {'ten': 10}
    # .>.> Called <{function.__name}> with  params: {args}, {kwargs}. Result: {result}'
    ```

    Setting up the Logger
    ---------------------
"""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class Logger(object):

    __instance = None

    def __new__(cls):
        cls.__doc__ = __doc__ + cls.setup.__doc__
        if cls.__instance is None:
            instance = object.__new__(cls)
            cls.__instance = instance
        return cls.__instance

    def __init__(self):
        """Doesn't actually build the instance.
        Need to separately call the __setup."""
        self.__setup_status = False
        pass

    def __guarantee_setup(function):
        """A specific wrapper to handle if
        the overall class was setup properly."""
        def wrapper(*args, **kwargs):
            if not args[0].__setup_status:
                args[0].setup()
                args[0].warn("The logger command wasn't properly setup," +
                             f"fallback values were used. {function}")
            result = function(*args, **kwargs)
            return result
        return wrapper

    def setup(self, append: bool = False,
              verbosity: int = 2, use_colour: bool = True,
              structure_string: str = ".>", add_timestamp: bool = True,
              logfile: str = None, suppress_terminal: bool = False):
        """Set the parameters for the Messenger class.

        Parameters
        ----------
        append: bool
            If False, will write to new files based off of timestamp.
            If True, will override the current files w/o timestamp
        verbosity: int
            Set the verbosity 1-5, 5 being higher
        use_colour: bool
            Turn on colour outputs
        structure_string: str
            Use a specified structure for logging.
                will multiply this string by verbosity level to
                determine left justification
        add_timestamp: bool
            Add a timestamp to the messages
        logfile: str
            This will serve as the base name for
                the logfile. If unset will just
                output to the terminal.
        suppress_terminal: bool
            If toggled will suppress terminal output
        """
        saved_args = locals()
        self.__setup_status = True
        self.__dict__.update(saved_args)
        self.set_verbosity(verbosity)

        # specifying colour options
        if use_colour:
            self.enable_colour()
        else:
            self.disable_colour()

        ctime = str(time__time()).split('.')[0]
        self.__write_status = 'a' if self.append else 'w'
        if not self.append and self.logfile is not None:
            self.logs = {'basename': self.logfile + f'-{ctime}'}
        else:
            self.logs = {'basename': self.logfile}
        self.logfile = ''
        msg = f'The logger has been setup with parameters ' +\
            f': {saved_args}'
        self.success(msg)

    def _get_structure_string(self, level):
        """Returns the string of the message with the specified level
        Which is dependent on the verbosity
        """
        string = self.structure_string * level
        return string

    def _get_time_string(self):
        """Returns the detailed datetime for extreme debugging
        """
        string = ''
        if self.add_timestamp:
            string = '[{}] '.format(datetime__today())
        return string

    def _make_full_msg(self, msg: str, verb_level: int):
        """Construct the full string that carries the message.

        With the specified verbosity parameters.
        """
        struct_string = self._get_structure_string(verb_level)
        time_string = self._get_time_string()
        return time_string + struct_string + msg

    def __teardown(self):
        """Close all opened files."""
        if not hasattr(self, 'logs'):
            return
        for key in self.logs:
            if key == 'basename':
                continue
            if not self.logs[key].closed:
                self.logs[key].close()

    def __sigHandler(self):
        """Handle Ctl-C or sigints."""
        self.__teardown()
        sys__exit(0)

    def _write(self, cmod: str, msg: str, fname: str = None):
        """Wrapper to write to terminal and file.

        Write the message to the file and print
        it to the terminal if it is wanted.
        """
        if not self.suppress_terminal:
            print("{}{}{}".format(cmod, msg, self._RST_))

        if self.logs['basename'] is None:
            return
        if fname is not None:
            self.logfile = '-'.join((self.logs['basename'], fname)) + '.log'
        elif self.logfile == '':
            self.logfile = '-'.join((self.logs['basename'], '1')) + '.log'

        if self.logfile in self.logs:
            if self.logs[self.logfile].closed:
                self.logs[self.logfile] = open(self.logfile,
                                               self.__write_status)
        else:
            self.logs[self.logfile] = open(self.logfile, self.__write_status)
        self.logs[self.logfile].write(msg + '\n')

    def __resolve_style(self, style: str):
        if hasattr(self, style):
            return getattr(self, style)
        return getattr(self, 'message')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    The following commands are the ones
    to use when calling the logger
    will handle writing to the log
    file and to the terminal
    """

    def decorator(self, style: str = 'warn', verbosity: int = 2):
        print(1)
        def real_decorator(function):
            print(10)
            def wrapper(*args, **kwargs):
                print(10)
                msg = f'Calling <{function.__name__}> with ' +\
                    f'params: {args}, {kwargs}'
                (self.__resolve_style(style))(msg, verbosity)
                result = function(*args, **kwargs)
                msg = f'Called <{function.__name__}> with ' +\
                    f'params: {args}, {kwargs}. Result: {result}'
                (self.__resolve_style(style))(msg, verbosity)
                return result
            return wrapper
        return real_decorator

    @__guarantee_setup
    def set_logfile_base(self, *, basename: str):
        """Set the basename for the logfile."""
        self.logs['basename'] = basename

    @__guarantee_setup
    def new_logfile(self, *, end_str: str = None):
        """Set the basename for the logfile."""
        if end_str is None:
            end_str = str(datetime__today())
        self.logfile = '-'.join((self.logs['basename'], end_str)) + '.log'
        self.logs[self.logfile] = open(self.logfile, self.__write_status)

    @__guarantee_setup
    def switch_logfile(self, *, fname: str):
        """Set the basename for the logfile."""
        fname = fname.strip('.log') + '.log'
        if fname in self.logs:
            self.logfile = fname

    @__guarantee_setup
    def get_logfile(self):
        """Return the verbosity level of the class.
        """
        return self.logfile

    @__guarantee_setup
    def set_verbosity(self, verbosity):
        """Set the verbosity level for the class.
        """
        self.verbosity = verbosity

    @__guarantee_setup
    def get_verbosity(self):
        """Return the verbosity level of the class.
        """
        return self.verbosity

    @__guarantee_setup
    def disable_colour(self):
        """Turn off all colour formatting.
        """
        for c in colours.__dict__:
            setattr(self, c, '')

    @__guarantee_setup
    def enable_colour(self):
        """Enable all colour formatting.
        """
        self.__dict__ = {**colours.__dict__, **self.__dict__}

    def __general_messager(self, colour: str, msg: str, verb_level: int):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(colour, full_msg)

    @__guarantee_setup
    def warn(self, msg, verb_level=2):
        self.__general_messager(self.WARNING, msg, verb_level)

    @__guarantee_setup
    def header1(self, msg, verb_level=0):
        self.__general_messager(self.HEADER, msg, verb_level)

    @__guarantee_setup
    def header2(self, msg, verb_level=1):
        self.__general_messager(self.Cyan, msg, verb_level)

    @__guarantee_setup
    def success(self, msg, verb_level=1):
        self.__general_messager(self.OKGREEN, msg, verb_level)

    @__guarantee_setup
    def failure(self, msg, verb_level=0):
        self.__general_messager(self.FAIL, msg, verb_level)

    @__guarantee_setup
    def message(self, msg, verb_level=2):
        self.__general_messager(self.OKBLUE, msg, verb_level)

    @__guarantee_setup
    def debug(self, msg, verb_level: int = 4):
        self.__general_messager(self.Yellow, msg, verb_level)

    @__guarantee_setup
    def pyinput(self, message: str = '', verb_level: int = 0):

        total_Message = "Please input {}: ".format(message)
        out = input(total_Message)
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(total_Message, verb_level)
            self._write(self.Yellow, full_msg, False)
        return out

    @__guarantee_setup
    def waiting(self, auto: bool, seconds: int = 10, verb_level: int = 0):
        if not auto:
            self.pyinput('[RET] to continue or CTRL+C to escape')
        elif verb_level <= self.verbosity:
            self.warn('Will continue in {}s. CTRL+C to escape'.format(seconds))
            time__sleep(seconds)

    def teardown(self):
        self.__teardown()

    def sigHandler(self):
        self.__sigHandler()


# Yielding singleton to module
_logger = Logger()

__doc__ += _logger.setup.__doc__
_logger.__doc__ = __doc__
Logger.__doc__ = __doc__

for func in __all__:
    if func == 'Logger':
        continue
    globals()[func] = getattr(_logger, func)

# handling exit
atexit__register(_logger.teardown)
signal__signal(signal__SIGTERM, _logger.sigHandler)

# end of code
