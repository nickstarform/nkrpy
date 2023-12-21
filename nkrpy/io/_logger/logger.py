"""."""
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
from ...misc import Format
from ..._types import LoggerClass

# global parameters
datetime__today = datetime__datetime.today
__all__ = [  # noqa
    "Log",
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
    "example",
]
__doc__ = """Generalized logger.

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
    from nkrpy import Log as logger

    logger.setup(basefilename)  # inputs described below

    logger.warn('This is a test message')

    @logger.decorator(style = 'debug', verbosity=5)
    def custom_function(*args, **kwargs):
        return args, kwargs

    print(custom_function(1, ten = 10))
    # .>.> Calling <{function.__name}> with  params: {args}, {kwargs}'
    # 1, {'ten': 10}
    # .>.> Called <{function.__name}> with  params: {args}, {kwargs}. Result: {result}'  # noqa
    ```

    Setting up the Logger
    ---------------------
    """
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class Logger(LoggerClass):

    __instance = None

    def __new__(cls):
        """Dunder."""
        cls.__doc__ = __doc__ + cls.setup.__doc__
        if cls.__instance is None:
            instance = object.__new__(cls)
            cls.__instance = instance
        return cls.__instance

    def __init__(self):
        """Dunder.

        Doesn't actually build the instance.
        Need to separately call the __setup.
        """
        self.__setup_status = False
        pass

    def __call__(self, *args, **kwargs):
        self.setup(*args, **kwargs)

    def __guarantee_setup(function):
        """Guarantee proper setup."""
        def wrapper(*args, **kwargs):
            if not args[0].__setup_status:
                args[0].__instance = None
                args[0].setup()
                args[0].warn("The logger command wasn't properly setup," +
                             f"fallback values were used. {function}")
            result = function(*args, **kwargs)
            return result
        return wrapper

    def setup(self, logfile: str = None, append: bool = False,
              verbosity: int = 2, use_colour: bool = True,
              structure_string: str = "-", add_timestamp: bool = True, suppress_terminal: bool = False):
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
        if self.__setup_status:
            return
        saved_args = locals()
        self.__setup_status = True
        self.__dict__.update(saved_args)
        self.set_verbosity(verbosity)

        # specifying colour options
        self.enable_colour()
        if not use_colour:
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

    def __del__(self):
        try:
            self.__teardown()
        except:
            if self.verbosity > 2:
                print('Unable to close logger')



    def __bool__(self):
        return self.__instance is not None

    def _get_structure_string(self, level):
        """Return vebosity dependent string."""
        string = self.structure_string * level
        return string

    def _get_time_string(self):
        """Return detailed timedate."""
        string = ''
        if self.add_timestamp:
            string = '[{}] '.format(datetime__datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        return string

    def _make_full_msg(self, msg: str, verb_level: int):
        """Construct the full string that carries the message.

        With the specified verbosity parameters.
        """
        struct_string = self._get_structure_string(verb_level)
        time_string = self._get_time_string()
        if not isinstance(msg, str):
            msg = str(msg)
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

    def _write(self, cmod: str, msg: str, logfile: str = None):
        """Handle terminal/file writing.

        Write the message to the file and print
        it to the terminal if it is wanted.
        """
        if not self.suppress_terminal:
            print("{}{}{}".format(cmod, msg, self.RESET))
        if logfile == -1:
            return
        if self.logs['basename'] is None:
            return
        if logfile is not None:
            self.logfile = '-'.join((self.logs['basename'], logfile)) + '.log'
        elif self.logfile == '':
            self.logfile = '-'.join((self.logs['basename'], '1')) + '.log'

        if self.logfile in self.logs:
            if self.logs[self.logfile].closed:
                self.logs[self.logfile] = open(self.logfile, self.__write_status)
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
        """Ganeral decorator method.

        This method is the decorator to use for logging other methods.

        Parameters
        ----------
        style: str
            The logging style which will be resolved.
        verbosity: int
            The verbosity level to set, [1-5]

        """
        def real_decorator(function):
            def wrapper(*args, **kwargs):
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
    def flush(self):
        for fname, logfile in self.logs.items():
            if not self.logs[fname].closed:
                logfile.flush()

    @__guarantee_setup
    def clear(self):
        truncatable = ['r+', '+', 'w', 'a']
        if self.__write_status not in truncatable:
            return
        for fname, logfile in self.logs.items():
            if self.logs[fname].closed:
                self.logs[fname] = open(fname, self.__write_status)
            if self.__write_status in truncatable:
                self.logs[fname].truncate(0)

    @__guarantee_setup
    def reopen(self, filename):
        if fname in self.logs:
            self.logs[fname] = open(fname, self.__write_status)

    @__guarantee_setup
    def set_logfile_base(self, *, basename: str):
        """Set the basename for the logfile."""
        self.logs['basename'] = basename + '/'

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
        """Get current logfile."""
        return self.logfile

    @__guarantee_setup
    def set_verbosity(self, verbosity):
        """Set verbosity level."""
        self.verbosity = verbosity

    @__guarantee_setup
    def get_verbosity(self):
        """Get verbosity level."""
        return self.verbosity

    @__guarantee_setup
    def disable_colour(self):
        """Disable colour output in terminal."""
        for c in Format.colours():
            setattr(self, c, '')

    @__guarantee_setup
    def enable_colour(self):
        """Enable colour output in terminal."""
        dicts = {**Format.colours(), **self.__dict__}
        self.__dict__ = dicts


    #############################
    # SETUP COMPLETE, WRITE FUNCTIONS
    @__guarantee_setup
    def example(self):
        self.linebreak('%', logfile=-1)
        self.header1('header1: THIS IS A HEADER', logfile=-1)
        self.header2('header2: THIS IS A MINOR HEADER', logfile=-1)
        self.message('message: THIS IS A MESSAGE', logfile=-1)
        self.success('success: THIS IS A SUCCESS', logfile=-1)
        self.failure('failure: THIS IS A FAILURE', logfile=-1)
        self.debug('debug: THIS IS A DEBUG', logfile=-1)
        self.warn('warn: THIS IS A WARNING', logfile=-1)
        self.linebreak('#', logfile=-1)

    @__guarantee_setup
    def __general_messager(self, colour: str, msg: str, verb_level: int, logfile=None):
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(colour, full_msg, logfile=logfile)

    @__guarantee_setup
    def linebreak(self, breakstr: str = '#', width: int = 75, logfile=None):
        """Warn level."""
        width -= len('YYYY-MM-DD HH:MM:SS ')
        msg = f'{breakstr}' * width
        self.__general_messager(self.HEADER, msg, 2, logfile=logfile)

    @__guarantee_setup
    def warn(self, msg: str, verb_level: int = 2, logfile=None):
        """Warn level."""
        msg = f'(WARN) {msg}'
        self.__general_messager(self.WARNING, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def header1(self, msg: str, verb_level: int = 0, logfile=None):
        """Highest Header level."""
        msg = f'(HEADER) {msg}'
        self.__general_messager(self.HEADER, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def header2(self, msg: str, verb_level: int = 1, logfile=None):
        """Lower Header level."""
        msg = f'(header) {msg}'
        self.__general_messager(self.CYAN_TEXT, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def success(self, msg: str, verb_level: int = 1, logfile=None):
        """Success level (highest)."""
        msg = f'(SUCCESS) {msg}'
        self.__general_messager(self.OKGREEN, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def failure(self, msg: str, verb_level: int = 0, logfile=None):
        """Failure level (highest)."""
        msg = f'(FAILURE) {msg}'
        self.__general_messager(self.FAIL, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def message(self, msg: str, verb_level: int = 2, logfile=None):
        """Message level (lowest)."""
        msg = f'(GENERAL) {msg}'
        self.__general_messager(self.OKBLUE, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def debug(self, msg: str, verb_level: int = 4, logfile=None):
        """Debug level."""
        msg = f'(DEBUG) {msg}'
        self.__general_messager(self.WARNING, msg, verb_level, logfile=logfile)

    @__guarantee_setup
    def pyinput(self, message: str = '', verb_level: int = 0, logfile=None):
        """Gather input from the terminal."""
        total_Message = "Please input {}: ".format(message)
        out = input(total_Message)
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(total_Message, verb_level)
            self._write(self.WARNING, full_msg, logfile=logfile)
        return out

    @__guarantee_setup
    def waiting(self, auto: bool, seconds: int = 10, verb_level: int = 0, logfile=None):
        """Wait for a Ret or sigint."""
        if not auto:
            self.pyinput('[RET] to continue or CTRL+C to escape', logfile=logfile)
        elif verb_level <= self.verbosity:
            self.warn('Will continue in {}s. CTRL+C to escape'.format(seconds), logfile=logfile)
            time__sleep(seconds)

    def teardown(self):
        """Clean teardown call."""
        self.__teardown()

    def sigHandler(self):
        """Gather sigint thrown."""
        self.__sigHandler()


# Yielding singleton to module
Log = Logger()

__doc__ += Log.setup.__doc__
Log.__doc__ = __doc__

for func in __all__:
    if func.startswith('Log'):
        continue
    globals()[func] = getattr(Log, func)

# handling exit
atexit__register(Log.teardown)
signal__signal(signal__SIGTERM, Log.sigHandler)

# end of code
