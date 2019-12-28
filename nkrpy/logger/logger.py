# internal modules
from os import remove
from os.path import isfile
from glob import glob
import datetime
import time

# external modules

# relative modules
from . import colours

__all__ = ('Logger', )
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


# function that creates a logger
class SingletonMetaClass(type):
    def __init__(cls, name, bases, dict):
        super(SingletonMetaClass, cls).__init__(name, bases, dict)
        original_new = cls.__new__

        def my_new(cls, *args, **kwargs):
            if cls.instance is None:
                cls.instance = original_new(cls, *args, **kwargs)
            return cls.instance

        cls.instance = None
        cls.__new__ = staticmethod(my_new)


class Logger(object):
    """
    The Messenger class which handles pretty
    logging both to terminal and to a log
    file which was intended for running 
    codes on clusters in batch jobs where
    terminals were not slaved.
    """

    use_structure = ".    "
    __instance = None
    __metaclass__ = SingletonMetaClass

    def __init__(self, verbosity: int = 2, use_colour: bool = True,
                 use_structure: bool = False, add_timestamp: bool = True,
                 logfile=None):
        """Set the parameters for the Messenger class.
        """
        self.verbosity = verbosity

        # specifying colour options
        self.use_colour = use_colour
        if use_colour:
            self.enable_colour()
        else:
            self.disable_colour()

        self.use_structure = use_structure
        self.add_timestamp = add_timestamp
        self.logfile = logfile

        # overrides existing file
        if logfile is not None:
            self.f = open(logfile, 'w')

    def set_verbosity(self, verbosity):
        """
        Set the verbosity level for the class
        """
        self.verbosity = verbosity

    def get_verbosity(self):
        """
        Returns the verbosity level of the class
        """
        return self.verbosity

    def disable_colour(self):
        """
        Turns off all colour formatting.
        """
        for c in (self.BOLD, self.HEADER1, self.HEADER2,
                  self.OKGREEN, self.WARNING, self.FAIL,
                  self.MESSAGE, self.DEBUG, self.CLEAR):
            c = ''

    def enable_colour(self):
        """
        Enable all colour formatting.
        """
        self.__dict__ = {**colours.__dict__, **self.__dict__}

    def _get_structure_string(self, level):
        """
        Returns the string of the message with the specified level
        Which is dependent on the verbosity
        """

        string = ''
        if self.use_structure:
            for i in range(level):
                string = string + self.structure_string
        return string

    def _get_time_string(self):
        """
        Returns the detailed datetime for extreme debugging
        """

        string = ''
        if self.add_timestamp:
            string = '[{}] '.format(datetime.datetime.today())
        return string

    def _make_full_msg(self, msg, verb_level):
        """
        Constructs the full string that carries the message
        with the specified verbosity parameters
        """
        struct_string = self._get_structure_string(verb_level)
        time_string = self._get_time_string()
        return time_string + struct_string + msg

    def _write(self, cmod: str, msg: str, out: bool = True):
        """
        Write the message to the file and print 
        it to the terminal if it is wanted
        """
        if out:
            print("{}{}{}".format(cmod, msg, self.CLEAR))
        if type(self.logfile) is str:
            self.f.write(msg + '\n')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    The following commands are the ones
    to use when calling the logger
    will handle writing to the log
    file and to the terminal
    """
    def warn(self, msg, verb_level=2):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.WARNING, full_msg)

    def header1(self, msg, verb_level=0):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.HEADER1, full_msg)

    def header2(self, msg, verb_level=1):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.HEADER2, full_msg)

    def success(self, msg, verb_level=1):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.OKGREEN, full_msg)

    def failure(self, msg, verb_level=0):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.FAIL, full_msg)

    def message(self, msg, verb_level=2):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.MESSAGE, full_msg)

    def debug(self, msg, verb_level: int = 4):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.DEBUG, full_msg)

    def pyinput(self, message: str = '', verb_level: int = 0):

        total_Message = "Please input {}: ".format(message)
        out = input(total_Message)
        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(total_Message, verb_level)
            self._write(self.DEBUG, full_msg, False)
        return out

    def waiting(self, auto: bool, seconds: int = 10, verb_level: int = 0):
        if not auto:
            self.pyinput('[RET] to continue or CTRL+C to escape')
        elif verb_level <= self.verbosity:
            self.warn('Will continue in {}s. CTRL+C to escape'.format(seconds))
            time.sleep(seconds)

    def _REMOVE_(self, file: str):
        """
        This is a restructure of the os.system(rm) or the os.remove command
        such that the files removed are displayed appropriately or not removed
        if the file is not found
        """
        if type(file) is str:
            for f in glob('*'+file+'*'):
                if isfile(f):
                    try:
                        remove(f)
                        self.debug("Removed file {}".format(f))
                    except OSError:
                        self.debug("Cannot find {} to remove".format(f))
        else:
            for f in file:
                if isfile(f):
                    try:
                        remove(f)
                        self.debug("Removed file {}".format(f))
                    except OSError:
                        self.debug("Cannot find {} to remove".format(f))

# end of code
