"""."""
# flake8: noqa

# internal modules
import unittest

# external modules

# relative modules
from nkrpy import logger
from test_singleton import singleton_helper

# global attributes
__all__ = ('test', )
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class TestBaseLogger(unittest.TestCase):
    __settings = {}
    __file = None

    def compline(self, lnum: int = 0):
        line = self.__file.readlines()
        line = line[lnum]
        line = ']'.join((line).split(']')[1:]).strip('\n').strip(' ')
        return line

    def setUp(self):
        self.__settings = {'append': False, 'verbosity': 3, 'use_colour': True,
                    'structure_string': "", 'add_timestamp': True,
                    'logfile': f'tests/logger/test-setup', 'suppress_terminal': True}
        logger.setup(**self.__settings)
        logger.teardown()
        name = logger.get_logfile()
        self.__file = open(name, 'r')
        line = self.compline()
        testline = 'The logger'
        self.__file .close()
        self.assertEqual(line[:len(testline)], testline)

    def test_decorator(self):
    
        @logger.decorator(style='debug')
        def test(*args, **kwargs):
            return args, kwargs

        _t = test(1, one = 1)
        name = logger.get_logfile()
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline(1)
        logger.teardown()
        self.__file .close()
        testline = '''Called <test> with params: (1,), {'one': 1}'''
        self.assertEqual(line[:len(testline)], testline)

    def test_failure(self):
        name = logger.get_logfile()
        testline = 'Testing failure'
        logger.failure(testline)
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline()
        logger.teardown()
        self.__file .close()
        self.assertEqual(line[:len(testline)], testline)
    
    def test_success(self):
        name = logger.get_logfile()
        testline = 'Testing success'
        logger.success(testline)
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline()
        logger.teardown()
        self.__file .close()
        self.assertEqual(line[:len(testline)], testline)
    
    def test_header1(self):
        name = logger.get_logfile()
        testline = 'Testing header1'
        logger.header1(testline)
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline()
        self.__file .close()
        logger.teardown()
        self.assertEqual(line[:len(testline)], testline)
    
    def test_header2(self):
        name = logger.get_logfile()
        testline = 'Testing header2'
        logger.header2(testline)
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline()
        self.__file .close()
        logger.teardown()
        self.assertEqual(line[:len(testline)], testline)
    
    def test_warn(self):
        name = logger.get_logfile()
        testline = 'Testing warn'
        logger.warn(testline)
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline()
        self.__file .close()
        logger.teardown()
        self.assertEqual(line[:len(testline)], testline)
    
    def test_message(self):
        name = logger.get_logfile()
        testline = 'Testing message'
        logger.message(testline)
        logger.teardown()
        self.__file = open(name, 'r')
        line = self.compline()
        self.__file .close()
        logger.teardown()
        self.assertEqual(line[:len(testline)], testline)

    def test_set_verbosity(self):
        start = logger.get_verbosity()
        logger.set_verbosity(start % 5)
        finish = logger.get_verbosity()
        self.assertEqual(start, finish % 5)

    def test_test_singleton(self):
        start = logger.get_verbosity()
        singleton_helper()
        finish = logger.get_verbosity()
        self.assertEqual(start, finish % 5)


if __name__ == '__main__':
    unittest.main()

# end of code

# end of file
