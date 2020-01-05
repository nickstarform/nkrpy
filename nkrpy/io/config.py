"""."""
# internal modules
from os.path import isfile
from os import getcwd, remove
from shutil import copyfile
from inspect import getfile
from importlib import import_module
from time import time as ctime

# external modules

# relative modules


# main class object for manipulating configuration files
class configuration(object):
    """Python Configuration generator/verifier."""

    def __init__(self, inputfile: str = None, cwd: str = None):
        """Dunder."""
        self.time = '{}'.format(str(ctime()).split('.')[0])

        if cwd is None:
            self.cwd = getcwd()
        else:
            self.cwd = cwd

        self.params = None

        if inputfile is None:
            self.example()
            raise RuntimeError('Configuration file unspecified...')
        else:
            self.inputfile = self.remove_ext(self.find_file(inputfile))

    def read(self):
        """Read configuration."""
        temp = vars(import_module(self.inputfile))
        mainlibs = ['time', 'numpy', 'scipy',
                    'version', 'datetime', 'os',
                    'sys']
        b = {}
        for i in temp:
            if (i.split('.')[0] not in mainlibs) and ('__' not in i):
                b[i] = temp[i]
        self.params = b
        if self.remove:
            remove('{}/{}.py'.format(self.cwd, self.inputfile))

    @property
    def get_params(self):
        """Return params."""
        return self.params

    def set_params(self, **kwargs):
        """Set Params."""
        if kwargs is not None:
            if self.params is not None:
                temp = self.params
            for key, value in kwargs.items():
                if self.verify_params(key):
                    temp[key] = value
            self.params = temp

    def add_params(self, **kwargs):
        """Add/update param."""
        if kwargs is not None:
            if self.params is not None:
                temp = self.params
            for key, value in kwargs.items():
                if not self.verify_params(key):
                    temp[key] = value
            self.params = temp

    def verify_params(self, *args):
        """Verify files are in params."""
        dictkeys = [x for x in self.params]
        for k in args:
            if k not in dictkeys:
                return False
            else:
                pass
        return True

    @staticmethod
    def remove_ext(inputfile):
        """Remove file extension."""
        return '.'.join(inputfile.split('.')[:-1])

    def find_file(self, inputfile):
        """Find input file within directory."""
        if isfile(inputfile):
            if (len(inputfile.split('/')) > 1) or ('.py' not in inputfile):
                dest = "{}/config_{}.py".format(self.cwd, self.time)
                copyfile(inputfile, dest)
                self.remove = True
            else:
                self.remove = False
                dest = inputfile
        else:
            raise RuntimeError('Input file not found: {}'.format(inputfile))
        return dest.split('/')[-1]

    def example(self, defaultconfig: str):
        """Populate an example file."""
        src = "{}.py".format(self.remove_ext(getfile(defaultconfig)))
        dest = "{}/defaultconfig.py".format(self.cwd)
        if not isfile(dest):
            copyfile(src, dest)


if __name__ == "__main__":
    print('Testing module\n')
    print("{}".format(__doc__))

# end of code

# end of file
