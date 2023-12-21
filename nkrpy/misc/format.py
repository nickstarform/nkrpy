# flake8: noqa
"""General colour definitions.

A color init string consists of one or more of the following numeric codes:
* Attribute codes:
  00=none 01=bold 04=underscore 05=blink 07=reverse 08=concealed
* Text color codes:
  30=black 31=red 32=green 33=yellow 34=blue 35=magenta 36=cyan 37=white
* Background color codes:
  40=black 41=red 42=green 43=yellow 44=blue 45=magenta 46=cyan 47=white
* Extended color codes for terminals that support more than 16 colors:
  (the above color codes still work for these terminals)
  ** Text color coding:
      38;5;COLOR_NUMBER
  ** Background color coding:
      48;5;COLOR_NUMBER
COLOR_NUMBER is from 0 to 255.
"""

# internal modules

# external modules

# relative modules
from .._types import FormatClass

# global attributes
__all__ = ['Format']
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

class Formatter(FormatClass):

    __base_colours = {
        'HEADER': '\033[95m',
        'OKBLUE': '\033[94m',
        'OKGREEN': '\033[92m',
        'WARNING': '\033[93m',
        'FAIL': '\033[91m',
        'SUCCESS' : '\033[92m',
        'BOLD': '\033[1m',
        'RESET': '\033[0m',  # resets color and format

        'DIMINISH': '\033[2m',
        'UNDERLINE': '\033[4m',
        'BLINK': '\033[5m',  # only in supported terms, otherwise shown as reverse
        'REVERSE': '\033[7m',
        'HIDDEN': '\033[8m',
        'BOLDUNDERLINE': '\033[1m\033[4m',
        # ends the format (only un-reverse is working)
        'BOLD2UNDERLINE': '\033[21m',

        # Maps 16 color to 256 colors

        # Regular Colors
        'BLACK_TEXT': '\033[38;5;0m',
        'RED_TEXT': '\033[38;5;1m',
        'GREEN_TEXT': '\033[38;5;2m',
        'YELLOW_TEXT': '\033[38;5;3m',
        'BLUE_TEXT': '\033[38;5;4m',
        'MAGENTA_TEXT': '\033[38;5;5m',
        'CYAN_TEXT': '\033[38;5;6m',
        'WHITE_TEXT': '\033[38;5;7m',

        # Background
        'BLACK_BKGD': '\033[48;5;0m',
        'RED_BKGD': '\033[48;5;1m',
        'GREEN_BKGD': '\033[48;5;2m',
        'YELLOW_BKGD': '\033[48;5;3m',
        'BLUE_BKGD': '\033[48;5;4m',
        'MAGENTA_BKGD': '\033[48;5;5m',
        'CYAN_BKGD': '\033[48;5;6m',
        'WHITE_BKGD': '\033[48;5;7m',

        # High Intensty
        'IBLACK_TEXT': '\033[38;5;8m',
        'IRED_TEXT': '\033[38;5;9m',
        'IGREEN_TEXT': '\033[38;5;10m',
        'IYELLOW_TEXT': '\033[38;5;11m',
        'IBLUE_TEXT': '\033[38;5;12m',
        'IMAGENTA_TEXT': '\033[38;5;13m',
        'ICYAN_TEXT': '\033[38;5;14m',
        'IWHITE_TEXT': '\033[38;5;15m',

        # High Intensty backgrounds
        'IBLACK_BKGD': '\033[48;5;8m',
        'IRED_BKGD': '\033[48;5;9m',
        'IGREEN_BKGD': '\033[48;5;10m',
        'IYELLOW_BKGD': '\033[48;5;11m',
        'IBLUE_BKGD': '\033[48;5;12m',
        'IMAGENTA_BKGD': '\033[48;5;13m',
        'ICYAN_BKGD': '\033[48;5;14m',
        'IWHITE_BKGD': '\033[48;5;15m',
        }

    __user_colours = {}

    def colours(self):
        return {**self.__base_colours, **self.__user_colours}

    def __resolve_colour(self, name: str):
        name = name.upper().replace(' ', '')
        if name == '':
            name = 'WHITE_TEXT'
        all_colours = {**self.__base_colours, **self.__user_colours}
        if name in all_colours:
            return all_colours[name]
        setname = set(name)
        checking = [[n, min([len(set(n) - setname), len(setname - set(n))])] for n in all_colours]
        checking.sort(key=lambda x: x[-1])
        return all_colours[checking[0][0]]

    def __add_user_defined_colour(self, name: str, value: str):
        self.__user_colours[name.upper()] = value

    def __call__(self, name: str = '', value: str = None):
        if value is not None:
            self.__add_user_defined_colour(name, value)
        else:
            return self.__resolve_colour(name)

    def __repr__(self):
        test = 'TEST'
        build = ''
        all_colours = {**self.__base_colours, **self.__user_colours}
        reset = all_colours['RESET']
        for k, v in all_colours.items():
            build += f'''
            {k}: {v}{test}{reset}'''
        return build

    def __str__(self, fmt: str = None):
        return self.__repr__()

    def __all__(self):
        return list(self.__all__) + [k for k in ({**self.__base_colours, **self.__user_colours}).keys()]

Format = Formatter()

# end of code

# end of file
