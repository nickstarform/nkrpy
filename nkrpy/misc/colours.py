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

# global attributes
__all__ = (
# high intensity background
'IBLACK_BKGD',
'IRED_BKGD',
'IGREEN_BKGD',
'IYELLOW_BKGD',
'IBLUE_BKGD',
'IMAGENTA_BKGD',
'ICYAN_BKGD',
'IWHITE_BKGD',
# high intensity text
'IBLACK_TEXT',
'IRED_TEXT',
'IGREEN_TEXT',
'IYELLOW_TEXT',
'IBLUE_TEXT',
'IMAGENTA_TEXT',
'ICYAN_TEXT',
'IWHITE_TEXT',
# background
'BLACK_BKGD',
'RED_BKGD',
'GREEN_BKGD',
'YELLOW_BKGD',
'BLUE_BKGD',
'MAGENTA_BKGD',
'CYAN_BKGD',
'WHITE_BKGD',
# text
'BLACK_TEXT',
'RED_TEXT',
'GREEN_TEXT',
'YELLOW_TEXT',
'BLUE_TEXT',
'MAGENTA_TEXT',
'CYAN_TEXT',
'WHITE_TEXT',
# special
'HEADER',
'OKBLUE',
'OKGREEN',
'WARNING',
'FAIL',
'SUCCESS',
'BOLD',
'RESET',
'DIMINISH',
'UNDERLINE',
'BLINK',
'REVERSE',
'HIDDEN',
'BOLDUNDERLINE',
'BOLD2UNDERLINE')
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
SUCCESS = OKGREEN
BOLD = '\033[1m'
RESET = '\033[0m'  # resets color and format

DIMINISH = '\033[2m'
UNDERLINE = '\033[4m'
BLINK = '\033[5m'  # only in supported terms, otherwise shown as reverse
REVERSE = '\033[7m'
HIDDEN = '\033[8m'
BOLDUNDERLINE = '\033[1m\033[4m'
# ends the format (only un-reverse is working)
BOLD2UNDERLINE = '\033[21m'

# Maps 16 color to 256 colors

# Regular Colors
BLACK_TEXT   = '\033[38;5;0m'
RED_TEXT     = '\033[38;5;1m'
GREEN_TEXT   = '\033[38;5;2m'
YELLOW_TEXT  = '\033[38;5;3m'
BLUE_TEXT    = '\033[38;5;4m'
MAGENTA_TEXT = '\033[38;5;5m'
CYAN_TEXT    = '\033[38;5;6m'
WHITE_TEXT   = '\033[38;5;7m'

# Background
BLACK_BKGD   = '\033[48;5;0m'
RED_BKGD     = '\033[48;5;1m'
GREEN_BKGD   = '\033[48;5;2m'
YELLOW_BKGD  = '\033[48;5;3m'
BLUE_BKGD    = '\033[48;5;4m'
MAGENTA_BKGD = '\033[48;5;5m'
CYAN_BKGD    = '\033[48;5;6m'
WHITE_BKGD   = '\033[48;5;7m'

# High Intensty
IBLACK_TEXT   = '\033[38;5;8m'
IRED_TEXT     = '\033[38;5;9m'
IGREEN_TEXT   = '\033[38;5;10m'
IYELLOW_TEXT  = '\033[38;5;11m'
IBLUE_TEXT    = '\033[38;5;12m'
IMAGENTA_TEXT = '\033[38;5;13m'
ICYAN_TEXT    = '\033[38;5;14m'
IWHITE_TEXT   = '\033[38;5;15m'

# High Intensty backgrounds
IBLACK_BKGD   = '\033[48;5;8m'
IRED_BKGD     = '\033[48;5;9m'
IGREEN_BKGD   = '\033[48;5;10m'
IYELLOW_BKGD  = '\033[48;5;11m'
IBLUE_BKGD    = '\033[48;5;12m'
IMAGENTA_BKGD = '\033[48;5;13m'
ICYAN_BKGD    = '\033[48;5;14m'
IWHITE_BKGD   = '\033[48;5;15m'

# end of code

# end of file
