"""Convert a given image flux density."""

# internal modules
from argparse import ArgumentParser as ap

# external modules
from nkrpy.apo.fits import read, write
from nkrpy.constants import pi
from nkrpy.radio_functions import *
import numpy as np

# relative modules

# global attributes
__all__ = ('test', 'main')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1

def validate(func):
    def wrapper(f: str, a1: bool = False, a2: bool = False):
        both_f_either_t = (a1 and a2) or not (a1 or a2)
        if both_f_either_t:
            raise Exception('Incorrect input.' +
                            'Please select either Jy->K or K->Jy')
        else:
            return func(f, a1, a2)
    return wrapper


def rad_2_as(rad):
    rad = float(rad)
    return rad * 3600.


@validate
def main(filename: str, jy_k: bool, k_jy: bool):
    """Main caller function."""
    header, data = read(filename)
    inp = [float(header['RESTFRQ']) / 1E9,
           rad_2_as(header['BMAJ']),
           rad_2_as(header['BMIN'])]
    print(f'Flux density unit: {header["BUNIT"]}\nRFREQ: {inp[0]}GHZ\nBeam: {inp[1]}x{inp[2]}')
    unit = str(header['BUNIT']).split('/')[0]
    if unit == 'Jy' or unit == 'K':
        cv = 1000
    elif unit == 'mJy' or unit == 'mK':
        cv = 1

    if jy_k:
        nd = jy_2_k(*inp, data * cv)
        nh = 'K/beam'
    elif k_jy:
        nd = k_2_jy(*inp, data) / cv
        nh = 'Jy/beam'
    if nh:
        header['BUNIT'] = nh
        write(f'Temp_{filename}', header=header, data=nd)


if __name__ == "__main__":
    """Directly Called."""
    parser = ap()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-jy', default=False, action='store_true', help='convert to jy') # noqa
    parser.add_argument('-k', default=False, action='store_true', help='convert to jy') # noqa
    args = parser.parse_args()

    print('Testing module')
    main(args.i, args.k, args.jy)
    print('Test Passed')

# end of code
