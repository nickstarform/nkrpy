"""."""

# internal modules
from datetime import datetime

# external modules

# relative modules
# from ...misc.load import load_cfg

# global attributes
__all__ = ('test',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
date = str(datetime.now())

t_header = ';\n' +\
           f'; File generated from {__package__} on {date}\n' +\
           '; Towards _OBJECT_\n' +\
           ';\n'
t_setup = f'#count _C_\n' +\
          f'#interval _I_\n' +\
          f'#binning _B_\n' +\
          f'#filter _F_\n'
t_target = f'_N_\t_RA_\t_DEC_\n'


def mosaic(data, output, obj, count=1, interval=1, binning=1, filt='B,V,R,I'):
    """Mosaic colours together."""
    # assuming data is 2d, with each row 3vals <name, ra, dec>
    count = str(count).split(',')
    interval = str(interval).split(',')
    binning = str(binning).split(',')
    filt = str(filt).split(',')
    _h = t_header.replace('_OBJECT_', obj)
    for ite, fil in enumerate(filt):
        c, i, b, fil = list(map(lambda x: x.strip(' '),
                                [count[ite], interval[ite],
                                 binning[ite], fil]))
        _s = t_setup.replace('_C_', c).replace('_I_', i)\
                    .replace('_B_', b).replace('_F_', fil)
        _t = ''
        for x in data:
            _t += t_target.replace('_N_', x[0].replace(' ', ''))\
                          .replace('_RA_', x[1])\
                          .replace('_DEC_', x[2])
        _fin = _h + _s + _t
        if output:
            with open(output + f'_{fil}.txt', 'w') as f:
                f.write(_fin + '\n')
        else:
            return True


def singlet():
    pass


def main():
    pass


def test():
    """Testing function for module."""
    assert mosaic((('Test Point 1', '1', '1'),), None, 'Test', 1, 1, 1, 'B')


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
