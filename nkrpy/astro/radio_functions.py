"""Various Radio Functions."""

# internal modules

# external modules

# relative modules
from ..io.fits import read, write
from ..misc.decorators import validate

# global attributes
__all__ = ('k_2_jy', 'jy_2_k', 'convert_file')
__doc__ = """Numerous radio Functions that are used to
modify typical datasets."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def rad_2_as(rad):
    return float(rad) * 3600.


def k_2_jy(freq, theta_major, theta_minor, brightness):
    """Convert Kelvin to Jy.

    @param freq ghz
    @param theta arcseconds
    @param brightness Kelvin/beam
    @return jan mJy/beam.
    """
    conv = (1.222E3 * (freq ** -2) / theta_minor / theta_major) ** -1
    print(f'Conversion: {conv}')
    return brightness * conv


def jy_2_k(freq, theta_major, theta_minor, intensity):
    """Convert Kelvin to Jy.

    @param freq ghz
    @param theta arcseconds
    @param intensity mJy/beam
    @return temp Kelvin/beam.
    """
    conv = 1.222E3 * (freq ** -2) / theta_minor / theta_major
    print(f'Conversion: {conv}')
    return intensity * conv


@validate
def convert_file(filename: str, jy_k: bool, k_jy: bool):
    """."""
    header, data = read(filename)
    inp = [float(header['RESTFRQ']) / 1E9,
           rad_2_as(header['BMAJ']),
           rad_2_as(header['BMIN'])]
    print(f'Flux density unit: {header["BUNIT"]}')
    print(f'RFREQ: {inp[0]}GHZ\nBeam: {inp[1]}x{inp[2]}')
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

# end of code
