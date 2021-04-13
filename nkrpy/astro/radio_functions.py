"""Various Radio Functions."""

# internal modules

# external modules

# relative modules
from ..io.fits import read as nkrpy_read
from ..io.fits import write as nkrpy_write
from ..misc.decorators import validate

# global attributes
__all__ = ('k_2_jy', 'jy_2_k', 'convert_file')
__doc__ = """Numerous radio Functions that are used to
modify typical datasets."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def rad_2_as(rad):
    return float(rad) * 3600.


def k_2_jy(freq: float, theta_major: float,
           theta_minor: float, brightness: float) -> float:
    """Convert Kelvin to Jy.

    Parameters
    ----------
    Parameters
    ----------
    freq: float
        ghz
    theta_major: float
        arcseconds
    theta_minor: float
        arcseconds
    brightness: float
        Kelvin/beam.
    """
    conv = (1.222E3 * (freq ** -2) / theta_minor / theta_major) ** -1
    return brightness * conv


def jy_2_k(freq: float, theta_major: float,
           theta_minor: float, intensity: float) -> float:
    """Convert Kelvin to Jy.

    Parameters
    ----------
    freq: float
        ghz
    theta_major: float
        arcseconds
    theta_minor: float
        arcseconds
    intensity: float
        mJy/beam

    """
    conv = 1.222E3 * (freq ** -2) / theta_minor / theta_major
    return intensity * conv


@validate
def convert_file(filename: str, jy_k: bool = False, k_jy: bool = False) -> tuple:
    """Convert a file between the types jy and kelvin.

    File must have restfrq bmaj bmin bunit defined

    Parameters
    ----------
    filename: str
        name of the file to convert. Must
    jy_k: bool
        default False, convert from jy to kelvin
    k_jy: bool
        default False, convert from kelvin to jy

    Returns
    -------
    tuple
        "conversionType_oldFilename", newHeader, new3DData

    """
    assert not (jy_k and k_jy)
    header, data = nkrpy_read(filename)
    header = header[0]
    data = data[0]
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
        fname = f'{nh.replace("/", "_")}_{filename}'
        header['BUNIT'] = nh
        nkrpy_write(fname, header=header, data=nd)
        return fname, header, nd
    return None, None, None

# end of code
