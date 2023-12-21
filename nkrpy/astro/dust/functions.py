"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np

# relative modules
from .._functions import blackbody_hz
from . import kappa
from ..._unit import Unit as nc__unit  # noqa


# global attributes
__all__ = ["dustmass", ]
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

def dustmass(dist=100, dist_unit="pc", wav=0.1,
             wav_unit="cm", flux_jy=0, temp_k=20,
             model_name="oh1994", beta=1.7, gas_density=1e6, opacity=None):
    """Calculate dust mass.

    @param dist
    dist_unit
    wavelength
    wavelength_unit
    flux
    temp
    model
    beta
    opacity
    Assuming temp in Kelvin, flux in Janskys
    """
    dist = nc__unit(vals=dist, baseunit=dist_unit, convunit='cm').get_vals()
    wav_micron = nc__unit(vals=wav, baseunit=wav_unit, convunit='micron').get_vals()
    freq = nc__unit(vals=wav, baseunit=wav_unit, convunit='hz').get_vals()
    intensity = blackbody_hz(nu_hz=freq, temperature_k=temp_k) # noqa
    if not opacity:
        opacity = kappa(wav_micron,
                        model_name=model_name,
                        density=gas_density,
                        beta=beta)  # cm^2/g
    if not isinstance(opacity, (tuple, list, np.ndarray)):
        opacity = [opacity]
    _ret = []
    for x in opacity:
        _tmp = (dist ** 2 * flux_jy * 1e-23 / x / intensity)  # noqa in grams units (no ISM assump.)
        _ret.append([x, _tmp])
    return np.array(_ret)


def test():
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
