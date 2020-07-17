"""."""
# flake8: noqa

# internal modules
import unittest
import os
__cwd = os.getcwd()

# external modules

# relative modules
from nkrpy import radio_functions

# global attributes
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class TestRadioFunctions(unittest.TestCase):
    freq = 231  # GHz
    tmajor = 0.2  # arcsec
    tminor = 0.1  # arcsec
    fluxiness = 1000  # mJy/beam
    brightness = 10  # K/beam

    def test_k_2_jy():
        jy = astro.k_2_jy(freq, tmajor, tminor, brightness)
        self.assertEqual(jy, 8.733387888707039)

    def test_jy_2_k():
        k = astro.jy_2_k(freq, tmajor, tminor, fluxiness)
        self.assertEqual(k, 1.1450310151608851)
    
    def test_convert():
        fname = __path__ + '../test_100x100.fits'
        f1, h1, d1 = radio_functions.convert_file(fname, True, False)
        f2, h2, d2 = radio_functions.convert_file(fname, False, True)
        self.assertIsNotNone(f1)
        self.assertIsNotNone(f2)


if __name__ == '__main__':
    unittest.main()

# end of code

# end of file
