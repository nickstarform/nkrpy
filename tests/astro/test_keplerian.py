"""."""
# flake8: noqa

# internal modules
import unittest

# external modules

# relative modules
from nkrpy import constants, astro

# global attributes
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class TestKeplerian(unittest.TestCase):

    def test_orbital_params():
        testing1, testing2 = astro.orbital_params(1, 10, 0.1, 0.9, pi / 2., 3. * pi / 4., size=2)
        testing3 = xyz_2_orbital(testing2, mass=1.)

        self.assertEqual(testing1.shape[-1], 7)
        self.assertEqual(testing2.shape[-1], 6)        


if __name__ == '__main__':
    unittest.main()

# end of code

# end of file
