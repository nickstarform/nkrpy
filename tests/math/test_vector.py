"""."""
# flake8: noqa

# internal modules
import unittest

# external modules
from nkrpy.math import vector

# relative modules

# global attributes
__all__ = ('TestVector',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class TestVector(unittest.TestCase):

    a = vector([0, 0, 0], [0, 1, 0])
    b = vector([0, 0, 0], [1, 0, 0])

    def test_vec_multiply(self):
        c = self.a * self.b
        d = self.b * self.a
        self.assertEqual(c, vector([0, 0, 0], [0, 0, 1]))

    def test_dot_multiply(self):
        c = self.a ^ self.b
        d = self.b ^ self.a

    def test_scalar_multiply(self):
        c = self.a * 2.
        d = 2. * self.a

    def test_pow(self):
        c = self.a ** 2
        c = self.a ** 3
        c = self.a ** 4
        c = self.a ** 5

    def test_scalar_divide(self):
        c = self.a / 2.  # should pass
        d = self.a / 2.  # should fail

    def test_add(self):
        c = self.a + self.b
        d = self.b + self.a

    def test_sub(self):
        c = self.a - self.b
        d = self.b - self.a

    def test_bool(self):
        # test if mag is >, <
        c = self.a > self.b
        c = self.b < self.a
        c = self.b >= self.a
        c = self.b <= self.a
        # check if vecs are equal
        c = self.b == self.a
        c = self.b != self.a
        c = self.b == vector([0, 0, 0], [1, 0, 0])


if __name__ == '__main__':
    unittest.main()

# end of code

# end of file
