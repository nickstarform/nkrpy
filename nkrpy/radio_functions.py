"""Various Radio Functions."""

# internal modules

# external modules

# relative modules

# global attributes
__all__ = ('k_2_jy', 'jy_2_k')
__doc__ = """Numerous radio Functions that are used to modify typical datasets."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1

def k_2_jy(freq, theta_major, theta_minor, brightness):
    """Convert Kelvin to Jy."""
    """
    @param freq ghz
    @param theta arcseconds
    @param brightness Kelvin/beam
    @return jan mJy/beam."""
    conv = (1.222E3 * (freq ** -2) / theta_minor / theta_major) ** -1
    print(f'Conversion: {conv}')
    return brightness * conv


def jy_2_k(freq, theta_major, theta_minor, intensity):
    """Convert Kelvin to Jy."""
    """
    @param freq ghz
    @param theta arcseconds
    @param intensity mJy/beam
    @return temp Kelvin/beam."""
    conv = 1.222E3 * (freq ** -2) / theta_minor / theta_major
    print(f'Conversion: {conv}')
    return intensity * conv

def main():
    """Main caller function."""
    pass


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
