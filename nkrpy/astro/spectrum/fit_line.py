"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules

# relative modules
from nkrpy.math import gauss, polynomial
from nkrpy.astro import atomiclines as lines
from nkrpy.unit import unit

# global attributes
__all__ = ('test', 'fit')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def fit(data: np.ndarray, xr: list, xunit: str = 'micron', linename: str=None, linecenter: unit=None, linewidth: unit=None):
    """Summary line.

    Extended description of function.

    >>> main()

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value
    """
    
    width = (xr[-1] - xr[0]) / 2.
    linewidth = linewidth if linewidth else unit(vals=width / center / 10., baseunit=xunit)
    linewidth = linewidth.convert(xunit)
    linecenter = linecenter if linecenter else unit(vals=(xr[-1] + xr[0]) / 2., baseunit=xunit)
    linecenter = linecenter.convert(xunit)
    med = np.median(data[1, :])
    dmax = np.max(data[1, :])
    dmin = np.min(data[1, :])
    rms = np.std(data[1, :])
    polyest = [0, 0, med]
    polylim = [[-np.inf, np.inf], [-np.inf, np.inf], [dmin, dmax]]
    gauss1est = [dmax, linecenter, linewidth] + polyest
    gauss1lim = [[-dmax, dmax], [linecenter - width, linecenter + width], [linewidth / 10., width]] + polylim
    gauss2est = [dmax, linecenter, linewidth] + gauss1est + polyest
    gauss2lim = [[-dmax, dmax], [linecenter - width, linecenter + width], [linewidth / 10., width]] + gauss1lim + polylim
    fits = []
    # fit poly
    popt, pcov = scipy.optimize.curve_fit(polynomial, data[0, :], data[1, :], sigma=data[2, :], p0=polyest, bounds=polylim)
    res = polynomial(data[0, :], *popt)
    polystd = np.std((res - data[1, :]) / data[1, :])
    fits.append([polystd, res])
    # fit gauss
    def _fit(x, amp, mu, sig, a, b, c):
        return gauss(x, amp, mu, sig) + polynomial(x, a, b, c)
    popt, pcov = scipy.optimize.curve_fit(_fit, data[0, :], data[1, :], sigma=data[2, :], p0=gauss1est, bounds=gauss1lim)
    res = _fit(data[0, :], *popt)
    gauss1std = np.std((res - data[1, :]) / data[1, :])
    fits.append([gauss1std, res])
    # fit gauss
    def _fit(x, amp, mu, sig, amp2, mu2, sig2, a, b, c):
        return gauss(x, amp, mu, sig) + gauss(x, amp2, mu2, sig2) + polynomial(x, a, b, c)
    popt, pcov = scipy.optimize.curve_fit(_fit, data[0, :], data[1, :], sigma=data[2, :], p0=gauss2est, bounds=gauss2lim)
    res = _fit(data[0, :], *popt)
    gauss2std = np.std((res - data[1, :]) / data[1, :])
    fits.append([gauss2std, res])

    fit.sort(key=lambda x: x[0])
    return fit[-1]


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
