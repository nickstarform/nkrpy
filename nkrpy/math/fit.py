"""Various math functions."""

# standard modules
import types

# external modules
import numpy as np
from scipy.special import wofz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import rv_continuous, norm
from IPython import embed

# relative modules
from .miscmath import binning
from .. import unit
from .. import constants

# global attributes
__all__ = ('fit_conf', 'sigma_clip_fit', 'quad', 'voigt',
           'gauss', 'ndgauss', 'polynomial', 'baseline',
           'plummer_density', 'plummer_mass', 'plummer_radius',
           'linear')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def fit_conf(x, y, func: types.FunctionType, opt, ci: float = 0.95):
    """Fit a confidence interval.

    Parameters
    ----------
    x: iterable
    y: iterable
    func: function
        function to fit
    opt: iterable
        starting parameters for function to fit
    ci: float
    """
    # Define confidence interval.
    ci = 0.95
    # Convert to percentile point of the normal distribution.
    # See: https://en.wikipedia.org/wiki/Standard_score
    pp = (1. + ci) / 2.
    # Convert to number of standard deviations.
    nstd = norm.ppf(pp)
    #func = np.poly1d(opt)
    from IPython import embed
    #embed()

    # Find best fit.
    popt, pcov = curve_fit(func, x, y, p0=opt)
    # Standard deviation errors on the parameters.
    perr = np.sqrt(np.diag(pcov))
    # Add nstd standard deviations to parameters to obtain the upper confidence
    # interval.
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
    return (popt_up, popt_dw)

"""
from nkrpy.miscmath import sigma_clip_fit, binning
import matplotlib.pyplot as plt
import numpy as np
import inspect

d = '/home/reynolds/local/APO/further_analysis/hops370'
f = ('lam.npy','flux.npy')
xdo,ydo = map(lambda g: np.loadtxt(f'{d}/{g}'), f)
#ydo *= -1. * ydo

def func(x, a,b,c,d,e,f,g,h, i):
    return i * x ** 8 +\
        h * x ** 7 +\
        g * x ** 6 +\
        f * x ** 5 +\
        e * x ** 4 +\
        d * x ** 3 +\
        c * x ** 2 +\
        b * x + a


p0 = (np.full(len(inspect.signature(func).parameters.values()) - 1, 1, dtype=np.float)).tolist()

xd, yd = map(lambda x: binning(x, 1), [xdo,ydo])
s = sigma_clip_fit(xd, yd, func, p0, 3, 2)
print(s)

plt.figure()
plt.scatter(xdo, ydo, color='black', marker='.', lw=1)
plt.plot(xdo, func(xdo, *s[1]), color='blue', lw=1)
plt.scatter(xdo[s[0]], ydo[s[0]], color='red', marker='.', lw=1)
plt.show()
"""

def sigma_clip_fit(xdata, ydata, func, p0,
                   sigma_clip: int = 5,
                   max_iterations: int = 5):
    """Fit data with arbitrary function and sigma_clip.

    Parameters
    ----------
    xdata: Iterable
        Iterable of the xdata (1D)
    ydata: Iterable
        Iterable of the ydata (1D)
    func: function
        Arbitrary function to fit. Must be of form func(x, p1,p2 ...)
    p0: Iterable
        The starting parameters for the above function.
        This will help converge results
    sigma_clip: int
        The sigma level to clip data when applying fit
    max_iterations: int
        The number of times to perform the fit. Setting this higher
        takes longer but is more accurate.

    Returns
    -------
    p0: Iterable
        The converged parameters for the above function.
    err: float
        The sigma clipped error for the data.

    Usage
    -----
    '''
    def func(x, a,b,c):
        return b * x + c
    x = some wavelength values
    y = some flux values
    x, y are 1D and are roughly linear with some error and
        really large emission/absorption
    '''

    sigma_clip_fit(x, y, func, [1], sigma_clip: int=5, max_iterations: int=5)
    """
    def plot(x, y, f, p):
        plt.figure()
        plt.scatter(x, y, color='black', marker='.', lw=1)
        plt.plot(x, func(x, *p), color='blue', lw=1)
        plt.show()

    if not isinstance(xdata, np.ndarray):
        xdata = np.array(xdata)
    if not isinstance(ydata, np.ndarray):
        ydata = np.array(ydata)
    if xdata.shape[0] < 3:
        return
    xd, yd = map(lambda x: binning(x, 3), [xdata, ydata])
    ind = np.argsort(xd)
    nx, ny = map(lambda x: x[ind], [xd, yd])
    mask = np.full(nx.shape[0], False, dtype=bool)
    chk_last_mask = np.full(nx.shape[0], True, dtype=bool)
    rolling_sigma = []
    for i in range(max_iterations):
        if np.array_equal(chk_last_mask, mask):
            continue
        else:
            chk_last_mask = np.copy(mask)
        popt, pcov = curve_fit(func, nx[~mask], ny[~mask], p0=p0)
        p0 = popt
        fn = func(nx, *p0)
        y = ny / fn
        y = y / np.median(y)
        tmp_mask = np.copy(mask)
        rolling_sigma.append(y[~(mask + tmp_mask)].std())
        mask = ((np.abs(y - 1.) > sigma_clip * rolling_sigma[-1]) + mask + tmp_mask)

    embed()
    finfn = func(xdata, *p0)
    finy = ydata / finfn
    finmask = np.abs(finy - 1.) > rolling_sigma[-1]
    err = (ydata[~finmask] - func(xdata[~finmask], *p0)).std()
    return (finmask, p0, err)


def voigt(x, mu, alpha, gamma):
    """
    Voigt Profile x, alpha, gamma
    """
    sigma = alpha / np.sqrt(2. * np.log(2))
    return np.real(wofz((x - mu + 1.j * gamma) / sigma / np.sqrt(2.))) /\
        sigma / np.sqrt(2.*np.pi)


def emissivegaussian(x, mu: float, fwhm: float, flux: float, skew: float = 1):
    """Emissive gaussian.

    Parameters
    ----------
    x: iterable
    mu: float
        center of gaussian, in same units as x
    fwhm: float
        fwhm of the line, in km/s
    flux: float
        integrated flux of the line
    skew: float
        amount to skew gaussian
    """
    sigma = unit('km', 'm', fwhm) / (2. * np.sqrt(2. * np.log(2)))
    c = unit('cm', 'm', constants.c)
    s = mu * sigma / c
    a = flux / np.sqrt(2. * np.pi) / s
    dx = (x - mu) / s
    dx[x > mu] /= skew
    if skew > 1:
        return 2. * a * np.exp(-0.5 * dx ** 2) / (1. + skew)
    return a * np.exp(-0.5 * dx ** 2)


def gauss(x, mu, sigma, a):
    """Define a single gaussian."""
    return a * np.exp(-(x - mu) ** 2 / 2. / sigma**2)


def ndgauss(x, params):
    """N dimensional gaussian.

    assumes params is a 2d list
    """
    for i, dim in enumerate(params):
        if i == 0:
            final = gauss(x, *dim)
        else:
            final = np.sum(gauss(x, *dim), final, axis=0)

    return final


def addconst(func, c):
    """Add constant."""
    return func + c


def polynomial(x, params):
    """Polynomial function.

    assuming params is a 1d list
    constant + 1st order + 2nd order + ...
    """
    for i, dim in enumerate(params):
        if i == 0:
            final = [dim for y in x]
        else:
            final = np.sum(dim * x ** i, final, axis=0)

    return final


def baseline(x, y, order=2):
    """Fit a baseline.

    Input the xvals and yvals for the baseline
    Will return the function that describes the fit
    """
    fit = np.polyfit(x, y, order)
    fit_fn = np.poly1d(fit)
    return fit_fn


def plummer_density(x, mass, a):
    """Return Plummer density giving cirical radius and mass)."""
    return 3. * mass / (4. * np.pi * a ** 3) * (1. + (x / a) ** 2)**(-5. / 2.)


def plummer_mass(x, mass, a):
    """Return the mass for a plummer sphere."""
    return mass * x ** 3 / ((x ** 2 + a ** 2) ** (3. / 2.))


def plummer_radius(mass_frac, a):
    """Sampling function to evenly sample mass distribution."""
    return a * ((1. / mass_frac) ** (2. / 3.) - 1.) ** (-0.5)


def linear(x, a, b):
    """Linear function."""
    return a * x + b


def quad(x, a, b, c):
    """Linear function."""
    return a * x ** 2 + b * x + c


# end of code

# end of file
