"""Misc Common Functions."""

# internal modules
from math import cos, sin, acos, ceil
import os
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

# external modules
import numpy as np
from scipy.optimize import curve_fit

# relative modules

# set the filename manually
__filename__ = __file__.split('/')[-1].strip('.py')


def typecheck(obj):
    """Check if object is iterable (array, list, tuple) and not string."""
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def addspace(arbin, spacing='auto'):
    if typecheck(arbin):
        if str(spacing).lower() == 'auto':
            spacing = max([len(x) for x in map(str, arbin)]) + 1
            return [_add(x, spacing) for x in arbin]
        elif isinstance(spacing, int):
            return [_add(x, spacing) for x in arbin]
    else:
        arbin = str(arbin)
        if spacing.lower() == 'auto':
            spacing = len(arbin) + 1
            return _add(arbin, spacing)
        elif isinstance(spacing, int):
            return _add(arbin, spacing)
    raise(TypeError, f'Either input: {arbin} or spacing: {spacing} are of incorrect types. NO OBJECTS')


def _add(sstring, spacing=20):
    """Regular spacing for column formatting."""
    sstring = str(sstring)
    while True:
        if len(sstring) >= spacing:
            sstring = sstring[:-1]
        elif len(sstring) < (spacing - 1):
            sstring = sstring + ' '
        else:
            break
    return sstring + ' '

def list_files(dir):
    """List all the files within a directory."""
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        if (len(files) > 0):
            for file in files:
                r.append(subdir + "/" + file)
    return r


def list_files2(startpath, ignore='', formatter=['  ', '| ', 1, '|--']):
    """Intelligent directory stepper + ascii plotter."""
    """
    Will walk through directories starting at startpath
    ignore is a csv string 'ignore1, ignore2, ignore3' that will ignore any
        file or directory with same name
    Formatter will format the output in:
        [starting string for all lines,
         the iterating level parameter,
         the number of iterations for the level parameter per level,
         the final string to denote the termination at file/directory]
    example:
    |--/
    | |--CONTRIBUTING.md
    | |--.gitignore
    | |--LICENSE
    | |--CODE_OF_CONDUCT.md
    | |--README.md
    | |--PULL_REQUEST_TEMPLATE.md
    | |--refs/
    | | |--heads/
    | | | |--devel
    """
    s, a, b, c = formatter
    full = []
    for root, firs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = s + a * b * level + c
        full.append('{}{}/'.format(indent, os.path.basename(root)))
        subindent = s + a * b * (level + 1) + c
        for f in files:
            if f.split('/')[-1] not in ignore.split(', '):
                full.append('{}{}'.format(subindent, f))
    """
    master = os.walk(startpath)
    step = 0
    while step < len(master)-1:
        root, dirs, files = master[step]
        if root.split('/')[-1] not in ignore.split(', '):
            level = root.replace(startpath, '').count(os.sep)
            indent = s + a * b * (level) + c
            full.append('{}{}/'.format(indent, os.path.basename(root)))
            subindent = s + a * b * (level+1) + c
            for f in files:
                if f.split('/')[-1] not in ignore.split(', '):
                    full.append('{}{}'.format(subindent, f))
        step += 1
    """
    return full


def equivalent_width(spectra, blf, xspec0, xspec1, fit='gauss',
                     params=[1, 1, 1]):
    """Compute spectral line eq. width."""
    from .miscmath import gauss
    """
    finds equivalent width of line
    spectra is the full 2d array (lam, flux)
    blf is the baseline function
    xspec0 (xspec1) is the start(end) of the spectral feature


    # PROBABLY EASIER TO JUST ASSUME GAUSSIAN OR SIMPLE SUM


    def gaussc(x, A, mu, sig, C):
        return A*np.exp(-(x-mu)**2/2./sig**2) + C

    from scipy.optimize import curve_fit

    featx, featy = lam[featurei], flux[featurei]
    expected1=[1.3, 2.165, 0.1, 1.]
    params1, cov1=curve_fit(gaussc, featx, featy, expected1)
    sigma=params1[-2]

    # I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)
    # sigma = dw / 2 / sqrt (2 * ln (core/Iw))
    # fwhm = 2.355 * sigma = 2(2ln2)^0.5 * sigma
    # flux = core * sigma * sqrt (2*pi)
    # eq. width = abs (flux) / cont

    fwhm = 2. * (2. * np.log(2.))**0.5 * sigma
    core, center, nsig, cont = params1
    flx = core * sigma * np.sqrt (2.*np.pi)
    eqwidth = abs(flx) / cont
    print(eqwidth)
    print(fwhm)
    """

    specfeatxi = np.array(between(spectra, xspec0, xspec1))[:, 0]
    specfeatxv = np.array(between(spectra, xspec0, xspec1))[:, 1]

    if fit == 'gauss':
        _params2, _cov2 = curve_fit(gauss, specfeatxv, spectra[specfeatxi, 1],
                                    *params)
        _sigma2 = np.sqrt(np.diag(_cov2))
        function = gauss(specfeatxv, *_expected2)
        return
    elif fit == 'ndgauss':
        pass


def between(l1, val1, val2):
    """Find values between bounds, exclusively."""
    """Return the value and index for everything in iterable
    between v1 and v2, exclusive."""
    if val1 > val2:
        low = val2
        high = val1
    elif val1 < val2:
        low = val1
        high = val2
    else:
        print('Values are the same')
        return []

    l2 = []
    for j, i in enumerate(l1):
        if(i > low) and (i < high):
            l2.append([j, i])
    return l2
