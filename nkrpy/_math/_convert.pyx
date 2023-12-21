"""Astronomical Conversions."""

# standard modules

# external modules
import numpy as np
cimport numpy as cnp

# relative modules
from ..misc.functions import typecheck
from ._miscmath import radians as nkrpy__radians
from ._miscmath import deg as nkrpy__deg

# global attributes
__all__ = ['icrs2degrees', 'degrees2icrs',
           'j20002b1950', 'b19502j2000',
           'gal2j2000', 'j20002gal',
           'b19502helio', 'helio2b1950',
           'j20002helio', 'helio2j2000',
           'gal2helio', 'helio2gal',
           'b19502gal', 'gal2b1950',
           'cyl2cart', 'cart2cyl',
           'cyl2sph', 'sph2cyl',
           'cart2sph', 'sph2cart']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def icrs2icrs(icrs, joiner: list=['h', 'm', 's']):
    icrs = icrs2degrees(icrs)
    icrs = degrees2icrs(icrs)
    assert len(joiner) == len(icrs)
    return [''.join(x) for x in zip(map(lambda y: '%.2f' % y, icrs), joiner)]


def icrs2degrees(icrs, is_ra=False):
    """Convert ICRS to Degrees.

    Assume only single coordinate
    possible delimiters '°', 'd', 'h', 'm', 's', "'", '"', ' ', ':', '.'
    Examples
    --------
    >>> icrs2deg('11 11 11.5')

    """
    icrs_str = icrs
    if not typecheck(icrs):
        # turn from string into list
        icrs = (f'{icrs}').strip(' ')
        if 'd' in icrs or 'h' in icrs or "'" in icrs:
            # split hms, dms, h''' d''', °ms °'''
            icrs = [icrs]
            i = 0
            for delim in ['°', 'd', 'h', 'm', 's', "'", '"']:
                if i < len(icrs) and delim in icrs[i]:
                    t = icrs[i].split(delim)
                    del icrs[i]
                    icrs.extend(t)
                    i += 1
        elif ' ' in icrs:
            # split based on space
            """
            Input
            -----
            1 1 1.1 -> 1,1,1.1
            1 1.1 -> 1,1.1
            """
            icrs = icrs.split(' ')
        elif ':' in icrs:
            # split based on colon
            """
            Input
            -----
            1:1:1.1 -> 1:1:1.1
            1:1.1 -> 1:1.1
            """
            icrs = icrs.split(':')
        elif icrs.count('.') == 0:
            return float(icrs)
        elif icrs.count('.') > 0:
            # decimal degree given
            """
            Input
            -----
            1.1 -> 1,6
            1.1.1 -> 1,1.1
            """
            if icrs.count('.') == 1:
                return float(icrs)
            elif icrs.count('.') == 2:
                t = ([icrs.split('.')[0]])
                t.extend(['.'.join(icrs.split('.')[1:])])
                icrs = t
            elif icrs.count('.') == 3:
                t = (icrs.split('.')[:2])
                t.extend(['.'.join(icrs.split('.')[-2:])])
                icrs = t
    if len(icrs) <= 1:
        return float('.'.join(icrs))
    while len(icrs) < 3:
        last = float(icrs[-1])
        icrs[-1] = int(last)
        icrs.append((last - int(last)) * 60.)

    icrs = [float(i) if i != '' else 0 for i in icrs]
    ite = -1. if '-' in icrs_str else 1
    icrs = [abs(i) for i in icrs]
    ret = sum([v / 60 ** i for i, v in enumerate(icrs)]) * ite
    if is_ra:
        return ret / 15
    return ret


def degrees2icrs(degrees, is_ra=False):
    """Convert degrees to ICRS.

    'DD.DDD'
    The strings can have any form of non-numeric, non-decimal delimiter
    Examples
    --------
    >>> deg2icrs('11.18652778')

    """
    if is_ra:
        degrees *= 1 / 15
    ite = 1 if degrees > 0 else -1
    deg = abs(degrees)
    ret = [0 for i in range(3)]
    ret[0] = deg - deg % 1
    ret[1] = ((deg % 1) * 60)
    ret[2] = (ret[1] % 1) * 60
    ret[1] = ret[1] - (ret[1] % 1)
    ret[0], ret[1] = map(int, [ret[0], ret[1]])
    for i in range(len(ret)):
        v = ret[i]
        if v != 0.:
            ret[i] *= ite
            return ret
    return ret


# http://star-www.st-and.ac.uk/~fv/webnotes/chapter8.htm
def j20002b1950():
    """."""
    pass


def b19502j2000():
    """."""
    pass


def cart2cyl():
    """."""
    pass


def gal2j2000(ga: list):
    """Convert Galactic to Equatorial coordinates (J2000.0).

    Input: [l,b] in decimal degrees
    Returns: [ra,dec] in decimal degrees

    Source:
    - Book: "Practical astronomy with your calculator" (Peter Duffett-Smith)
    - Wikipedia "Galactic coordinates"

    Usage
    -----
    >>> ga2equ([0.0, 0.0]).round(3)
    array([ 266.405,  -28.936])
    >>> ga2equ([359.9443056, -0.0461944444]).round(3)
    array([ 266.417,  -29.008])
    """
    l, b = map(nkrpy__radians, ga)
    # North galactic pole (J2000) -- according to Wikipedia
    pole_ra = nkrpy__radians(192.859508)
    pole_dec = nkrpy__radians(27.128336)
    posangle = nkrpy__radians(122.932 - 90.0)
    ra = np.atan2((np.cos(b) * np.cos(l - posangle)),
                  (np.sin(b) * np.cos(pole_dec) -
                   np.cos(b) * np.sin(pole_dec) * np.sin(l - posangle))
                  ) + pole_ra
    dec = np.asin(np.cos(b) * np.cos(pole_dec) * np.sin(l - posangle) +
                  np.sin(b) * np.sin(pole_dec))
    return (nkrpy__deg(ra), nkrpy__deg(dec))


def gal2b1950(ga):
    """Convert Galactic to Equatorial coordinates (B1950.0).

    Input: [l,b] in decimal degrees
    Returns: [ra,dec] in decimal degrees

    Source:
    - Book: "Practical astronomy with your calculator" (Peter Duffett-Smith)
    - Wikipedia "Galactic coordinates"

    Usage
    -----
    >>> ga2equ([0.0, 0.0]).round(3)
    array([ 266.405,  -28.936])
    >>> ga2equ([359.9443056, -0.0461944444]).round(3)
    array([ 266.417,  -29.008])
    """
    l, b = map(nkrpy__radians, ga)
    pole_ra = nkrpy__radians(192.25)
    pole_dec = nkrpy__radians(27.4)
    posangle = nkrpy__radians(123.0 - 90.0)
    ra = np.atan2((np.cos(b) * np.cos(l - posangle)),
                  (np.sin(b) * np.cos(pole_dec) -
                   np.cos(b) * np.sin(pole_dec) * np.sin(l - posangle))
                  ) + pole_ra
    dec = np.asin(np.cos(b) * np.cos(pole_dec) * np.sin(l - posangle) +
                  np.sin(b) * np.sin(pole_dec))
    return (nkrpy__deg(ra), nkrpy__deg(dec))


def j20002gal():
    """."""
    pass


def b19502gal():
    """."""
    pass


def gal2helio():
    """."""
    pass


def helio2gal():
    r"""Helio2gal.

    Distances to astronomical objects beyond the Solar System are usually
      heliocentric, i.e. with respect to the Sun. However, Galactocentric
      distances are sometimes required. Heliocentric distances can be
      easily converted to Galactocentric distances by subtracting the
      heliocentric space vector of the Galactic centre, $\vec{R}_{0}$,
      from the heliocentric space vector of the object under study,
      $\vec{R}_{\rm hel}$, thus

    \vec{R}_{\rm gal} = \vec{R}_{\rm hel} - \vec{R}_{0}

    where

    \vec{R}_{\rm hel} = \begin{pmatrix} d \cos(b) \cos(l) \\ d \cos(b)
    \sin(l) \\ d \sin(b) \end{pmatrix} , \qquad{} \vec{R}_{0} =
    \begin{pmatrix} R_{0} \\ 0 \\ 0 \end{pmatrix} \end{equation*}

    with $R_{0}$ being the distance of the Sun from the Galactic centre
    and $l$, $b$ and $d$ the Galactic longitude, Galactic latitude and
    heliocentric distance, respectively, of the object under study.
    Hence, the distance of the object from the Galactic centre is given
    as;

    \begin{equation*} |\vec{R}_{\rm gal}| = \sqrt{[d \cos(b) \cos(l) -
    R_{0}]^{2} + d^{2} \cos^{2}(b) \sin^{2}(l) + d^{2} \sin^{2}(b)}
    \, . \end{equation*}

    This result is generally valid for all possible object positions
    irrespective of whether they are inside or outside the Solar
    circle.

    """
    pass


def helio2j2000():
    """."""
    pass


def helio2b1950():
    """."""
    pass


def j20002helio():
    """."""
    pass


def b19502helio():
    """."""
    pass


def __recasting(inp):
    """Recast input to 2d numpy.

    Check
    [.., .., ..]
    [[.., .., ..], ...]

    Return
    ------
    np.ndarray
        [[.., .., ..], ...]

    """
    if not isinstance(inp, np.ndarray):
        inp = np.array(inp, dtype=np.float)
    if len(inp.shape) == 1:
        inp = inp[np.newaxis, :]
    if not 1 == inp.shape.index(3):
        inp = inp.reshape(-1, 3)
    return inp


def cart2sph(xyz) -> np.ndarray:
    """Angle in radians."""
    xyz = __recasting(xyz)
    xy2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    r = np.sqrt(xy2 + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xy2), xyz[:, 2])  # from z axis
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([r, theta, phi])


def sph2cart(rtp) -> np.ndarray:
    """Angle in radians."""
    rtp = __recasting(rtp)
    r, theta, phi = rtp[:, 0], rtp[:, 1], rtp[:, 2]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def sph2cyl(rtp) -> np.ndarray:
    """Angle in radians."""
    rtp = __recasting(rtp)
    p = rtp[:, 0] * np.sin(rtp[:, 2])
    z = rtp[:, 0] * np.cos(rtp[:, 2])
    return np.array([p, rtp[:, 1], z])


def cyl2sph(ptz) -> np.ndarray:
    """Angle in radians."""
    ptz = __recasting(ptz)
    r = np.sqrt(ptz[:, 0] ** 2 + ptz[:, 2] ** 2)
    phi = np.arccos(ptz[:, 2] / r)
    return np.array([r, ptz[:, 1], phi])


def cyl2cart(ptz):
    """Angle in radians."""
    ptz = __recasting(ptz)
    x = ptz[:, 0] * np.cos(ptz[:, 1])
    y = ptz[:, 0] * np.sin(ptz[:, 1])
    return np.array([x, y, ptz[:, 2]])

# end of code

# end of file
