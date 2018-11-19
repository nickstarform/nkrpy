"""Find Keplerian Parameters. orbital_params(lsma,usma,le,ue,li,ui,mass,size)."""

"""Use orbital_params or orbital_2_xyz as the main function call.
For former: input distribution params and generates a gaussian distribution of 
orbital params in both cartesian and the orbital elements.
The orbital_2_xyz converts orbital elements to xyz components.
Keep everything in AU, days, solar masses, radians"""


import numpy as np
from nkrpy.constants import pi
try:
    from collections.abc import Iterable
except:
    from collections import Iterable

__all__ = ['orbital_params', 'orbital_2_xyz']

# Acceptable Numerical Error
eps = 1E-12


def typecheck(obj):
    """Check if object is iterable (array,list,tuple) and not string."""
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def _2d(ite, dtype):
    """Create 2d array."""
    _shape = tuple([len(ite), len(ite[0])][::-1])
    _return = np.zeros(_shape, dtype=dtype)
    for i, x in enumerate(ite):
        if typecheck(x):
            for j, y in enumerate(x):
                _return[j, i] = y
        else:
            for j in range(_shape[0]):
                _return[j, i] = x
    return _return


def _1d(ite, dtype):
    """Create 1d array."""
    _shape = len(ite)
    _return = np.zeros(_shape, dtype=dtype)
    for i, x in enumerate(ite):
        if typecheck(x):
            _return[i] = x[0]
        else:
            _return[i] = x
    return _return


def _list_array(ite, dtype=np.float64):
    """Transform list to numpy array of dtype."""
    assert typecheck(ite)
    inner = typecheck(ite[0])
    if inner:
        try:
            _return = _2d(ite, dtype)
        except TypeError as te:
            print(str(te) + '\nNot a 2D array...')
            _return = _1d(ite, dtype)
    else:
        _return = _1d(ite, dtype)
    print('Converted to shape with:', _return.shape)
    return _return


def gaussian_sample(lower_bound, upper_bound, size=100, scale=None):
    """Sample from a gaussian given limits."""
    loc = (lower_bound + upper_bound) / 2.
    if scale is None:
        scale = (upper_bound - lower_bound) / 2.
    results = []
    while len(results) < size:
        samples = np.random.normal(loc=loc, scale=scale,
                                   size=size - len(results))
        results += [sample for sample in samples
                    if lower_bound <= sample <= upper_bound]
    return results


def ecc(a, b):
    """Determine eccentricity given semi-major and semi-minor."""
    return (1. - ((a ** 2) / (b ** 2))) ** 0.5


def mean_anomoly(ecc, tan):
    """Compute the mean anomoly out to 5 taylor terms."""
    """Each line is a new taylor term."""
    mean = tan \
        - (2. * ecc * np.sin(tan))  \
        + ((((3. / 4.) * (ecc ** 2)) +\
            ((1. / 8.) * (ecc ** 4))) * np.sin(2. * tan)) \
        - ((1. / 3.) * (ecc ** 3) * np.sin(3 * tan))  \
        + ((5. / 32.) * (ecc ** 4) * np.sin(4 * tan))
    return mean


def ecc_anomoly(ecc, tan):
    """Find the eccentric anomoly."""
    _tmp = np.tan(tan / 2.) * ((1. - ecc) / (1. + ecc)) ** 0.5
    ecc_a = np.arctan(_tmp) * 2.
    return ecc_a


def period(a, mu):
    """Given the semimajor and Standard grav. parameter, return period."""
    return 2. * pi * ((a ** 3) / mu) ** 0.5


def grav_param(mass):
    """Return the standard grav. param. given mass in solar masses."""
    new_g = 4. * (pi ** 2)  # in AU/solarmass/yr
    new_g = new_g / (365.25 ** 2)  # now in days
    return new_g * mass


def _a2ecc(sma, ecc=0):
    """Given semi major axis and eccentricity, yield semi, minor axis."""
    return ((sma ** 2) / (1. - (ecc ** 2))) ** 0.5


def radians(d):
    """Convert degrees to radians."""
    return d / 180. * pi


def deg(r):
    """Convert radians to degrees."""
    return r * 180. / pi


def rad(a, ecc, nu):
    """Compute the distance given the semi-major, ecc, and true anomoly."""
    """nu should be in degrees."""
    nu = radians(nu)
    return a * (1. - (ecc ** 2)) / (1. + (ecc * np.cos(nu)))


def velocity(r, a, mu):
    """Given distance, semimajor and standard grav param, give the velocity."""
    return (mu * ((2. / r) - (1. / a))) ** 0.5


def mag(a):
    """Compute the magnitude of a vector."""
    ret = 0
    for x in a:
        ret += x
    return x


def orbital_params(lower_smajora, upper_smajora, lower_ecc, upper_ecc,
                   lower_inc, upper_inc, central_mass=1, size=100):
    """Main function to generate samples of orbital params."""
    # compute the standard gravitational parameter
    mu = grav_param(central_mass)

    # sample given params within bounds
    sample_smajora = gaussian_sample(lower_smajora, upper_smajora, size)
    sample_inc = gaussian_sample(lower_inc, upper_inc, size)
    sample_ecc = gaussian_sample(lower_ecc, upper_ecc, size)

    # randomly sample other params
    sample_lan = gaussian_sample(0., 2. * pi, size)
    sample_aop = gaussian_sample(0., 2. * pi, size)
    sample_tan = gaussian_sample(0., 2. * pi, size)
    for i in range(size):
        if (sample_aop[i] + sample_tan[i]) >= (2. * pi):
            sample_tan[i] = (sample_aop[i] + sample_tan[i]) % (2. * pi)
            sample_aop[i] = 0.

    keplerian = _list_array([sample_smajora, sample_inc, sample_ecc,
                 sample_lan, sample_aop, sample_tan, mu], float)

    cartesian = np.zeros((keplerian.shape[0], 6), dtype=float)
    for i in range(size):
        cartesian[i] = orbital_2_xyz(sample_smajora[i], sample_inc[i],
                                     sample_ecc[i], sample_lan[i],
                                     sample_aop[i], sample_tan[i], mu)

    return keplerian, cartesian


def orbital_2_xyz(a, inc, ecc, lan, aop, tan, mu):
    """Function to change orbital elements."""
    """Semi major axis, eccentricity, inclination
    logitude of accending node, argument of pericenter
    true anomoly, and stand. grav. parameter.
    SMA [AU]
    mu [AU^3/day^2]
    returns in AU and AU/day
    """
    # Find Eccentric Anomaly
    ecc_an = ecc_anomoly(ecc, tan)

    nu = 2. * np.arctan(np.sqrt((1. + ecc) / (1. - ecc)) * np.tan(ecc_an / 2.))
    rn = a * (1. - ecc * np.cos(ecc_an))
    h = np.sqrt(mu * a * (1. - (ecc ** 2)))

    # find cartesian components of position
    x = rn * ((np.cos(lan) * np.cos(aop + nu)) -
              np.sin(lan) * np.sin(aop + nu) * np.cos(inc))
    y = rn * ((np.sin(lan) * np.cos(aop + nu)) +
              np.cos(lan) * np.sin(aop + nu) * np.cos(inc))
    z = rn * (np.sin(inc) * np.sin(aop + nu))

    # find cartesian components of velocity
    vx = ((x * h * ecc / (rn * a * (1. - ecc ** 2)) * np.sin(nu)) -
          h / rn * (np.cos(lan) * np.sin(aop + nu) +
                    np.sin(lan) * np.cos(aop + nu) * np.cos(inc)))
    vy = ((y * h * ecc / (rn * a * (1. - ecc ** 2)) * np.sin(nu)) -
          h / rn * (np.sin(lan) * np.sin(aop + nu) -
                    np.cos(lan) * np.cos(aop + nu) * np.cos(inc)))
    vz = ((z * h * ecc / (rn * a * (1. - ecc ** 2)) * np.sin(nu)) +
          h / rn * np.sin(inc) * np.cos(aop + nu))

    return x, y, z, vx, vy, vz


if __name__ == '__main__':
    print('Testing')
    _t = 3.1415926535
    testing1, testing2 = orbital_params(1, 10, 0.1, 0.9, _t / 2., 3. * _t / 4., size=2)

    assert testing1.shape[-1] == 7
    assert testing2.shape[-1] == 6
    print(f'Orbital Params:\nsma, inc, ecc, lan, aop, tan, mu\n{testing1}')
    print(f'Orbital Params (Cart):\nx, y, z, vx, vy, vz\n{testing2}')
    print('Testing complete')

# end of file
