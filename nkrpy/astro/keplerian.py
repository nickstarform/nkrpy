"""Find Keplerian Parameters."""

# standard modules


# external modules
import numpy as np
from numpy import arccos as acos
from numpy.linalg import norm

# relative modules
from ..misc.constants import pi
from ..math import (list_array, cross, dot, radians,
                    gaussian_sample)

__all__ = ('orbital_params', 'orbital_2_xyz', 'mean_anomoly',
           'eccentricity_vector', 'ecc_anomoly', 'keplerian_velocity',
           'xyz_2_orbital', 'specific_orbital_energy')

__doc__ = """orbital_params(lsma,usma,le,ue,li,ui,mass,size).
Use orbital_params or orbital_2_xyz as the main function call.
For former: input distribution params and generates a gaussian distribution of
orbital params in both cartesian and the orbital elements.
The orbital_2_xyz converts orbital elements to xyz components.
Keep everything in AU, days, solar masses, radians"""

# Acceptable Numerical Error
eps = 1E-15


def eccentricity_vector(position, velocity, mu):
    """Return eccentricity vector.

    :param position: Position (r) [AU]
    :param velocity: Velocity (v) [AU/day]
    :return: Eccentricity vector (ev) [-]
    """
    ev = 1. / mu * ((norm(velocity) ** 2 - mu / norm(position)) *
                    position - dot(position, velocity) * velocity)
    return ev


def mean_anomoly(ecc, tan):
    """Compute the mean anomoly out to 5 taylor terms.

    Each line is a new taylor term.
    """
    mean = tan \
        - (2. * ecc * np.sin(tan))  \
        + ((((3. / 4.) * (ecc ** 2)) +
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


def rad(a, ecc, nu):
    """Compute the distance given the semi-major, ecc, and true anomoly.

    nu should be in degrees.
    """
    nu = radians(nu)
    return a * (1. - (ecc ** 2)) / (1. + (ecc * np.cos(nu)))


def keplerian_velocity(r, a, mu):
    """Given distance, semimajor and standard grav param, give the velocity."""
    return (mu * ((2. / r) - (1. / a))) ** 0.5


def orbital_params(lower_smajora, upper_smajora, lower_ecc, upper_ecc,
                   lower_inc, upper_inc, central_mass=1, size=100):
    """Generate samples of orbital params."""
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

    keplerian = list_array([sample_smajora, sample_inc, sample_ecc,
                            sample_lan, sample_aop, sample_tan, mu], float)

    cartesian = np.array([sample_smajora, sample_inc,
                          sample_ecc, sample_lan,
                          sample_aop, sample_tan,
                          np.full(keplerian.shape[0], mu, np.float)])
    cartesian = orbital_2_xyz(cartesian)
    return keplerian, cartesian


def orbital_2_xyz(params):
    """Change orbital elements.

    Semi major axis, eccentricity, inclination
    logitude of accending node, argument of pericenter
    true anomoly, and stand. grav. parameter.
    SMA [AU]
    mu [AU^3/day^2]
    returns in AU and AU/day
    """
    # Find Eccentric Anomaly
    params = np.array(params)
    assert len(params.shape) == 2
    assert params.shape[-1] == 7
    ret = np.zeros([params.shape[0], 6], dtype=float)

    for i, x in enumerate(params):
        a, inc, ecc, lan, aop, tan, mu = x
        ecc_an = ecc_anomoly(ecc, tan)

        nu = 2. * np.arctan(np.sqrt((1. + ecc) /
                                    (1. - ecc)) *
                            np.tan(ecc_an / 2.))
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

        ret[i] = np.array([x, y, z, vx, vy, vz])
    return ret


def specific_orbital_energy(position, velocity, mu):
    """Return specific orbital energy.

    :param position: Position (r) [AU]
    :param velocity: Velocity (v) [AU/day]
    :param mass: Central mass [Msun]
    :return: Specific orbital energy (E) [J/kg]
    """
    return norm(velocity) ** 2 / 2. - mu / norm(position)


def xyz_2_orbital(params, mass=None):
    """Change orbital elements.

    :param x,y,z: Position vector [AU]
    :param vx,vy,vz: Velocity vector [AU/day]
    :param mass: Central mass [Msun]
    :return a, e, i, lan, aop, f, mu
    """
    params = np.array(params)
    assert len(params.shape) == 2
    assert params.shape[-1] >= 6
    ret = np.zeros([params.shape[0], 7], dtype=float)
    for ite, p in enumerate(params):
        if p.shape == 7:
            x, y, z, vx, vy, vz, mass = p
        else:
            x, y, z, vx, vy, vz = p
        r, v = np.array([x, y, z]), np.array([vx, vy, vz])
        mu = grav_param(mass)
        h = cross(r, v)
        n = cross([0, 0, 1], h)
        ev = eccentricity_vector(r, v, mu)
        E = specific_orbital_energy(r, v, mu)
        a = -mu / (2. * E)
        e = norm(ev)
        # Inc.: angle between the angular momentum and its z component.
        i = acos(h[-1] / norm(h))

        if abs(i - 0) < eps:
            # For non-inclined orbits, raan is undefined;
            # set to zero by convention
            raan = 0
            if abs(e - 0) < eps:
                # For circular orbits, place periapsis
                # at ascending node by convention
                aop = 0
            else:
                # Argument of periapsis is the angle between
                # eccentricity vector and its x component.
                aop = acos(ev[0] / norm(ev))
        else:
            # Right ascension of ascending node is the angle
            # between the node vector and its x component.
            raan = acos(n[0] / norm(n))
            if n[1] < 0:
                raan = 2. * pi - raan

            # Argument of periapsis is angle between
            # node and eccentricity vectors.
            try:
                aop = acos(dot(n, ev) / (norm(n) * norm(ev)))
            except Exception:
                aop = 0.0

        if abs(e - 0) < eps:
            if abs(i - 0) < eps:
                # True anomaly is angle between position
                # vector and its x component.
                f = acos(r[0] / norm(r))
                if v[0] > 0:
                    f = 2. * pi - f
            else:
                # True anomaly is angle between node
                # vector and position vector.
                f = acos(dot(n, r) / (norm(n) * norm(r)))
                if dot(n, v) > 0:
                    f = 2. * pi - f
        else:
            if ev[-1] < 0:
                aop = 2. * pi - aop

            # True anomaly is angle between eccentricity
            # vector and position vector.
            f = acos(dot(ev, r) / (norm(ev) * norm(r)))

            if dot(r, v) < 0:
                f = 2. * pi - f

        ret[ite] = np.nan_to_num(np.array([a, i, e, raan, aop, f, mu]))
    return ret

# end of file
