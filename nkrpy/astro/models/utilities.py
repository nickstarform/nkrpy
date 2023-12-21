"""."""
# flake8: noqa
# cython modules
import numpy as np
cimport numpy as np

# internal modules
from copy import deepcopy

# external modules
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.signal import convolve as scipy__convolve

# relative modules
from ...misc import constants
from ..._math._miscmath import width_convolution_kernel
from .._wcs import WCS
from ...misc import FrozenDict
from ...io import fits

cdef make_cyl_grid(WCS wcs, float ra, float dec, float sinpa, float cospa, float sini, float cosi, float cosi2):
        cdef np.ndarray[double, ndim=1] ra_offset, dec_offset
        cdef np.ndarray[double, ndim=2] ra_grid, dec_grid, x_grid, y_grid, x_grid2, y_grid2, r, theta
        cdef np.ndarray[np.uint8_p, ndim=2] mask
        # in deg
        ra_offset = -(wcs.array(return_type='wcs', axis=wcs.axis1['dtype']) - ra)
        dec_offset = (wcs.array(return_type='wcs', axis=wcs.axis2['dtype']) - dec)
        ra_grid, dec_grid = np.meshgrid(ra_offset, dec_offset)
        x_grid = (-ra_grid * cosp - dec_grid * sinp) # disk major axis
        y_grid = (-ra_grid * sinp + dec_grid * cosp) # disk minor axis
        x_grid2 = x_grid ** 2
        y_grid2 = y_grid ** 2

        mask = y_grid != 0

        theta[mask] = 2 * np.arctan((y_grid[mask] / cosi) \
                                 / (x_grid[mask] \
                                    + np.sqrt(x_grid2[mask] + (y_grid2[mask] / cosi2)))) - constants.pi / 2
        return r, theta



cdef np.ndarray[double, ndim=3] convolve_cube(np.ndarray[double, ndim=3] cube, 
                                         np.ndarray[double, ndim=3] convolved_cube,
                                         double bma, double bmi, double bpa):
    return convolved_cube

cdef make_intensity(np.ndarray[double, ndim=2] intensity,
                                               np.ndarray[double, ndim=2] r,
                                               double rin,
                                               double rout, char profile,
                                               double fwhm, float pwrlaw_exp,, float pwrlaw_rc, double r_taper, double fwhm2sigma):
    cdef np.ndarray[np.uint8_p, ndim=2] valid_radii
    valid_radii = (r >= rin) * (r <= rout)
    if profile == 'gaussian':
        """Gaussian Intensity
        
        Fr = f0 * e ^ -(r - r_inner) / 2s ** 2 | cutoff at r_outer
        """
        intensity[valid_radii] = np.exp(-r[valid_radii] ** 2 /(2 * fwhm2sigma ** 2))
    elif profile == 'powerlaw':
        """Powerlaw Intensity

        Fr = F0 * (r - r_inner) ^ i | cutoff at r_outer
        """
        intensity[valid_radii] = r[valid_radii] ** pwrlaw_exp
    elif profile == 'ring':
        """Constant Flux Density Ring

        Fr = f0 between disk_radii_inner and disk_radii_outer
        """
        intensity[valid_radii] = 1
    if r_taper < rout:
        """Taper disk

        Fr = f0 between disk_radii_inner and disk_radii_outer
        """
        taper_radii = (r >= r_taper) * valid_radii
        intensity[taper_radii] *= np.exp(-1 * (r[taper_radii] / r_taper) ** (2. - pwrlaw_exp))
    return intensity


cdef public np.ndarray[double, ndim=3] make_cube(np.ndarray[double, ndim=2] intensity,
                                          np.ndarray[double, ndim=3] velocities,
                                          np.ndarray[double, ndim=3] vproj,
                                          double vsys, double sigma_linewidth):
    cdef np.ndarray[double, ndim=3] cube
    cube = np.broadcast_to(intensity, shape=(3, intensity.shape[0], intensity.shape[1])) * np.exp(-(velocities - vproj - vsys)**2 / (2 * sigma_linewidth ** 2))
    return cube


cdef public np.ndarray[double, ndim=3] make_velocity_profile(np.ndarray[double, ndim=3] r,
                                          np.ndarray[double, ndim=3] vtheta,
                                          np.ndarray[double, ndim=3] vr,
                                          double mstar,
                                      double rc, char profile):
    # valid for all r < rc and r != 0
    cdef np.ndarray[double, ndim=3] mask, gmm_r, gmmrc_r2
    cdef double gmmstar, gmmstar_rc
    gmmstar = constants.g * constants.msun * mstar
    gmmstar_rc = constants.g * constants.msun * mstar * rc
    mask = (r < rc) * (r != 0)
    if mask.sum() > 0:
        gmm_r = gmmstar / r[mask]
        gmmrc_r2 = gmmstar_rc / (r[mask] ** 2)
        if 'infall' == profile:
            vr[mask] = np.sqrt(2 *  gmm_r - gmmrc_r2) # in cm/s
            vtheta[mask] += np.sqrt(gmm_r * rc) # assume Keplerian rotation within rc
        if 'keplerian' == profile:
            vtheta[mask] = np.sqrt(gmm_r) # assume Keplerian rotation within rc
    return vtheta, vr, mask

cdef public np.ndarray[double, ndim=3] make_velocity_proj(
                                          np.ndarray[double, ndim=3] vproj,
                                          np.ndarray[double, ndim=3] vtheta,
                                          np.ndarray[double, ndim=3] theta,
                                          np.ndarray[double, ndim=3] vr,
                                          np.ndarray[double, ndim=3] r,
                                          double sin_i):
    cdef np.ndarray[double, ndim=3] mask
    mask = r != 0
    vproj[mask] = (np.sin(theta[mask]) * vtheta[mask] + np.cos(theta[mask]) * vr[mask]) * sin_i * 1e-5 # in cm / s to km/s
    return vproj
