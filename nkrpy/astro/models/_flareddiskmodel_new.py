"""."""
# flake8: noqa
# cython modules

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

# global attributes
__all__ = ['FlaredDiskModel']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)



INTENSITY_LAWS = ['gaussian', 'powerlaw', 'uniform']
VELOCITY_PROFILES = ['keplerian', 'infall']


FlaredDiskModel_kwargs = dict(
                 # some base params
                 noise = None, # must be in flux density / beam
                 fwhm_linewidth=None, # in km/s
                 wcs=None, # can take a nkrpy.astro.WCS object instead of the image params
                 # image params
                 beam=[None, None, None], # beam params in arcsec jy/beam Noise, bma, bmi, bpa
                 restfreq = None, # line params restfreq in hz, restfreq assumed at nchan center
                 # system params
                 mstar=1, peakint=None, vsys=None, # in msun, intensity/beam, km/s, 
                 ra=None, 
                 dec=None, # center in deg coordinates
                 distance_pc=None,
                 # disk params all in deg
                 inclination=None, # defined 0 is faceon in degrees
                 position_angle=None, # defined east of north Blue in degrees
                 radial_intensity_r_inner=None, # Truncates the inner of the disk
                 radial_intensity_r_truncate=None, # hard truncates the disk
                 radial_intensity_rc_taper=None, # begins truncate at rc w/ exp decay
                 disk_intensity_law='gaussian',
                 radial_intensity_gaussian_fwhm=None, # eqn e^r^2/2sig^2
                 radial_intensity_pwrlw_pwr=None,
                 radial_intensity_pwrlw_rc=None, # eqn (r/rc) ** -pwr
                 # envelope param, assume sphere
                 envelope_velocity_profile='infall',
                 envelope_intensity_law='gaussian',
                 disk_to_envelope_velocity_profile_rc=None, # eqn vr = sqrt(2G*mstar/r - (G*mstar*rc/r**2)) and vtheta = sqrt(2G*mstar*rc/r))
                 envelope_r_truncate=None,
                 envelope_flux=None, # in intensity/beam
)


class FlaredDiskModel():
    # magic sets that define what keys will change a certain grid
    __refresh_wcs = {'wcs'}
    __refresh_angles = {'position_angle', 'inclination'}
    __remake_cyl_grid = {'wcs', 'ra', 'dec', 'position_angle', 'inclination'}
    __remake_diskint_image = {'radial_intensity_r_inner', 'radial_intensity_r_truncate',
                          'radial_intensity_pwrlw_rc',
                          'radial_intensity_gaussian_fwhm', 'radial_intensity_pwrlw_pwr',
                          'disk_intensity_law', 'radial_intensity_rc_taper',
                          *__remake_cyl_grid}
    __remake_envint_image = {'envelope_r_truncate', 'envelope_intensity_law'}
    __remake_vel_grid = {'mstar', *__remake_cyl_grid}
    __remake_velproj = {*__remake_cyl_grid, *__remake_vel_grid}
    __remake_noise = {'noise','wcs'}
    __remake_cube = {'vsys', 'fwhm_linewidth', *__remake_vel_grid, *__remake_vel_grid, *__remake_int_image}
    __remake_scaled_cube = {'peakint', *__remake_noise, *__remake_cube}
    __remake_convolved_cube = {'beam', *__remake_scaled_cube}
    __remake_kernel = {'wcs', 'beam'}

    def __init__(self, **kwargs):
        assert kwargs.keys() == FlaredDiskModel_kwargs.keys()
        __fwhm2sig = 1 / (2 * np.sqrt(2 * np.log(2)))
        __log2 = np.log(2)
        __deg2rad = constants.pi / 180.
        __gmsun = constants.g * constants.msun
        self.__fwhm2sig = __fwhm2sig
        self.__log2 = __log2
        self.__deg2rad = __deg2rad
        self.__gmsun = __gmsun
        self.kwargs = kwargs
        self.__new_params = set(kwargs.keys())
        self.solveWCSparams()

    def __compare_keys(self, kwargs, key):
        if kwargs[key] is None:
            return False
        if kwargs[key] == self.kwargs[key]:
            return False
        return True

    def __call__(self, **kwargs):
        # solve for any new keys
        if len(kwargs) > 0:
            self.__new_params = []
            resolved_kwargs = {**FlaredDiskModel_kwargs, **kwargs}
            for key in FlaredDiskModel_kwargs.keys():
                if self.__compare_keys(resolved_kwargs, key):
                    self.__new_params.append(key)
            self.__new_params = set(self.__new_params)
        else:
            kwargs = {}
        if len(self.__new_params) > 0:
            self.kwargs = {**self.kwargs, **kwargs}

        # now handle each case
        if len(self.__refresh_wcs - self.__new_params) != len(self.__refresh_wcs):
            self.refresh_wcs()
        if len(self.__refresh_angles - self.__new_params) != len(self.__refresh_angles):
            self.refresh_angles()
        if len(self.__remake_cyl_grid - self.__new_params) != len(self.__remake_cyl_grid):
            # remake cyclindrical grid
            self.__make_cyl_grid()
        if len(self.__remake_int_image - self.__new_params) != len(self.__remake_int_image):
            # remake cyclindrical grid
            self.__make_int_image()
        if len(self.__remake_vel_grid - self.__new_params) != len(self.__remake_vel_grid):
            # remake cyclindrical grid
            self.__make_vel_grid()
        if len(self.__remake_velproj - self.__new_params) != len(self.__remake_velproj):
            # remake cyclindrical grid
            self.__make_velproj()
        if len(self.__remake_noise - self.__new_params) != len(self.__remake_noise):
            # remake cyclindrical grid
            self.__make_noise()
        if len(self.__remake_kernel - self.__new_params) != len(self.__remake_kernel):
            # remake cyclindrical grid
            self.__make_kernel()
        if len(self.__remake_cube - self.__new_params) != len(self.__remake_cube):
            # remake cyclindrical grid
            self.__make_cube()
        if len(self.__remake_scaled_cube - self.__new_params) != len(self.__remake_scaled_cube):
            # remake cyclindrical grid
            self.__make_scaled_cube()
        if len(self.__remake_convolved_cube - self.__new_params) != len(self.__remake_convolved_cube):
            # remake cyclindrical grid
            self.__make_convolved_cube()
        self.__new_params = {}
        pass

    def refresh_wcs(self):
        imsize = wcs.axis1['axis']
        channelwidth=wcs.axis3['delt'] # in hz
        cellsize=abs(wcs.axis1['delt']) # in deg
        numchans=wcs.axis3['axis'] # cube params
        _noise_image = np.ones((numchans, imsize, imsize), dtype=float)
        __ones_image = np.ones((imsize, imsize), dtype=float)
        _theta = __ones_image.copy() - 1 # theta = 0 along the l.o.s.
        _disk_intensity = __ones_image.copy() - 1
        _envelope_intensity = __ones_image.copy() - 1
        _vtheta = __ones_image.copy() - 1
        _vr = __ones_image.copy() - 1
        _vproj = __ones_image.copy() - 1
        _cube = _noise_image.copy() - 1
        _convolve_cube = _noise_image.copy() -1
        self.__ones_image = __ones_image
        self._theta = _theta
        self._disk_intensity = _disk_intensity
        self._envelope_intensity = _envelope_intensity
        self._cube = _cube
        self._convolve_cube = _convolve_cube
        self._vtheta = _vtheta
        self._vr = _vr
        self._vproj = _vproj

        restfreq = wcs.get_head('restfrq') # line params restfreq in hz, fwhm in km/s
        _velocities = ((restfreq - self.wcs.array(return_type='wcs', axis=self.wcs.axis3['dtype'])) / self.restfreq * constants.c * 1e-5)[..., np.newaxis, np.newaxis]# in cm / s to km/s
        self.restfreq = restfreq
        self.deg2cm =  3600 * self.distance_pc * constants.au
        self.pixel2cm =  self.cellsize * self.deg2cm
        pass


    def refresh_angles(self):
        self.kwargs['_pa_rad'] = self.kwargs['position_angle'] * self.__deg2rad
        self.kwargs['_inc_rad'] = self.kwargs['inclination'] * self.__deg2rad
        self.kwargs['_sinpa'] = np.sin(self.kwargs['_pa_rad'])
        self.kwargs['_cospa'] = np.cos(self.kwargs['_pa_rad'])
        self.kwargs['_sini'] = np.sin(self.kwargs['_inc_rad'])
        self.kwargs['_cosi'] = np.cos(self.kwargs['_inc_rad'])
        self.kwargs['_cosi2'] = self.kwargs['_cosi'] ** 2
        pass

    def __make_cyl_grid(self):
        r, theta = make_cyl_grid(self.kwargs['wcs'],
            self.kwargs['ra'], self.kwargs['dec'],
            self.kwargs['sinpa'], self.kwargs['cospa'],
            self.kwargs['sini'], self.kwargs['cosi'],
            self.kwargs['cosi2'])
        self.kwargs['_r'] = r
        self.kwargs['_theta'] = theta
        pass
    def __remake_diskint_image(self):
        make_intensity(self.disk_intensity, self.kwargs['_r'],
            self.kwargs['r_inner'], self.kwargs['r_outer'],
            self.kwargs['disk_intensity_law'],
            self.kwargs['radial_intensity_gaussian_fwhm'],
            self.kwargs['radial_intensity_pwrlw_pwr'],
            self.kwargs['radial_intensity_pwrlw_rc'],
            self.kwargs['rc_taper'], self.__fwhm2sig)
        pass
    def __remake_envint_image(self):
        make_intensity(self.envelope_intensity, self.kwargs['_r'],
            self.kwargs['r_inner'], self.kwargs['r_outer'],
            self.kwargs['disk_intensity_law'],
            self.kwargs['radial_intensity_gaussian_fwhm'],
            self.kwargs['radial_intensity_pwrlw_pwr'],
            self.kwargs['radial_intensity_pwrlw_rc'],
            self.kwargs['rc_taper'], self.__fwhm2sig)
        pass
    def __make_vel_grid(self):
        pass
    def __make_velproj(self):
        pass
    def __make_noise(self):
        pass
    def __make_kernel(self):
        pass
    def __make_cube(self):
        pass
    def __make_scaled_cube(self):
        pass
    def __make_convolved_cube(self):
        pass



    def refresh_params(self):
        self._pa = (-self.position_angle + 90) * self.__deg2rad
        self._inc = self.inclination * self.__deg2rad
        self._sinp = np.sin(self._pa)
        self._cosp = np.cos(self._pa)
        self._cosi = np.cos(self._inc)
        self._sini = np.sin(self._inc)
        self._cosi2 = self._cosi**2
        self.deg2cm =  3600 * self.distance_pc * constants.au
        self.pixel2cm =  self.cellsize * self.deg2cm
        self.r_inner_cm = self.deg2cm * self.r_inner
        self.r_truncate_cm = self.deg2cm * self.r_truncate
        self.rc_taper_cm = self.deg2cm * self.rc_taper
        self.radial_intensity_pwrlw_rc_cm = self.deg2cm * self.radial_intensity_pwrlw_rc
        self.velocity_profile_rc_cm = self.deg2cm * self.velocity_profile_rc
        self.envelope_r_truncate_cm = self.deg2cm * self.envelope_r_truncate
        self.radial_intensity_gaussian_fwhm_cm = self.deg2cm * self.radial_intensity_gaussian_fwhm
        self.radial_intensity_gaussian_sigma = self.radial_intensity_gaussian_fwhm_cm * self.__fwhm2sig
        self.sigma_linewidth = self.fwhm_linewidth * self.__fwhm2sig

    def solveWCSparams(self, wcs=None):
        if hasattr(self, 'wcs'):
            wcs = self.wcs
        else:
            self.wcs = wcs
        self.imsize = wcs.axis1['axis']
        self._imsize_ar = np.arange(self.imsize)
        self.numchans=wcs.axis3['axis'] # cube params
        _noise_image = np.ones((self.numchans, self.imsize, self.imsize), dtype=float)
        self._noise_image = _noise_image - 1
        if self.noise is not None:
            self._noise_image = np.random.normal(loc=0, scale=self.noise * self.__fwhm2sig / 2, size=self._noise_image.shape)
        beam=wcs.get_beam() # beam params in arcsec jy/beam Noise, bma, bmi, bpa
        restfreq = wcs.get_head('restfrq') # line params restfreq in hz, fwhm in km/s
        self.channelwidth=wcs.axis3['delt'] # in hz
        self.cellsize=abs(wcs.axis1['delt']) # in deg
        self.centerpix[1] = wcs(self.centerwcs[1], 'pix', wcs.axis2['dtype'])
        self.centerpix[0] = wcs(self.centerwcs[0], 'pix', wcs.axis1['dtype'], declination_degrees=self.centerwcs[1])
        self.centerwcs[0] = wcs(self.imsize / 2, 'wcs', wcs.axis1['dtype'], declination_degrees=self.centerwcs[1])
        self.centerwcs[1] = wcs(self.imsize / 2, 'wcs', wcs.axis2['dtype'])
        if all([b is not None for b in beam]):
            self.beam = beam
        if restfreq is not None:
            self.restfreq = restfreq
        self.beam2pixel =  constants.pi * self.beam[0] * self.beam[1] / (4 * self.__log2) / self.cellsize ** 2

        _velocities = ((self.restfreq - self.wcs.array(return_type='wcs', axis=self.wcs.axis3['dtype'])) / self.restfreq * constants.c * 1e-5)[..., np.newaxis, np.newaxis]# in cm / s to km/s
        self._velocities = _velocities
        pass

    def writeFits(self, filename):
        if not filename.endswith('.fits'):
            filename += '.fits'
        if hasattr(self, 'convolved_cube'):
            data = self.convolved_cube
        elif hasattr(self, 'cube'):
            data = self.cube
        else:
            data = self.__ones_image
        header = self.wcs.create_fitsheader_from_axes()
        fits.write(f=filename, data=data, header=header)
        pass

    def __solve_kernel(self, bma, bmi, bpa):
        bma, bmi = map(lambda x: x * self.__fwhm2sig / self.cellsize, (bma, bmi))
        smoothing_kernel = Gaussian2DKernel(x_stddev=bmi, y_stddev=bma, theta=bpa * self.__deg2rad).array[np.newaxis, ...]
        self.__kernel = smoothing_kernel
        pass

    def __convolve(self, cube):
        # beam params assumed to be pixels
        #Creating a Gaussian smoothing kernel. 
        # astropy
        # convolved_cube = convolve(array = cube, kernel=self.__kernel)
        kernel = self.__kernel
        convolved_cube = scipy__convolve(in1 = cube, in2=kernel, mode='same', method='fft')
        return convolved_cube

    def convolve_beam(self): # convolve beam into cube
        # if you  want to change the conv then change the beam
        # beam params assumed in deg
        self.__solve_kernel(*self.beam)
        if not hasattr(self, '__kernel'):
            self.__solve_kernel(*self.beam)

        self.convolved_cube = self.__convolve(self.cube)
        # kind of an adhoc treatment
        '''
        cdef double bma, bmi
        cdef float bpa, a, width
        cdef int i, channels, b_int
        kernel3D 
        gaussian2DKernel
        cdef double amp
        if self.inclination > 80:
            if bma is None:
                bma, bmi, bpa = self.beam
            a = -2
            b = bma * self.__fwhm2sig
            b_int = round(b, 0)
            channels = self.cube.shape[0]
            kernel3D = np.zeros((channels, b_int, b_int), dtype=float)
            for i in prange(channels, nogil=True):
                i_med = i + 1 - channels / 2
                width = b + i_med * a
                amp = 1 / 2 * width ** 0.5
                gaussian2DKernel = amp * np.exp(- ((()/()) ** 2 + (()/()) ** 2))
                kernel3D[i, ...] = Gaussian2DKernel(x_stddev=width, y_stddev=width, theta=0, x_size=b_int, y_size=b_int)
            self.__kernel = kernel3D
            self.convolved_cube = self.__convolve(cube = self.convolved_cube)
        '''
        return self.convolved_cube

    def __make_cube(self):
        # in pixels
        ra_offset = -(self._imsize_ar - self.centerpix[0])
        dec_offset = (self._imsize_ar - self.centerpix[1])
        ra_grid, dec_grid = np.meshgrid(ra_offset, dec_offset)
        x_grid = (-ra_grid * self._cosp - dec_grid * self._sinp) # disk major axis
        y_grid = (-ra_grid * self._sinp + dec_grid * self._cosp) # disk minor axis
        x_grid2 = x_grid ** 2
        y_grid2 = y_grid ** 2

        mask = y_grid != 0

        # now pull from local def
        theta = self._theta
        vproj = self._vproj
        vtheta = self._vtheta
        vr = self._vr
        disk_intensity = self._disk_intensity
        envelope_intensity = self._envelope_intensity


        theta[mask] = 2 * np.arctan((y_grid[mask] / self._cosi) \
                                 / (x_grid[mask] \
                                    + np.sqrt(x_grid2[mask] + (y_grid2[mask] / self._cosi2)))) - constants.pi / 2
        r = np.sqrt(x_grid2 + (y_grid2 / self._cosi2)) #  in pixels
        r *= self.pixel2cm # in cm

        ########## DISK
        rin = self.r_inner_cm  # inner cutoff
        rout = self.r_truncate_cm # outer cutoff
        rc = self.velocity_profile_rc_cm # break radius for velocity
        valid_radii =(r >= rin) * (r <= rout)
        if self.radial_intensity_law == 'gaussian':
            """Gaussian Intensity
            
            Fr = f0 * e ^ -(r - r_inner) / 2s ** 2 | cutoff at r_outer
            """
            disk_intensity[valid_radii] = np.exp(-r[valid_radii] ** 2 /(2 * self.radial_intensity_gaussian_sigma ** 2))
        elif self.radial_intensity_law == 'powerlaw':
            """Powerlaw Intensity

            Fr = F0 * (r - r_inner) ^ i | cutoff at r_outer
            """
            disk_intensity[valid_radii] = r[valid_radii] ** self.radial_intensity_pwrlw_pwr
        elif self.radial_intensity_law == 'ring':
            """Constant Flux Density Ring

            Fr = f0 between disk_radii_inner and disk_radii_outer
            """
            disk_intensity[valid_radii] = 1
        if self.rc_taper_cm < self.r_truncate_cm:
            """Taper disk

            Fr = f0 between disk_radii_inner and disk_radii_outer
            """
            taper_radii = (r >= self.rc_taper_cm) * (r <= self.r_truncate_cm)
            disk_intensity[taper_radii] = 0.5  - np.arctan(r[taper_radii] - self.rc_taper_cm) / constants.pi
        disk_intensity *= self.peakint / disk_intensity.max()



        gmmstar = self.__gmsun * self.mstar
        gmmstar_rc = self.__gmsun * self.mstar * rc

        mask = (r < rc) * (r != 0)
        if mask.sum() > 0:
            gmm_r = gmmstar / r[mask]
            gmmrc_r2 = gmmstar_rc / (r[mask] ** 2)
            if 'infall' == self.disk_velocity_profile:
                vr[mask] += np.sqrt(2 *  gmm_r - gmmrc_r2) # in cm/s
            if 'keplerian' == self.disk_velocity_profile:
                vtheta[mask] += np.sqrt(gmm_r) # assume Keplerian rotation within rc

        '''
        # envelope portion of velocity
        mask = r >= rc
        if mask.sum() > 0:
            gmm_r = gmmstar  / r[mask]
            gmmrc_r2 = gmmstar_rc / (r[mask] ** 2)
            if 'infall' == self.envelope_velocity_profile:
                vr[mask] = np.sqrt(2 * gmm_r - gmmrc_r2)
            if 'keplerian' == self.envelope_velocity_profile:
                vtheta[mask] = np.sqrt(gmm_r)
        '''

        mask = r != 0
        # sin cos flipped?
        #                                RADIAL                         TANGENTIAL
        vproj[mask] = (np.sin(theta[mask]) * vtheta[mask] + np.cos(theta[mask]) * vr[mask]) * self._sini * 1e-5 # in cm / s to km/s
        
        '''
        ######### ENVELOPE
        if self.envelope_r_truncate > 0:
            """Spherical envelope
            # an analytic column density profile fit to prestellar cores by dapp and basu 2018

            Fr = f0 between disk_radii_inner and disk_radii_outer
            """
            env_radii = (r > self.velocity_profile_rc) * (r < self.envelope_r_truncate)
            envelope_intensity[env_radii] += self.envelope_sph_flux
        '''
        # now create cube VELOCITY, RA, DEC
        self.cube = (disk_intensity + envelope_intensity)[np.newaxis,...] * np.exp(-(self._velocities - vproj - self.vsys)**2 / (2 * self.sigma_linewidth ** 2))
        # now disk portion
        # make projections
        if self.noise is not None and self.noise > 0:
            disk_intensity += self._noise_image[0, ...]
            self.cube += self._noise_image
        self.intensityimage = disk_intensity + envelope_intensity
        pass

