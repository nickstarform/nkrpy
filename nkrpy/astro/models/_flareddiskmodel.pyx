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



INTENSITY_LAWS = ['gaussian', 'powerlaw']
VELOCITY_PROFILES = ['keplerian', 'infall']


FlaredDiskModel_kwargs = dict(
                 # some base params
                 noise = 0, # must be in flux density / beam
                 fwhm_linewidth=0, # in km/s
                 wcs=None, # can take a nkrpy.astro.WCS object instead of the image params
                 # image params
                 imsize = 0, numchans=0, # cube params
                 beam=[0, 0, 0], # beam params in arcsec jy/beam Noise, bma, bmi, bpa
                 restfreq = 0, # line params restfreq in hz, restfreq assumed at nchan center
                 channelwidth=0, # in hz 
                 cellsize=0, # in deg
                 # system params
                 mstar=1, peakint=0, vsys=0, # in msun, intensity/beam, km/s, 
                 channel_weights=0, # must be the same shape as the WCS input image
                 centerwcs=[0, 0], # center in deg coordinates
                 distance_pc=0,
                 # disk params all in deg
                 inclination=0, # defined 0 is faceon in degrees
                 position_angle=0, # defined east of north Blue in degrees
                 r_inner=0, # Truncates the inner of the disk
                 r_truncate=0, # hard truncates the disk
                 radial_intensity_law='gaussian',
                 radial_intensity_gaussian_fwhm=0, # eqn e^r^2/2sig^2
                 radial_intensity_pwrlw_pwr=0,
                 radial_intensity_pwrlw_rc=0, # eqn (r/rc) ** -pwr
                 velocity_profile=['keplerian'], 
                 # envelope param
                 velocity_profile_rc=0, # eqn vr = sqrt(2G*mstar/r - (G*mstar*rc/r**2)) and vtheta = sqrt(2G*mstar*rc/r))
)

class BaseFlaredDiskModel:
    pass

class FlaredDiskModel(BaseFlaredDiskModel):
    # handle everything in pixels, every parameter must be declared
    def __init__(self, *args, **kwargs):
        __fwhm2sig = 1 / (2 * np.sqrt(2 * np.log(2)))
        __log2 = np.log(2)
        __deg2rad = constants.pi / 180.
        __gmsun = constants.g * constants.msun
        self.__fwhm2sig = __fwhm2sig
        self.__log2 = __log2
        self.__deg2rad = __deg2rad
        self.__gmsun = __gmsun
        superkwargs = {**FlaredDiskModel_kwargs, **kwargs}
        if len(args) > 0:
            masterkeys = list(FlaredDiskModel_kwargs.keys())
            for i, v in enumerate(args):
                superkwargs[masterkeys[i]] = v
        self.kwargs = [superkwargs]
        self.set_vars(**superkwargs)
        self.reset_params()
        self.refresh_params()
        self.intensityimage = 0
        self.cube = 0

    def set_vars(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, resolve_wcs = False, **kwargs):
        if kwargs:
            self.set_vars(**kwargs)
            if resolve_wcs:
                self.solveWCSparams(kwargs)
            self.refresh_params()
        if not hasattr(self, '_pa'):
            self.refresh_params()
        self.__make_cube()

    def reset_params(self):
        superkwargs = self.kwargs[0]
        self.set_vars(**superkwargs)
        self.solveWCSparams(superkwargs)
        __ones_image = np.ones((self.imsize, self.imsize), dtype=float)
        _theta = __ones_image.copy() - 1 # theta = 0 along the l.o.s.
        _intensity = __ones_image.copy() - 1
        _vtheta = __ones_image.copy() - 1
        _vr = __ones_image.copy() - 1
        _vproj = __ones_image.copy() - 1
        self.__ones_image = __ones_image
        self._theta = _theta
        self._intensity = _intensity
        self._vtheta = _vtheta
        self._vr = _vr
        self._vproj = _vproj

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
        self.radial_intensity_pwrlw_rc_cm = self.deg2cm * self.radial_intensity_pwrlw_rc
        self.velocity_profile_rc_cm = self.deg2cm * self.velocity_profile_rc
        self.radial_intensity_gaussian_fwhm_cm = self.deg2cm * self.radial_intensity_gaussian_fwhm
        self.radial_intensity_gaussian_sigma = self.radial_intensity_gaussian_fwhm_cm * self.__fwhm2sig
        self.sigma_linewidth = self.fwhm_linewidth * self.__fwhm2sig

    def solveWCSparams(self, wcs=None):
        if hasattr(self, 'wcs'):
            wcs = self.wcs
        elif isinstance(wcs, nkrpy.WCSClass):
            self.wcs = wcs
        elif wcs is not None:
            wcs = WCS(**wcs)
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

    @staticmethod
    def rotation(xyz, pa_radians=0, inc_radians=0):
        '''Rotation

        Rotates a 3d meshgrid by euler angles
        Everything should be constructed before this and then rotate
        xyz: 3D array of meshgrid XYZ stacked, so 3xMxN
        y is north, z is along LoS
        '''
        if inc_radians == 0 and pa_radians == 0:
            return xyz
        cpa, spa = np.cos(pa_radians), np.sin(pa_radians)
        cinc, sinc = np.cos(inc_radians), np.sin(inc_radians)
        mp = np.array([[cpa, spa, 0], [-spa, cpa, 0], [0, 0, 1]])
        mi = np.array([[cinc, 0, sinc], [0, 1, 0], [-sinc, 0, cinc]])
        t = np.transpose(xyz, (1,2,0))
        if inc_radians != 0:
            t = np.dot(t, mi)
            x,y,z = np.transpose(t, (2,0, 1))
        if pa_radians != 0:
            t = np.dot(t, mp)
            x,y,z = np.transpose(t, (2,0, 1))
        return np.array([x,y,z])


    @staticmethod
    def __make_grid(size):
        # y is north, z is along LoS
        # makes coordinate grid
        size += size % 2 + 1
        x = np.arange(0, size, 1, dtype=int)
        x -= (size - 1) / 2
        X,Y,Z = np.meshgrid(x, x, x)
        R = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
        Phi = np.arccos(Y / R) # angle from y axis
        Theta = np.arccos(Z / X) # angle from LoS
        Flux = np.zeros(R.shape, dtype=float)
        return ((X, Y, Z), (R, Phi, Theta))

    @staticmethod
    def make_surface_density_disk(XYZ, Sph, Ring_kwargs={}, Gaussian_kwargs = {}, Pwrlaw_kwargs = {}):
        '''Disk Surface Creator

        xyz: 3D array of meshgrid XYZ stacked, so 3xMxN
        y is north, z is along LoS
        '''
        if Gaussian_kwargs:

    @staticmethod
    def __make_coordinate_cube_intensity_model(self, ):
        phi = np.linspace(0, 2 * np.pi, 1000)
        rho = np.linspace(0, np.pi, 1000)



        x,y,z = self._x, self._y, self._z # y, z are in sky plane, x is along LoS
        flux = self._flux_cube
        r2 = x ** 2 + y ** 2 + z ** 2
        r = r2 ** 0.5
        theta = np.arctan2(y, x)
        phi = np.arccos(z, r) # east of north
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        if envelope:
            # sma, smi, rin, rout, pa
            a2 = smi ** 2
            b2 = sma ** 2
            c2 = a2
            p2 = 1 / (sinphi **2 * costheta ** 2 / a2 + sinphi **2 * sintheta ** 2 / b2 + cosphi ** 2 / c2)
            mask = np.abs(r - p ** 0.5) <= thickness
            pass
        if disk:
            # sma, smi, 

        if outflow:
            # 






    def __make_intensity_model(self, radial_intensity_law, intensity, r, rin=0, rout=np.inf, radial_intensity_gaussian_sigma=None, radial_intensity_pwrlw_pwr=None,rc_taper_cm=None):
        valid_radii =(r >= rin) * (r <= rout)
        if radial_intensity_law.startswith('gaussian'):
            """Gaussian Intensity
            
            Fr = f0 * e ^ -(r - r_inner) / 2s ** 2 | cutoff at r_outer
            """
            intensity[valid_radii] = np.exp(-r[valid_radii] ** 2 /(2 * self.radial_intensity_gaussian_sigma ** 2))
        elif radial_intensity_law.startswith('powerlaw'):
            """Powerlaw Intensity

            Fr = F0 * (r - r_inner) ^ i | cutoff at r_outer
            """
            intensity[valid_radii] = (r[valid_radii] / rc_taper_cm)  ** (-radial_intensity_pwrlw_pwr)
        elif radial_intensity_law.startswith('ring'):
            """Constant Flux Density Ring

            Fr = f0 between disk_radii_inner and disk_radii_outer
            """
            intensity[valid_radii] = 1
        if radial_intensity_law.endswith('taper'):
            intensity[valid_radii] *= np.e ** (-(r[valid_radii] / rc_taper_cm)  ** (2-radial_intensity_pwrlw_pwr))


    def __make_cube(self):
        # in pixels
        ra_offset = self.wcs.array(return_type='wcs', axis=self.wcs.axis1['dtype']) - self.centerwcs[0]
        dec_offset = self.wcs.array(return_type='wcs', axis=self.wcs.axis2['dtype']) - self.centerwcs[1]
        ra_grid, dec_grid = np.meshgrid(ra_offset, dec_offset)
        x_grid = (-ra_grid * self._cosp + dec_grid * self._sinp) # disk major axis
        y_grid = (ra_grid * self._sinp + dec_grid * self._cosp) # disk minor axis
        x_grid2 = x_grid ** 2
        y_grid2 = y_grid ** 2

        mask = y_grid != 0

        # now pull from local def
        theta = self._theta
        vproj = self._vproj
        vtheta = self._vtheta
        vr = self._vr
        disk_intensity = self._intensity


        theta[mask] = 2 * np.arctan((y_grid[mask] / self._cosi) \
                                 / (x_grid[mask] \
                                    + np.sqrt(x_grid2[mask] + (y_grid2[mask] / self._cosi2)))) - constants.pi / 2
        r = np.sqrt(x_grid2 + (y_grid2 / self._cosi2)) #  in pixels
        r *= self.deg2cm # in cm

        ########## DISK
        rin = self.r_inner_cm  # inner cutoff
        rout = self.r_truncate_cm # outer cutoff
        rc = self.velocity_profile_rc_cm # break radius for velocity
        valid_radii =(r >= rin) * (r <= rout)
        self.__make_intensity_model(intensity=disk_intensity, r=r, rin=rin, rout=rout, radial_intensity_law=self.radial_intensity_law, radial_intensity_gaussian_sigma=self.radial_intensity_gaussian_sigma, radial_intensity_pwrlw_pwr=self.radial_intensity_pwrlw_pwr, rc_taper_cm=self.radial_intensity_pwrlw_rc_cm)

        # normalize disk
        disk_intensity *= self.peakint / disk_intensity.max()


        gmmstar = self.__gmsun * self.mstar
        gmmstar_rc = gmmstar * rc

        if 'infall' in self.velocity_profile:
            mask = (r < rout) * (r >= rin)
            gmm_r = gmmstar / r[mask]
            gmmrc_r2 = gmmstar_rc / (r[mask] ** 2)
            vr[mask] += np.sqrt(2 * gmm_r - gmmrc_r2) # in cm/s
            vtheta[mask] += np.sqrt(gmmrc_r2)
        if 'keplerian' in self.velocity_profile:
            mask = (r < rc) * (r >= rin)
            gmm_r = gmmstar / r[mask]
            gmmrc_r2 = gmmstar_rc / (r[mask] ** 2)
            vtheta[mask] += np.sqrt(gmm_r) # assume Keplerian rotation within rc


        mask = r != 0
        # sin cos flipped?
        #                                RADIAL                         TANGENTIAL
        vproj[mask] = (np.sin(theta[mask]) * vtheta[mask] + np.cos(theta[mask]) * vr[mask]) * self._sini * 1e-5 # in cm / s to km/s
        
        # now create cube VELOCITY, RA, DEC
        disk_intensity[~np.isfinite(disk_intensity)] = 0
        self.cube = (disk_intensity)[np.newaxis,...] * np.exp(-(self._velocities - vproj - self.vsys)**2 / (2 * self.sigma_linewidth ** 2))
        # now disk portion
        # make projections

        if self.noise is not None and self.noise > 0:
            disk_intensity += self._noise_image[0, ...]
            self.cube += self._noise_image
        self.intensityimage = disk_intensity
        pass

    def __add__(self, newobj):
        if not isinstance(newobj, BaseFlaredDiskModel):
            raise ValueError(f'{newobj} is not a valid FlaredDiskModel.')
            return
        new = deepcopy(self)
        new.kwargs = newobj.kwargs + new.kwargs
        new.intensityimage += newobj.intensityimage
        new.cube += newobj.cube
