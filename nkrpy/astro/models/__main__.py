"""."""
# flake8: noqa
# cython modules

# internal modules
import time
import cProfile
global starttime
import pprofile
import matplotlib.pyplot as plt

# external modules
import numpy as np

# relative modules
from ...io import fits
from ...misc import constants
from .._wcs import WCS
from ._flareddiskmodel import FlaredDiskModel

# global attributes
starttime = time.time()


def FlaredDiskModelTest():
    def profiletime():
        global starttime
        print(time.time() - starttime)
        starttime = time.time()
    p = cProfile.Profile()
    wcs = WCS({
        'CRVAL1': 123.5971809583,
        'CRPIX1': 251.0,
        'CDELT1': -1.38889e-05,
        'CUNIT1': 'DEG',
        'NAXIS1': 500,
        'CTYPE1': 'RA---SIN',
        'CRVAL2': -34.5176157708,
        'CRPIX2': 251.0,
        'CDELT2': 1.38889e-05,
        'CUNIT2': 'DEG',
        'NAXIS2': 500,
        'CTYPE2': 'DEC--SIN',
        'CRVAL3': 219560354100.0,
        'CRPIX3': 1.0,
        'CDELT3': -122306.5429077,
        'CUNIT3': 'HZ',
        'NAXIS3': 59,
        'CTYPE3': 'FREQ',
        'CRVAL4': 1.0,
        'CRPIX4': 1.0,
        'CDELT4': 1.0,
        'CUNIT4': '',
        'NAXIS4': 1,
        'CTYPE4': 'STOKES',
        'NAXIS': 4,
        'BMAJ': 0.0002123370601071,
        'BMIN': 0.0001954009301133,
        'BPA': 85.61248016357,
        'restfreq':219560354100.0,
        'specsys': 'lsrk',
        'velref': 257,
        'velo-lsr': 4.5,
        'bunit': 'Jy/beam',
        'btype': 'intensity',
        'radesys': 'icrs',
    })
    fdm = FlaredDiskModel(
        wcs=wcs,
        noise = 5.5e-3, # must be in flux density / beam
        fwhm_linewidth=0.1, # in km/s
        # system params
        mstar=1.1, peakint=5e-1, vsys=4.85, # in msun, intensity/beam, km/s, 
        centerwcs=[123.5971809583, -34.51761577083], # center in deg coordinates
        distance_pc=400,
        # disk params all in deg
        inclination=60, # defined 0 is faceon in degrees
        position_angle=-90, # defined east of north Blue in degrees
        r_inner=0, # Truncates the inner of the disk
        r_truncate=4/3600, # hard truncates the disk
        radial_intensity_law='gaussian',
        radial_intensity_gaussian_fwhm=3 / 3600, # eqn e^r^2/2sig^2
        radial_intensity_pwrlw_pwr=0.25,
        velocity_profile=['keplerian'], 
        velocity_profile_rc=4/3600, # eqn vr = sqrt(2G*mstar/r - (G*mstar*rc/r**2)) and vtheta = sqrt(2G*mstar*rc/r))
    )
    prof = pprofile.Profile()
    #profiletime()
    #p.enable()
    fdm()
    #p.disable()
    #profiletime()
    print('Finished Setup')
    #p.dump_stats('/tmp/fdm.prof')
    fdm.writeFits(filename='/tmp/fdm.fits')
    fits.write(f='/tmp/fdm-intensity.fits', data=fdm.intensityimage, header=fdm.wcs.create_fitsheader_from_axes())
    #p.enable()
    with prof():
        fdm.convolve_beam()
    #prof.print_stats()
    #p.disable()
    #profiletime()
    print('Finished convolved.')
    #p.dump_stats('/tmp/fdm_convolve.prof')
    fdm.writeFits(filename='/tmp/fdm_convolve.fits')
    import matplotlib.pyplot as plt
    rf = wcs.get_head('restfreq')
    channelvsys = int(round(wcs(rf - fdm.vsys * 1e3*1e2 / constants.c * rf, return_type='pix', axis=wcs.axis3['dtype']), 0))
    fig, ax = plt.subplots()
    moment0 = np.nansum(fdm.cube, axis=0)
    ax.imshow(moment0, origin='lower', cmap='magma')
    fig.savefig('/tmp/fdm-m0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.cube[:channelvsys, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fdm-bm0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.cube[channelvsys:, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fdm-rm0.pdf', dpi=150)
    ax.cla()
    moment0 = np.nansum(fdm.convolved_cube, axis=0)
    ax.imshow(moment0, origin='lower', cmap='magma')
    fig.savefig('/tmp/fdm_convolve-m0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.convolved_cube[:channelvsys, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fdm_convolve-bm0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.convolved_cube[channelvsys:, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fdm_convolve-rm0.pdf', dpi=150)
    ax.cla()


def FlaredEnvelopeModelTest():
    def profiletime():
        global starttime
        print(time.time() - starttime)
        starttime = time.time()
    p = cProfile.Profile()
    wcs = WCS({
        'CRVAL1': 123.5971809583,
        'CRPIX1': 251.0,
        'CDELT1': -1.38889e-05,
        'CUNIT1': 'DEG',
        'NAXIS1': 500,
        'CTYPE1': 'RA---SIN',
        'CRVAL2': -34.5176157708,
        'CRPIX2': 251.0,
        'CDELT2': 1.38889e-05,
        'CUNIT2': 'DEG',
        'NAXIS2': 500,
        'CTYPE2': 'DEC--SIN',
        'CRVAL3': 219560354100.0,
        'CRPIX3': 1.0,
        'CDELT3': -122306.5429077,
        'CUNIT3': 'HZ',
        'NAXIS3': 59,
        'CTYPE3': 'FREQ',
        'CRVAL4': 1.0,
        'CRPIX4': 1.0,
        'CDELT4': 1.0,
        'CUNIT4': '',
        'NAXIS4': 1,
        'CTYPE4': 'STOKES',
        'NAXIS': 4,
        'BMAJ': 0.0002123370601071,
        'BMIN': 0.0001954009301133,
        'BPA': 85.61248016357,
        'restfreq':219560354100.0,
        'specsys': 'lsrk',
        'velref': 257,
        'velo-lsr': 4.5,
        'bunit': 'Jy/beam',
        'btype': 'intensity',
        'radesys': 'icrs',
    })
    fdm = FlaredDiskModel(
        wcs=wcs,
        noise = 5.5e-3, # must be in flux density / beam
        fwhm_linewidth=0.1, # in km/s
        # system params
        mstar=1.1, peakint=5e-1, vsys=4.85, # in msun, intensity/beam, km/s, 
        centerwcs=[123.5971809583, -34.51761577083], # center in deg coordinates
        distance_pc=400,
        # disk params all in deg
        inclination=60, # defined 0 is faceon in degrees
        position_angle=42, # defined east of north Blue in degrees
        r_inner=0/3600, # Truncates the inner of the disk
        r_truncate=60/3600, # hard truncates the disk
        radial_intensity_law='gaussian',
        radial_intensity_gaussian_fwhm=20 / 3600, # eqn e^r^2/2sig^2
        radial_intensity_pwrlw_pwr=0.25,
        velocity_profile=['infall', 'keplerian'], 
        velocity_profile_rc=4/3600, # eqn vr 
    )
    prof = pprofile.Profile()
    #profiletime()
    #p.enable()
    fdm()
    #p.disable()
    #profiletime()
    print('Finished Setup')
    #p.dump_stats('/tmp/fem.prof')
    fdm.writeFits(filename='/tmp/fem.fits')
    fits.write(f='/tmp/fem-intensity.fits', data=fdm.intensityimage, header=fdm.wcs.create_fitsheader_from_axes())
    #p.enable()
    with prof():
        fdm.convolve_beam()
    #prof.print_stats()
    #p.disable()
    #profiletime()
    print('Finished convolved.')
    #p.dump_stats('/tmp/fem_convolve.prof')
    fdm.writeFits(filename='/tmp/fem_convolve.fits')
    import matplotlib.pyplot as plt
    rf = wcs.get_head('restfreq')
    channelvsys = int(round(wcs(rf - fdm.vsys * 1e3*1e2 / constants.c * rf, return_type='pix', axis=wcs.axis3['dtype']), 0))
    fig, ax = plt.subplots()
    moment0 = np.nansum(fdm.cube, axis=0)
    ax.imshow(moment0, origin='lower', cmap='magma')
    fig.savefig('/tmp/fem-m0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.cube[:channelvsys, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fem-bm0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.cube[channelvsys:, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fem-rm0.pdf', dpi=150)
    ax.cla()
    moment0 = np.nansum(fdm.convolved_cube, axis=0)
    ax.imshow(moment0, origin='lower', cmap='magma')
    fig.savefig('/tmp/fem_convolve-m0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.convolved_cube[:channelvsys, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fem_convolve-bm0.pdf', dpi=150)
    ax.cla()
    ax.imshow(np.nansum(fdm.convolved_cube[channelvsys:, ...], axis=0),origin='lower', cmap='magma')
    fig.savefig('/tmp/fem_convolve-rm0.pdf', dpi=150)
    ax.cla()

if __name__ == '__main__':
    #FlaredDiskModelTest()
    FlaredEnvelopeModelTest()


# rm -f /tmp/pprofile.prof && ~/python3 -m nkrpy.astro.models && ~/python3 -c "import numpy as np;from nkrpy.io import fits; h,d = fits.read('/tmp/fdm_convolve.fits'); h1,d1 = fits.read('/home/reynolds/local/ALMA/ALMAc6-BHR7/pdspy/data/BHR7-13mm_SBLB_C18O_natural.pbcor.fits'); fits.write(f='/tmp/data-model.fits', header=h, data=np.abs((d-d1)))" && casaviewer /tmp/data-model.fits
