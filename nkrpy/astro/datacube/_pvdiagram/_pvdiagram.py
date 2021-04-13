"""."""
# flake8: noqa
# cython modules

# internal modules
import os
import sys

# external modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.ndimage import interpolation as inter
from scipy import optimize
from scipy.optimize import curve_fit
import emcee
import corner
from IPython import embed as embed
from scipy.signal import savgol_filter

# relative modules
from ...._unit import Unit
from ....misc import constants
from ....io import _fits as fits
from ... import WCS
from ._lasso import SelectFromCollection, plotter
from ..momentmaps import momentmap
from ...image.functions import (binning, center_image, rotate_image, sum_image, select_rectangle, remove_padding)
# global attributes
__all__ = ('keplerian_vel_2_rad',
           'pvdiagram', 'default_config')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

import time


class Timer():
    def __init__(self, debug = False):
        self.start()
        self.debug = debug
    def start(self):
        self.__start = time.time()
        self.__last = self.__start
    def restart(self):
        self.start()
    def log(self):
        self.__current = time.time() - self.__last
        if self.debug:
            print(f'Time dif - {self.__current: 0.4f}s')
        self.__last = time.time()


def timer(debug):
    if debug:
        print(Timer.log())


def keplerian_vel_2_rad(mass, velocity, dist, inc):
    """Keplerian rotation.

    velocity in km/s
    mass in solar masses
    dist in pc
    inc in radians
    """
    mask = velocity == 0
    velocity[mask] = 1e-10
    radii = constants.g * constants.msun * mass / (np.abs(velocity) * 100000. * np.sin(inc)) ** 2
    radii *= (180. / constants.pi * 3600.) / (Unit(baseunit='pc', convunit='cm', vals=dist))
    radii[velocity < 0] *= -1.
    velocity[mask] = 0
    return radii


def keplerian_rad_2_vel(mass, rad, dist, inc):
    """Keplerian rotation.

    radii in arcsec
    mass in solar masses
    dist in pc
    inc in radians
    """
    radii = (rad / 3600. / 180. * constants.pi) * (Unit(baseunit='pc', convunit='cm', vals=dist))
    mask = radii <= 0
    radii[mask] = 1e-10
    velsqrd = constants.g * constants.msun * mass / (radii * np.sin(inc) ** 2)
    velsqrd = velsqrd.astype(np.float64)
    vel = np.sqrt(velsqrd) / 100. / 1000.
    return vel


def default_config(config):
    dconfig = {
        'header': 'Dictionary object of the header',
        'wcs': 'The WCS module as provided by nkrpy.astro.WCS',
        'ra': 'The x position of the object in the same units as the image',
        'dec': 'The y position of the object in the same units as the image.',
        'inclination': 'Inclination of the object in radians',
        'positionangle': 'position angle of the object in degress',
        'pixelwidth': 'the number of pixels to sum',
        'mass': 'mass of the central gravitational source in msun',
        'mass_err': 'error of the mas measurement in msun',
        'v_source': 'source velocity in km/s',
        'distance_source': 'distance to the source in pc',
        'aswidth': 'The width in arcsec to plot',
        'vwidth': 'The width in km/s to plot',
        'contour_params': 'Dict with the keys "color": str, "start": int, "interval": float, "max": int to set contours',
        'debug_mode': 'True, False to verbosely output',
        'fit': 'True, False to run fitting routine.',
        'path': 'path to save and load files',
        'contour': 'Plot data as contours instead of image. If True, must have contour_params specified',
        'grayscale': 'Plot data as grayscale or viridis. If true plots as grayscale',
        'momentmap': 'Plot a moment 1 map overlaid as contours',
    }
    run = True
    for f in dconfig:
        if f not in config:
            run = False
            v = dconfig[f]
            print(f'{f} not found in your input config\nMust be {f}: {v}')
    if not run:
        sys.exit()
    return


def pvdiagram(fig, ax, config, image, pfig, pax):
    """
    ra, dec, arcsec_width in decimal deg
    pa, inc in deg
    width in pixels
    mass in msun
    vsys, vwidth in km/s
    dsource in pc
    image:
        Assuming image is a 3d array with first axis the freq/vel axis."""

    """
    Reading in configuration and manipulating base values
    -----------------------------------------------------
    """
    default_config(config)
    wcs = config['wcs']
    ra = config['ra']
    dec = config['dec']
    inc = Unit(vals=config['inclination'], baseunit='deg', convunit='radians').get_vals()
    pa = config['positionangle']
    width = config['pixelwidth']
    assert width % 2 == 1
    mass_est = config['mass']
    mass_est_err = config['mass_err']
    v_sys = config['v_source']
    d_source = config['distance_source']
    arcsec_width = config['aswidth']
    v_width = config['vwidth']
    contour_params = config['contour_params']
    test = config['debug_mode']
    timer = Timer(test)
    if test:
        for k, v in config.items():
            print(f'{k}: {v}')

    def fit(v, mass):
        # v in km.s
        # m in sol m
        # returns fit in arcsec
        return fit2(v, mass, 0)

    def fit2(v, mass, vsys):
        # v in km.s
        # m in sol m
        # returns fit in arcsec
        if not isinstance(v, np.ndarray):
            v = np.array([v], dtype=np.float64)
        mask = v - vsys == 0
        v[mask] = 1e-10
        v = np.abs(v - vsys)
        v = Unit(vals=v, baseunit='km', convunit='cm')
        a = constants.g * constants.msun * mass * np.sin(inc) ** 2 / v ** 2
        # a is in cm
        a *= 1. / Unit(vals=d_source, baseunit='pc', convunit='cm').get_vals()
        # now in radians
        a *= Unit(vals=1., baseunit='radians', convunit='arcsec').get_vals()
        return a # now in arcsec

    """
    Try to read the data
    --------------------
    """
    try:
        axis = wcs.get_axis_number('freq')
        freq_array = wcs.array(return_type='wcs', axis='freq')
        rf = wcs.get_head('restfrq')
        vel_array = (rf - freq_array) / rf
        vel_array *= constants.c / 100. / 1000.
        # now in km / s
    except Exception as e:
        if test:
            print('Freq axis not found, asumming vel axis:', e)
        vel_array = wcs.array(return_type='wcs', axis='vel')
        uni = wcs.get('vel').kwargs['uni']
        unit = uni.split('m')[0]
        unit += 'm'
        vel_array = Unit(baseunit=uni, convunit='km', vals=vel_array).get_vals()
        rf = wcs.get_head('restfrq')
        freq_array = rf - vel_array * 1000. * 100. / constants.c * rf

    if len(image.shape) == 4:
        image = image[0]
    assert len(image.shape) == 3

    """
    Generate the new WCS conversions and select necessary data
    ----------------------------------------------------------
    """
    original  = image.copy() # vel, dec, ra
    timer.log()
    minv = v_width / -2. + v_sys
    maxv = v_width / 2. + v_sys
    m1 = (vel_array < maxv)
    m2 = (vel_array > minv)
    vel_array_mask = m1 * m2
    vdelt = (v_width)/ 11.
    minv = np.min(vel_array[vel_array_mask])
    maxv = np.max(vel_array[vel_array_mask])
    new_vcoord = np.linspace(minv, maxv, 11)
    ras = wcs(ra, 'pix', 'ra---sin')
    decs = wcs(dec, 'pix', 'dec--sin')
    # we now have the selection for the 3rd axis, work on other two
    masked = select_rectangle(image.T, xcen = ras, ycen = decs, ylen=arcsec_width / 2. / 3600. / wcs.get('ra---sin')['del'] + 2, xlen=(width - 1) / 2. + 2, pa=pa) # returns ra. dec,vel
    cut_image = remove_padding(masked).T # returns vel, dec, ra
    timer.log()
    if test:
        print('New Velocities: ', new_vcoord)
        print('Imageshape (pre mask): ', cut_image.shape)
        if not os.path.isdir('testpv'):
            os.mkdir('testpv')
        testfig, testax = plt.subplots(figsize=(15,15))
        vals = np.ravel(np.where(vel_array_mask == True))
        for i in range(np.min(vals) + int(vals.shape[0] / 4), np.min(vals) + int(vals.shape[0] / 1.5), 3):
            testax.imshow(cut_image[i, :, :], origin='lower', cmap='cividis', interpolation='nearest')
            testax.set_aspect(1./testax.get_data_ratio())
            testfig.savefig(f'testpv/test_orig-{i}.pdf', bbox_inches='tight')
            testax.cla()
    lvel = wcs(0, 'wcs', 'freq')
    uvel = wcs(cut_image.shape[0], 'wcs', 'freq')
    timer.log()
    embed() if config['interactive'] else None
    cut_image = cut_image[vel_array_mask, ...]
    timer.log()
    if (uvel - lvel) < 0: # force red-> blueshift for axis
        if test:
            print('flipping')
        cut_image = np.flip(cut_image, 0)
    rotated_image = rotate_image(cut_image.T, axis=0, angle=90-pa, use_skimage=True, resize=True, mode='constant', cval=np.nan) # now in pos, pos, vel
    unpad_rotated_image = remove_padding(rotated_image)
    cut_rotated_image = unpad_rotated_image[2:-2, 2:-2, ...]
    contour_std = cut_rotated_image.std(axis=1).std(axis=0)

    # First quartile (Q1) 
    contour_std = np.percentile(contour_std, 25, interpolation = 'midpoint')
    summed_image = np.nansum(cut_rotated_image, axis=1) # in pos, vel
    if test:
        print('SummedImageshape: ', summed_image.shape)
        testax.imshow(summed_image, origin='lower', cmap='cividis', interpolation='nearest')
        testax.set_aspect(1./testax.get_data_ratio())
        testfig.savefig('testpv/test_sum.pdf', bbox_inches='tight')
        testax.cla()
        print('acswidth|imageshape: ', arcsec_width, summed_image.shape)
    pvwcs = WCS({
        'naxis1': summed_image.shape[1],
        'naxis2': summed_image.shape[0],
        'ctype1': 'vel',
        'crpix1': 0,
        'crval1': minv,
        'cdelt1': np.abs(wcs.get('freq')['del'] / wcs.get_head('restfreq') * constants.c / 100. / 1000.),
        'cunit1': 'km/s',
        'ctype2': 'dist',
        'crpix2': 0,
        'crval2': arcsec_width / -2.,
        'cdelt2': arcsec_width / summed_image.shape[0],
        'cunit2': 'arcsec',
    })
    config['PV-WCS'] = pvwcs
    

    # plotting

    std = contour_std if contour_params['std'] is None else contour_params['std']
    if test:
        print(f'STD: {std:0.2e}, {std_region}')
    cax = ax.imshow(summed_image, origin='lower', cmap='cividis' if not config['grayscale'] else 'gray', interpolation='nearest', zorder=1)
    # plot moment map if requested
    _, mstd = momentmap(image.T, freq_axis=freq_array, moment=0)
    mstd = np.std(mstd)
    # plot moment map
    if config['momentmap']:
        moment = momentmap(image.T, freq_axis=freq_array[vel_array_mask], moment=1)
        contour_levels = np.sort(np.array([x * std for x in range(contour_params['max'])
                                           if x >= contour_params['start'] and
                                           ((x - contour_params['start']) %
                                           contour_params['interval']) == 0]))
        try:
            contour = ax.contour(moment, contour_levels, colors=contour_params['color'], zorder=2 if not config['contour'] else 1)
        except Exception:
            pass
    # plot contours
    if contour_params:
        print('plotting contour')
        contour_levels = np.sort(np.array([x * std for x in range(contour_params['max'])
                                           if x >= contour_params['start'] and
                                           ((x - contour_params['start']) %
                                           contour_params['interval']) == 0]))
        try:
            contour = ax.contour(summed_image, contour_levels, colors=contour_params['color'], zorder=10 if not config['contour'] else 1)
        except Exception:
            pass

    # formatting
    new_dcoord = np.linspace(-arcsec_width / 2., arcsec_width / 2., 11)
    y_label = [f"{i:.1f}" for i in new_dcoord]
    x_label = [f"{i:.1f}" for i in new_vcoord]
    xticks = np.linspace(0, summed_image.shape[1], 11)
    yticks = np.linspace(0, summed_image.shape[0], 11)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)
    cbar = None
    if cax is not None:
        cbar = plt.colorbar(cax)
    # gen array of offsets
    _t = pvwcs.array(axis='vel', return_type='wcs')
    hr_keplerian_pix = pvwcs.array(size=1000, axis='dist', return_type='pix')
    hr_keplerian_wcs = pvwcs.array(size=1000, axis='dist', return_type='wcs')
    mask = hr_keplerian_wcs != 0
    hr_keplerian_wcs = hr_keplerian_wcs[mask]
    hr_keplerian_pix = hr_keplerian_pix[mask]
    # split between left and right
    mask = hr_keplerian_wcs > 0
    hr_keplerian_wcs_left = hr_keplerian_wcs[~mask]
    hr_keplerian_wcs_right = hr_keplerian_wcs[mask]
    hr_keplerian_pix_right = hr_keplerian_pix[mask]
    hr_keplerian_pix_left = hr_keplerian_pix[~mask]
    # now create velocity
    hr_velocity_wcs_right = keplerian_rad_2_vel(mass_est, hr_keplerian_wcs_right, config['distance_source'], inc) + v_sys
    hr_velocity_wcs_left = -1 * keplerian_rad_2_vel(mass_est, -1. * hr_keplerian_wcs_left, config['distance_source'], inc) + v_sys
    hr_velocity_pix_left = pvwcs(hr_velocity_wcs_left, 'pix', 'vel')
    hr_velocity_pix_right = pvwcs(hr_velocity_wcs_right, 'pix', 'vel')
    mask = hr_velocity_wcs_right <= np.max(_t)
    hr_velocity_wcs_right = hr_velocity_wcs_right[mask]
    hr_keplerian_wcs_right = hr_keplerian_wcs_right[mask]
    hr_velocity_pix_right = hr_velocity_pix_right[mask]
    hr_keplerian_pix_right = hr_keplerian_pix_right[mask]
    mask = hr_velocity_wcs_left >= np.min(_t)
    hr_velocity_wcs_left = hr_velocity_wcs_left[mask]
    hr_keplerian_wcs_left = hr_keplerian_wcs_left[mask]
    hr_velocity_pix_left = hr_velocity_pix_left[mask]
    hr_keplerian_pix_left = hr_keplerian_pix_left[mask]
    #embed()
    if config['plot_estimate']:
        ax.plot(hr_velocity_pix_left, hr_keplerian_pix_left,  color='red', label='Keplerian rotation', linestyle=':', linewidth=3, zorder=15)
        ax.plot(hr_velocity_pix_right, hr_keplerian_pix_right,  color='red', label='Keplerian rotation', linestyle=':', linewidth=3, zorder=15)
        ax.axvline(pvwcs(v_sys, 'pix', 'vel'), color='white', linestyle='--',
                   label='v$_{source}$ =' + f' {v_sys} km s$^{-1}$')
        ax.axhline(pvwcs(0, 'pix', 'dist'), color='white', linestyle='--')
    if config['fit']:
        pix_arcsec_central, pix_vel_central = (np.array(summed_image.shape) / 2.).tolist()
        pixel_size_y_arcsec = arcsec_width / (pix_arcsec_central * 2)
        pixel_size_x_vel = v_width / (pix_vel_central * 2)
        #######################################################

        def inv(x, m):
            return pvwcs(fit(pvwcs(x, 'pix', 'vel'),m), 'pix', 'dist')

        def lnlike(params, x, y, yerr):
            m, v, a, lnf = params
            nx = x - v # km/s moved to 0
            ny = y - a # arcsec moved to 0
            ny = np.abs(ny)
            nx = np.abs(nx)
            model = fit(nx, m) # returns in arcsec
            inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
            f = (ny-model)**2*inv_sigma2
            invprob= -0.5*(np.sum(f - np.log(inv_sigma2))) # gaussian
            if invprob <= 0:
                return invprob
            print('Probs are inaccurate: ', invprob, params, x, y, yerr)

        def lnprior(params):
            m, v, a, lnf = params
            if 1e-3 < m < 1e2 and\
               -10.0 < lnf < 10.0 and\
               -10.0 < a < 10.0 and\
               -1e2 < v < 1e2:
                return 0.0
            return -np.inf

        def lnprob(params, x, y, yerr):
            lp = lnprior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(params, x, y, yerr)
        #####################################################
        fit_ar_orig = np.concatenate([v.vertices.reshape(-1, 2) for c in contour.collections for v in c.get_paths()])

        for i, row in enumerate(fit_ar_orig):
            x, y = row
            arc = pvwcs(x, 'wcs', 'dist') # pix_2_arc(i)
            vel = pvwcs(y, 'wcs', 'vel')
            # print(f'Pos: {arc} @ {fi}, Vel: {vel} @ {i}')

        fit_ar = fit_ar_orig.copy() # x,y in pixel space

        if config['redraw'] or not os.path.isfile(f'{config["savepath"]}/leftpoints{config["targetname"]}.txt') or not os.path.isfile(f'{config["savepath"]}/rightpoints{config["targetname"]}.txt'):
            rawfig = plotter('PV diagram: Lasso black points to fit')
            rawfig.int()
            rawfig.open((1,1), xlabels=x_label, ylabels=y_label, xticks=xticks, yticks=yticks, xlabel=r'Velocity (km s$^{-1}$)', ylabel='Position (arcsec)')
            rawfig.imshow(summed_image, origin='lower', cmap='cividis' if not config['grayscale'] else 'gray', interpolation='nearest', zorder=1)
            rawfig.scatter(fit_ar[:, 0], fit_ar[:, 1], 'scatter raw', color='black', s=2, zorder=15)
            rawfig.plot(hr_velocity_pix_right, hr_keplerian_pix_right,  'left_hr_rawfig', color='red', label='Keplerian rotation', linestyle=':', linewidth=3, zorder=10)
            rawfig.plot(hr_velocity_pix_left, hr_keplerian_pix_left, 'right_hr_rawfig', color='red', linestyle=':', linewidth=3, zorder=10)
            # actual defining mask
            left_fit_pix = rawfig.selection('scatter raw', 'left side').reshape(-1,2) # pix
            right_fit_pix = rawfig.selection('scatter raw', prompt='right side').reshape(-1,2) # pix
            rawfig.close()
            left_fit_wcs = np.array([pvwcs(left_fit_pix[:, 0], 'wcs', 'vel'), pvwcs(left_fit_pix[:, 1], 'wcs', 'dist')]).T
            right_fit_wcs = np.array([pvwcs(right_fit_pix[:, 0], 'wcs', 'vel'), pvwcs(right_fit_pix[:, 1], 'wcs', 'dist')]).T

            with open(f'{config["savepath"]}/leftpoints{config["targetname"]}.txt', 'w') as f:
                np.savetxt(f, left_fit_wcs, delimiter=';')
            with open(f'{config["savepath"]}/rightpoints{config["targetname"]}.txt', 'w') as f:
                np.savetxt(f, right_fit_wcs, delimiter=';')
        else:
            left_fit_wcs = np.loadtxt(f'{config["savepath"]}/leftpoints{config["targetname"]}.txt', delimiter=';', dtype=float)
            right_fit_wcs = np.loadtxt(f'{config["savepath"]}/rightpoints{config["targetname"]}.txt', delimiter=';', dtype=float)

        # normalize to upper quadrants
        left_fit_wcs[:, 1] = np.abs(left_fit_wcs[:, 1])
        right_fit_wcs[:, 1] = np.abs(right_fit_wcs[:, 1])

        # smooth data somehow
        fit_ar = []
        kep_ar = []
        for a in [left_fit_wcs, right_fit_wcs]:
            window = int(10**(np.log10(a.shape[0]) - 1))
            window = window if config['window'] == 0 else config['window']
            window = window if window % 2 == 1 else window + 1
            vel, arc = a[:, 0], a[:, 1]
            if a.shape[0] == 0 or not config['sg'] and not config['bin']:
                pass
            elif config['sg']:
                arc = savgol_filter(arc, window, 2, axis=-1)
            elif config['bin']:
                window = window if config['window'] else np.median(np.diff(np.sort(vel)))
                vel, arc = binning(vel, arc, windowsize=window)
            # weight with higher velocity = greater weight (disk fitting)
            # normed from 0 to 1
            a = np.abs(arc)
            kep_ar.append(a)
            fit_ar.append(np.array([vel, arc]))
        embed() if config['interactive'] else None

        kep_weights = np.concatenate(kep_ar)
        kep_weights *= 1. / np.max(kep_weights)
        fit_ar = np.concatenate(fit_ar, axis=-1).T
        yerr = fit_ar[:,1] * kep_weights * 0.5

        # data has now been selected and time for fitting
        t_data = np.hstack([fit_ar, yerr[:, None]]) # should be dist, vel
        f_true = .5

        if test:
            pax.cla()
            pax.set_title('Fitted Points')
            pax.scatter(left_fit_wcs[:,0], left_fit_wcs[:,1], color='orange',marker='.', zorder=20)
            pax.scatter(right_fit_wcs[:,0], right_fit_wcs[:,1], color='red',marker='.', zorder=20)
            pax.errorbar(t_data[:,0], t_data[:,1], yerr=t_data[:,2], color='blue', marker='.', linestyle="", zorder=1,errorevery=10)
            pax.set_ylim(np.min(fit_ar[:,0]), 1.1*np.max(fit_ar[:,1]))
            pax.set_aspect(1./pax.get_data_ratio())
            pfig.savefig(f"{config['savepath']}/pvfit-test.pdf",dpi=300)
        
        if config['mcmc']:
            nll = lambda *args: -lnlike(*args)
            initial = np.array([mass_est, 0, np.log(f_true)])
            initial += initial * np.random.randn(initial.shape[0])
            ndim, nwalkers = len(initial), 50
            embed() if config['interactive'] else None
            result = optimize.minimize(nll, initial, args=(x, y, yerr))
            pos = result["x"] + result["x"] * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
            pos, prob, state = sampler.run_mcmc(pos, 800)
            pax.cla()
            for j in range(ndim):
                for k in range(nwalkers):
                    pax.plot(sampler.chain[k,:,j])

                pfig.savefig(f"{config['savepath']}/burned_steps_{j}.pdf")

                pax.cla()
            lfig.clf()
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, 1000, rstate0=state)
            osamples = sampler.chain
            samples = osamples.copy().reshape((-1, ndim))
            af = sampler.acceptance_fraction
            print("Mean acceptance fraction:", np.mean(af))
            truths = np.median(samples, axis=0)
            print('Plotting Emcee Fits')

            m_true = np.median(samples, axis=0)[0]
            config['EMCEE-SAMPLES'] = [samples, prob]

            pax.cla()
            pax.scatter(x, y, color='red',lw=10,marker='.')
            xdl = np.linspace(-10,-0.25) 
            xdr = np.linspace(0.25, 10)
            for m, v, lnf in samples[np.random.randint(samples.shape[0], size=100)]:
                pax.plot(xdl, fit(xdl, m), color="k", alpha=0.3)
                pax.plot(xdr, fit(xdr, m), color="k", alpha=0.3)  
            pax.plot(xdl, fit(xdl, m_true),color='cyan')
            pax.plot(xdr, fit(xdr, m_true),color='cyan')
            pax.set_xlim(-3, 3)
            pax.set_ylim(0, 4)
            pax.set_aspect(1./pax.get_data_ratio())
            pfig.savefig(f"{config['savepath']}/pvfit.pdf",dpi=300)

            cfig = corner.corner(samples, labels=["$m$", "$v$", "$\ln\,f$"],
                                  truths=truths)
            cfig.savefig(f"{config['savepath']}/triangle.pdf",dpi=600)
            cfig.clf()
            """
            samples[:, 1] = np.exp(samples[:, 1])
            m_mcmc, f_mcmc = map(lambda v: (v[1], v[1]-v[0]),
                                         zip(*np.percentile(samples, [16, 50, 84],
                                                            axis=0)))
            print(f'M:{m_true}..{m_mcmc}\nE:{f_true}..{f_mcmc}')
            ax.plot(xl, inv(xl, m_true ), color="C", alpha=1,label=f'Mass: {m_true:.2f} M$_\odot$')
            ax.plot(xr, inv(xr, -1.* m_true ), color="C", alpha=1)
            eax.plot(xl, inv(xl, m_true ), color="C", alpha=1,label=f'Mass: {m_true:.2f} M$_\odot$')
            eax.plot(xr, inv(xr, -1.* m_true), color="C", alpha=1)
            ebar = plt.colorbar(ecs, ax=eax, fraction=0.046, pad=0.04)
            mass = m_true
            efig.tight_layout()
            eax.set_aspect(1./eax.get_data_ratio())
            efig.savefig(f"{config['savepath']}/pvfit-xtr.pdf",dpi=300)
            """
            pax.cla()
            for j in range(ndim):
                for k in range(nwalkers):
                    pax.plot(osamples[k,:,j])

                pfig.savefig(f"steps_{j}.pdf")

        fit_vel_wcs = np.arange(pvwcs.get('vel')['del'] / 10., config['vwidth'] / 2, pvwcs.get('vel')['del'] / 10.)

        fit_vel_wcs_left = -1. * fit_vel_wcs
        fit_vel_wcs_right = fit_vel_wcs
        fit_vel_pix_left = pvwcs(fit_vel_wcs, 'pix', 'vel')
        fit_vel_pix_right = pvwcs(fit_vel_wcs, 'pix', 'vel')

        # try initial fit
        try:
            midest = np.median(t_data[[0, int(t_data.shape[0] / 2), -1], 0])
            try:
                popt, pcov = optimize.curve_fit(fit2, t_data[:,0], t_data[:,1], sigma=yerr, p0=[config['mass'], midest], bounds=[[config['mass'] / 2., midest - 2], [config['mass'] * 2., midest + 2]])
            except:
                popt = [config['mass'], config['v_source']]

            def specialfit(xdata, ydata, yerr):
                pf, pa = plt.subplots()
                pa.scatter(xdata, ydata)
                popt, pcov = optimize.curve_fit(fit2, xdata, ydata, sigma=yerr, p0=[config['mass'], config['v_source']], bounds=[[config['mass'] / 2., config['v_source'] - 2.], [config['mass'] * 2., config['v_source'] + 2.]])
                fkwl = fit2(fit_vel_wcs_left, *popt)
                fkwr = fit2(fit_vel_wcs_right, *popt)
                pa.plot(fit_vel_wcs_left, fkwl)
                pa.plot(fit_vel_wcs_right, fkwr)
                pa.set_xlim(np.min(xdata), np.max(xdata))
                pa.set_ylim(np.min(ydata), np.max(ydata))
                pa.set_title(f'Fit {popt}')
                plt.show()
                fkpl = pvwcs(-1. * fkwl, 'pix', 'dist')
                fkpr = pvwcs(fkwr, 'pix', 'dist')
                return popt, pcov, fkwl, fkwr, fkpl, fkpr

            # fit savgol
            fit_last = np.inf
            count = 0
            while np.abs((fit_last - popt[1]) / popt[1]) > 1e-2 and count < 6:
                xb, yb = binning(t_data[:, 0], t_data[:, 1], windowsize=t_data.shape[0] / 2 / (count + 1))
                _, eb = binning(t_data[:, 0], t_data[:, 2], windowsize=t_data.shape[0] / 2 / (count + 1))
                popt, pcov, fit_kepl_wcs_left, fit_kepl_wcs_right, fit_kepl_pix_left, fit_kepl_pix_right = specialfit(xb, yb, eb)
                """
                left = (t_data[:, 0] - popt[1]) < 0
                dlen = t_data[left, :].shape[0]
                sgwindow = int(dlen / 2 ** (6 - count))
                sgwindow = dlen / 3 if sgwindow > dlen / 2 else sgwindow
                sgwindow  = sgwindow if sgwindow % 2 == 1 else sgwindow + 1
                print(f'{count}: {sgwindow}: {popt[1]}, {t_data[left, 1].shape}')
                sgl = savgol_filter(t_data[left, 1], sgwindow, 2, axis=-1)
                sgr = savgol_filter(t_data[~left, 1], sgwindow, 2, axis=-1)
                sg = np.concatenate([sgl, sgr])
                popt, pcov, fit_kepl_wcs_left, fit_kepl_wcs_right, fit_kepl_pix_left, fit_kepl_pix_right = specialfit(t_data[:, 0], sg, t_data[:, -1])
                fit_last = popt[1]
                print(f'{config["title"]} SAVGOL fit (mass, V_sys): {popt[0], popt[1]}')
                """
                count += 1
            embed() if config['interactive'] else None
            line = pvwcs(popt[1], 'pix', 'vel')
            ax.axvline(0, color='white', linestyle='--',
                       label='v$_{source}$ =' + f' {line} km s$^{-1}$')
            ax.axhline(0, color='white', linestyle='--')

            fit_kepl_wcs_left = -1. *fit2(np.abs(hr_velocity_wcs_left - v_sys), *popt)
            fit_kepl_wcs_right = fit2(hr_velocity_wcs_right - v_sys, *popt)
            fit_kepl_pix_left = pvwcs(fit_kepl_wcs_left, 'pix', 'dist')
            fit_kepl_pix_right = pvwcs(fit_kepl_wcs_right, 'pix', 'dist')
            ax.plot(hr_velocity_pix_left, fit_kepl_pix_left,  color='green', label='Keplerian rotation', linestyle=':', linewidth=3, zorder=20)
            ax.plot(hr_velocity_pix_right, fit_kepl_pix_right, color='green', linestyle=':', linewidth=3, zorder=20)
        except Exception as e:
            print('Failed to fit:', e)
        if test:
            pax.cla()
            fitted = fit2(t_data[:,0], *popt)
            pax.plot(t_data[:,0] + v_sys, fitted,'-',color='black')
            pax.set_aspect(1./pax.get_data_ratio())
            pfig.savefig(f"{config['savepath']}/pvfit-scipy.pdf",dpi=300)


    return summed_image, cbar


if __name__ == "__main__":
    """Directly Called."""

    string = """
    How I suggest calling the file.

        Read in header and data from fits file. Header must be in a dictionary format
        the data must be 3d as freq/vel/wav : y : x

        Provide the appropriate configuration parameters.

        if ra,dec in icrs format
        ra, dec = config['ircs'].split(',')
        ra, dec = icrs2deg(ra) * 15., icrs2deg(dec)

        inc = Unit('deg', 'rad', config['inclination']).get_vals()[0]
        config['inclination'] = inc

        # update config
        config['header'] = dict(header)
        config['wcs'] = wcs
        config['ra'] = ra
        config['dec'] = dec

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_title(config['title'], fontsize=20)
        ax.set_ylabel(r'Velocity (km s$^{-1}$)', fontsize=20)
        ax.set_xlabel(r'Offset ($^{\prime\prime}$)', fontsize=20)

        pv, cbar = pvdiagram(fig, ax, config, data)
        ax.axhline(int(pv.shape[0] / 2), color='white', linestyle='--',
                   label='v$_{source}$ = ' + str(config['v_source']) + ' km s$^{-1}$')
        ax.axvline(int(pv.shape[1] / 2), color='white', linestyle='--')
        pvshape = pv.shape
        cbar.set_label('Jy/Beam', fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        ax.tick_params('both', labelsize=18)
        ax.set_aspect(int(pvshape[1] / pvshape[0]))
        ax.set_ylim(0, pvshape[0] - 1)
        ax.set_xlim(0, pvshape[1])
        plots.set_style()
        savefile = construct_savefile_name(config)
        fig.savefig(savefile + '.pdf', bbox_inches='tight')
    """
    print(string)

# end of code

# end of file
