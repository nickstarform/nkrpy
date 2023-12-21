"""."""
# flake8: noqa
# cython modules

# internal modules
import os
import sys

# external modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from IPython import embed
from scipy.signal import savgol_filter
from scipy.ndimage.interpolation import shift
from skimage.transform import rotate as skimage_rotate

# relative modules
from ..._unit import Unit
from ...misc import constants, Format
from ... import _math
from ...io import fits
from .. import WCS
from ._lasso import SelectFromCollection, plotter
from .._functions import (binning, select_rectangle, remove_padding3d)
# global attributes
__all__ = ['keplerian_vel_2_rad',
           'pvdiagram', 'default_config', 'pvslicer']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

import time

def odd(num, even=False):
    return num if num % 2 != even else num + 1

def isodd(num):
    return True if num % 2 == 1 else False


def pvslicer(datacube3d, xcen_pix, ycen_pix, major_axis_width, minor_axis_width, channelstart=None, channelend=None, positionangle=0):
    # takes slices of data in pixel coordinates
    # positionangle defined east of north
    # data needs to be [..., channel, dec, ra]
    # returns 
    data = np.squeeze(data)
    if channelstart is None:
        channelstart = 0
    if channelend is None:
        channelend = data.shape[0] - 1
    minor_axis_width = odd(minor_axis_width)
    major_axis_width = odd(major_axis_width)
    channel_slice = data[channelstart:channelend + 1, ...]
    pv = np.zeros((major_axis_width, channelend-channelstart))

    mask_base = np.zeros(datacube3d.shape[1:])
    width = int((minor_axis_width - 1) / 2)
    for pos in range(major_axis_width):
        # make ones at row below
        pos = int(pos - (major_axis_width - 1) / 2)
        mask_base[ycen_pix - pos, xcen_pix] = 1
        mask_r = skimage_rotate(image=mask_base, center=(xcen_pix, ycen_pix), angle=positionangle)
        mask_base[ycen_pix - pos, xcen_pix] = 0








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
    radii = constants.g * constants.msun * mass * np.sin(inc) ** 2 / (np.abs(velocity) * 100000.) ** 2
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
    velsqrd = constants.g * constants.msun * mass * np.sin(inc) ** 2 / radii
    velsqrd = velsqrd.astype(np.float64)
    vel = np.sqrt(velsqrd) / 100. / 1000.
    return vel


def infall_rad_2_vel(mass, rad, dist, inc, rcrit):
    """Infall.

    radii in arcsec
    mass in solar masses
    dist in pc
    inc in radians
    rc in au
    """
    radii = (rad / 3600. / 180. * constants.pi) * (Unit(baseunit='pc', convunit='cm', vals=dist).get_vals())
    mask = radii <= 0
    radii[mask] = 1e-10
    velsqrd = constants.g * constants.msun * mass * np.sin(inc) ** 2 * rcrit * constants.au / radii ** 2
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
        'positionangle': 'position angle of the object in degrees',
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
        'critical_radius': 'The critical radius for tthe envelope'
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


def pvdiagram(fig, ax, config, image):
    """
    ra, dec, arcsec_width in decimal deg
    pa, inc in deg
    width in pixels
    mass in msun
    vsys, vwidth in km/s
    dsource in pc
    image:
        Assuming image is a 3d array with first axis the freq/vel axis.

    
    Blue is North, aranged along y axis

    Reading in configuration and manipulating base values
    -----------------------------------------------------
    """
    default_config(config)
    print(f'Saving in: {config["savepath"]}')
    params_2_return = {**config, 'vsys_err':0, 'fitasym_red/blue': 0, 'fit_red': 0, 'fit_blue': 0}
    wcs = config['wcs']
    ra = config['ra']
    dec = config['dec']
    inc = Unit(vals=config['inclination'], baseunit='deg', convunit='radians').get_vals()
    pa = -config['positionangle']
    width = config['pixelwidth']
    width += width % 2 + 1
    mass_est = config['mass']
    mass_est_err = config['mass_err']
    v_sys = config['v_source']
    d_source = config['distance_source']
    arcsec_width = config['aswidth']
    v_width = config['vwidth']
    contour_params = config['contour_params']
    test = config['debug_mode']
    rc = config['critical_radius']
    timer = Timer(test)
    d_source_cm = Unit(vals=d_source, baseunit='pc', convunit='cm').get_vals()
    if test:
        for k, v in config.items():
            #print(f'{k}: {v}')
            pass


    def fit_env(v, mass):
        # v in km.s
        # m in sol m
        # returns fit in arcsec
        return fit_env2(v, mass, v_sys, rc)


    def fit_env2(v, mass, vsys, rcrit):
        # v in km.s
        # m in sol m
        # returns fit in arcsec
        if not isinstance(v, np.ndarray):
            v = np.array([v], dtype=np.float64)
        mask = v - vsys == 0
        v[mask] = 1e-10
        v = np.abs(v - vsys)
        v = Unit(vals=v, baseunit='km', convunit='cm').get_vals()
        a = np.sqrt(constants.g * constants.msun * mass * np.sin(inc) ** 2 * rcrit * constants.au / v ** 2)
        # a is in cm
        a *= 1. / d_source_cm
        # now in radians
        a *= Unit(vals=1., baseunit='radians', convunit='arcsec').get_vals()
        return a # now in arcsec


    def fit(v, mass):
        # v in km.s
        # m in sol m
        # returns fit in arcsec
        return fit2(v, mass, v_sys)

    def fit2(v, mass, vsys):
        # v in km.s
        # m in sol m
        # returns fit in arcsec
        if not isinstance(v, np.ndarray):
            v = np.array([v], dtype=np.float64)
        mask = v - vsys == 0
        v[mask] = 1e-10
        v = np.abs(v - vsys)
        v = Unit(vals=v, baseunit='km', convunit='cm').get_vals()
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
    freq_array = wcs.array(return_type='wcs', axis=wcs.axis3['type'])
    rf = config['restfreq'] * 1e9
    vel_array = (rf - freq_array) / rf
    vel_array *= constants.c / 100. / 1000.
    image = np.squeeze(image)

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
    minv = np.min(vel_array[vel_array_mask])
    maxv = np.max(vel_array[vel_array_mask])
    new_vcoord = np.linspace(minv, maxv, 11)
    ras = wcs(ra, return_type='pix', axis=wcs.axis1['type'])
    decs = wcs(dec, return_type='pix', axis=wcs.axis2['type'])
    # we now have the selection for the 3rd axis, work on other two
    lvel = wcs(0, 'wcs', 'freq')
    uvel = wcs(image.shape[0], 'wcs', 'freq')
    velcut_image = image[vel_array_mask, ...]
    mask = select_rectangle(velcut_image.T.shape[:-1], ycen = ras, xcen = decs, xlen=abs(arcsec_width / 2. / 3600. / wcs.axis1['delt']) + 2, ylen=abs(width - 1) / 2. + 2, pa=pa) # returns ra. dec,vel
    masked = velcut_image * mask.T[np.newaxis, :, :]
    unpad_image = remove_padding3d(masked)[0].T # returns vel, dec, ra
    rotated_image = rotate_image(unpad_image.T, axis=0, angle=pa + 90, resize=True, mode='constant', cval=np.nan).T # now in pos, pos, vel

    unpad_rotated_image = remove_padding3d(rotated_image.T)[0].T
    cut_rotated_image = unpad_rotated_image[..., 2:-2, 2:-2]

    # First quartile (Q1) 
    summed_image = np.nansum(cut_rotated_image, axis=1).T # in pos, vel
    if config['centerpv'] and not config['loaded']:
        rawfig = plotter('PV Diagram: centering')
        rawfig.int()

        yt = np.arange(summed_image.shape[0]).astype(int)
        xt = np.arange(summed_image.shape[-1]).astype(int)
        rawfig.open((1,1), xlabels=[f'{xti-np.mean(xt)}' if xti % 3 == 0 else '' for xti in xt], ylabels=[f'{yti-np.mean(yt)}' if yti % 3 == 0 else '' for yti in yt], xticks=xt, yticks=yt, xlabel=r'Velocity (pixels)', ylabel='Position (pixels)')
        cont = 1
        config['pcenter'] = summed_image.shape[0] / 2.
        pcenter = config['pcenter']
        while cont:
            pcenter = round(config['pcenter'], 0)
            rawfig.f[1].cla()
            rawfig.refresh()
            rawfig.imshow(summed_image, origin='lower', cmap='gray', interpolation='nearest', zorder=1)
            nstd = np.percentile(np.abs(summed_image), 65)
            rms = np.nanstd(summed_image[np.abs(summed_image) <= 3*nstd])
            lvls = np.linspace(rms, np.nanmax(summed_image), 10)
            rawfig.f[1].contour(summed_image, levels=np.sort(lvls))
            rawfig.f[1].axhline(pcenter, color='red', linestyle='--', lw=1)
            cont = 0
            if (input(f"Blue emission should be North, flip across y axis? : ")) in [' ', 'y', 'yes']:
                summed_image = np.flip(summed_image, axis=-1)
                cont = 1
            if (input(f"Blue emission should be North, flip across x axis? : ")) in [' ', 'y', 'yes']:
                summed_image = np.flip(summed_image, axis=0)
                cont = 1
            pcenter = (input(f"Enter New Position offset in pixel/[ret] to continue: ")).replace(' ', '')
            if pcenter != '':
                pcenter = round(float(pcenter) + config['pcenter'], 0)
                config['pcenter'] = pcenter
                config['redraw'] = True
                cont = 1
            else:
                pcenter = config['pcenter']
        config['poffsets'] = round(config['pcenter'] - summed_image.shape[0] / 2.)
        cont = 1
        config['vcenter'] = summed_image.shape[-1] / 2.
        vy = vel_array[vel_array_mask]
        polyf = np.polyfit(np.arange(vy.shape[0], dtype=int), vy, 1)
        while cont:
            vcenter = config['vcenter']
            rawfig.f[1].cla()
            rawfig.refresh()
            rawfig.imshow(summed_image, origin='lower', cmap='gray', interpolation='nearest', zorder=1)
            rawfig.f[1].contour(summed_image)
            rawfig.f[1].axvline(vcenter, color='red', linestyle='--', lw=1)
            vcenter = (input(f"Enter New Velocity offset in pixel/[ret] to continue: ")).replace(' ', '')
            if vcenter != '':
                vcenter = float(vcenter)
                config['vcenter'] += vcenter
                p = np.poly1d(polyf)
                config['v_source'] = p(config['vcenter'])
                v_sys = config['v_source']
                print(f'New Velocity Center: {v_sys:0.2f}')
                config['redraw'] = True
                cont = 1
            else:
                cont = 0
        rawfig.close()
    elif config['loaded']:
        summed_image = config['summed_image']
    else:
        config['pcenter'] = 0
        config['poffsets'] = 0
    if not config['fast'] and input('Fix velocity center? [y] to confirm/[ret] to continue: ').replace(' ', '').lower().startswith('y'):
        config['fixv'] = True
    else:
        config['fixv'] = False
    config['summed_image'] = summed_image
    pcenter = config['pcenter']
    poffsets = config['poffsets']
    ylen = summed_image.shape[0]
    posres = arcsec_width / ylen
    #from IPython import embed; embed()
    if poffsets != 0:
        summed_image = shift(summed_image, [-poffsets, 0], cval=np.nan)
    summed_image[summed_image == 0] = np.nan
    summed_image = np.squeeze(remove_padding3d(summed_image[..., None])[0])
    config['ra'] += poffsets * np.sin(pa) * (posres) / 3600.
    config['dec'] += poffsets * np.cos(pa) * (posres) / 3600
    v_sys = config['v_source']
    pvwcs = WCS({
        'naxis1': summed_image.shape[1],
        'naxis2': ylen,
        'ctype1': 'vel',
        'crpix1': 0,
        'crval1': minv,
        'cdelt1': np.ptp(new_vcoord) / summed_image.shape[1],
        'cunit1': 'km/s',
        'ctype2': 'dist',
        'crpix2': summed_image.shape[0] / 2.,
        'crval2': 0,
        'cdelt2': abs(arcsec_width / ylen),
        'cunit2': 'arcsec',
    })
    config['PV-WCS'] = pvwcs

    # plotting
    cax = ax.imshow(summed_image, origin='lower', cmap='cividis' if not config['grayscale'] else 'gray', interpolation='nearest', zorder=1)
    # plot moment map if requested
    nanmask = ~np.isnan(summed_image)
    rms = np.std(summed_image[nanmask])
    if not contour_params:
        mi = 3
        ma = 100
        ite = 3
        color='black'
    else:
        mi = contour_params['start']
        ma = contour_params['max']
        ite = contour_params['interval']
        color=contour_params['color']
    baserms = np.percentile(summed_image, 65)
    upperrms = np.percentile(summed_image, 95)
    rms = np.nanstd(summed_image[summed_image < upperrms])
    contour_levels = np.linspace(mi, ma, 4, dtype=float)
    contour_levels *= rms
    contour_levels += baserms
    try:
        contour = ax.contour(summed_image, levels=contour_levels, colors=color, zorder=20)
    except Exception:
        pass
    # formatting
    xticks = np.ravel([pvwcs.array(stop = 0, startstop_type='wcs', size=6, return_type='pix', axis='vel'), pvwcs.array(start = 0, startstop_type='wcs', size=6, return_type='pix', axis='vel')])
    xticks = np.unique(xticks)
    yticks = np.ravel([pvwcs.array(stop = 0, startstop_type='wcs', size=6, return_type='pix', axis='dist'), pvwcs.array(start = 0, startstop_type='wcs', size=6, return_type='pix', axis='dist')])
    yticks = np.unique(yticks)
    y_label = [f"{i:.1f}" for i in pvwcs(yticks, return_type='wcs', axis='dist')]
    x_label = [f"{i:.1f}" for i in pvwcs(xticks, return_type='wcs', axis='vel')]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(x_label, rotation=45)
    ax.set_yticklabels(y_label, rotation=45)
    cbar = None
    if cax is not None:
        cbar = fig.colorbar(cax)
    # gen array of offsets
    _t = pvwcs.array(axis='vel', return_type='wcs')
    hr_kepdist_pix = pvwcs.array(size=1000, axis='dist', return_type='pix')
    hr_kepdist_wcs = pvwcs.array(size=1000, axis='dist', return_type='wcs')
    mask = hr_kepdist_wcs != 0
    hr_kepdist_wcs = hr_kepdist_wcs[mask]
    hr_kepdist_pix = hr_kepdist_pix[mask]
    # split between left and right
    mask = hr_kepdist_wcs  > 0
    hr_blue_kepdist_wcs = hr_kepdist_wcs[mask]
    hr_red_kepdist_wcs = hr_kepdist_wcs[~mask]
    hr_blue_kepdist_pix = hr_kepdist_pix[mask]
    hr_red_kepdist_pix= hr_kepdist_pix[~mask]
    # now create velocity
    hr_blue_velocity_wcs = -1. * keplerian_rad_2_vel(mass_est, hr_blue_kepdist_wcs, config['distance_source'], inc) + v_sys
    hr_red_velocity_wcs = keplerian_rad_2_vel(mass_est, np.abs(hr_red_kepdist_wcs), config['distance_source'], inc) + v_sys
    hr_blue_velocity_pix = pvwcs(hr_blue_velocity_wcs, 'pix', 'vel')
    hr_red_velocity_pix = pvwcs(hr_red_velocity_wcs, 'pix', 'vel')
    # cccreatte velocity for infall
    hr_blue_velocity_wcs_infall = -1. * infall_rad_2_vel(mass_est, hr_blue_kepdist_wcs, config['distance_source'], inc, rc) + v_sys
    hr_red_velocity_wcs_infall = infall_rad_2_vel(mass_est, np.abs(hr_red_kepdist_wcs), config['distance_source'], inc, rc) + v_sys
    hr_blue_velocity_pix_infall = pvwcs(hr_blue_velocity_wcs_infall, 'pix', 'vel')
    hr_red_velocity_pix_infall = pvwcs(hr_red_velocity_wcs_infall, 'pix', 'vel')
    mask = hr_blue_velocity_pix > 0
    hr_blue_velocity_wcs = hr_blue_velocity_wcs[mask]
    hr_blue_velocity_pix = hr_blue_velocity_pix[mask]
    hr_blue_velocity_wcs_infall = hr_blue_velocity_wcs_infall[mask]
    hr_blue_velocity_pix_infall = hr_blue_velocity_pix_infall[mask]
    hr_blue_kepdist_pix = hr_blue_kepdist_pix[mask]
    hr_blue_kepdist_wcs = hr_blue_kepdist_wcs[mask]
    mask = hr_red_velocity_pix < pvwcs.axis2['axis']
    hr_red_velocity_wcs = hr_red_velocity_wcs[mask]
    hr_red_velocity_pix = hr_red_velocity_pix[mask]
    hr_red_velocity_wcs_infall = hr_red_velocity_wcs_infall[mask]
    hr_red_velocity_pix_infall = hr_red_velocity_pix_infall[mask]
    hr_red_kepdist_wcs = hr_red_kepdist_wcs[mask]
    hr_red_kepdist_pix = hr_red_kepdist_pix[mask]
    if config['plot_estimate']:
        ax.plot(hr_blue_velocity_pix, hr_blue_kepdist_pix,  color='red', label=r'Est. M$_{*}$=' + f'{mass_est:0.1f}' + r' M$_{\odot}$ @ V$_{sys}$=' + f'{v_sys:0.1f} ' + r'km s$^{-1}$', linestyle=':', linewidth=3, zorder=15)
        ax.plot(hr_red_velocity_pix, hr_red_kepdist_pix,  color='red', linestyle=':', linewidth=3, zorder=15)
        ax.plot(hr_blue_velocity_pix_infall, hr_blue_kepdist_pix,  color='red', label=r'Est. M$_{*}$=' + f'{mass_est:0.1f}' + r' M$_{\odot}$ @ V$_{sys}$=' + f'{v_sys:0.1f} ' + r'km s$^{-1}$ @ r$_{c}$=' + f'{rc:0.1f} ' + r'au', linestyle='-.', linewidth=3, zorder=15)
        ax.plot(hr_red_velocity_pix_infall, hr_red_kepdist_pix,  color='red', linestyle='-.', linewidth=3, zorder=15)
    if config['fit']:
        fit_ar_lines = [v.vertices for c in contour.collections for v in c.get_paths()]
        fit_ar_points = np.concatenate(fit_ar_lines)
        x = fit_ar_points[:, 0]
        y = fit_ar_points[:, 1]
        if os.path.isfile(f'{config["savepath"]}/bluepoints{config["targetname"]}.txt'):
            blue_fit_wcs = np.loadtxt(f'{config["savepath"]}/bluepoints{config["targetname"]}.txt', delimiter=';', dtype=float)
        else:
            blue_fit_wcs = np.zeros(0)
        if os.path.isfile(f'{config["savepath"]}/redpoints{config["targetname"]}.txt'):
            red_fit_wcs = np.loadtxt(f'{config["savepath"]}/redpoints{config["targetname"]}.txt', delimiter=';', dtype=float)
        else:
            red_fit_wcs = np.zeros(0)
        if config['redraw'] or ((red_fit_wcs.shape[0] == 0) and (blue_fit_wcs == 0)):
            rawfig = plotter('PV diagram: Lasso black points to fit')
            rawfig.int()
            rawfig.open((1,1), xlabels=x_label, ylabels=y_label, xticks=xticks, yticks=yticks, xlabel=r'Velocity (km s$^{-1}$)', ylabel='Position (arcsec)')
            rawfig.scatter(x, y, datalabel='scatter raw', color='cyan', s=2, zorder=15)
            for lines in fit_ar_lines:
                lines = np.array(lines)
                rawfig.plot(lines[:, 0], lines[:, -1], datalabel='_plotting_contours', color='cyan', lw=1, zorder=15)
            rawfig.imshow(summed_image, origin='lower', cmap='gray', interpolation='nearest', zorder=1)
            rawfig.plot(hr_red_velocity_pix, hr_red_kepdist_pix,  'red_hr_rawfig', color='red', label='Keplerian rotation', linestyle=':', linewidth=3, zorder=10)
            rawfig.plot(hr_blue_velocity_pix, hr_blue_kepdist_pix, 'blue_hr_rawfig', color='blue', linestyle=':', linewidth=3, zorder=10)
            rawfig.set_ylim(1, summed_image.shape[0] - 1)
            rawfig.set_xlim(1, summed_image.shape[1] - 1)
            # actual defining mask
            rawfig.f[1].axhline(pvwcs(0, 'pix', 'dist'), linestyle='--', color='red')
            rawfig.f[1].axvline(pvwcs(v_sys, 'pix', 'vel'), linestyle='--', color='red')
            blue_fit_pix = rawfig.selection('scatter raw', prompt='blue side').reshape(-1,2) # pix
            red_fit_pix = rawfig.selection('scatter raw', prompt='red side').reshape(-1,2) # pix
            rawfig.close()
            # vel, dist arrays
            blue_fit_wcs = np.array([pvwcs(blue_fit_pix[:, 0], 'wcs', 'vel'), pvwcs(blue_fit_pix[:, 1], 'wcs', 'dist')]).T
            red_fit_wcs = np.array([pvwcs(red_fit_pix[:, 0], 'wcs', 'vel'), pvwcs(red_fit_pix[:, 1], 'wcs', 'dist')]).T
            # now fit offset
            with open(f'{config["savepath"]}/bluepoints{config["targetname"]}.txt', 'w') as f:
                np.savetxt(f, blue_fit_wcs, delimiter=';')
            with open(f'{config["savepath"]}/redpoints{config["targetname"]}.txt', 'w') as f:
                np.savetxt(f, red_fit_wcs, delimiter=';')

        # normalize to upper quadrants
        # smooth data somehow
        fit_ar = []
        for a in [blue_fit_wcs, red_fit_wcs]:
            if a.shape[0] == 0:
                continue
            vel, arc = a[:, 0], a[:, 1] # wcs coords
            if a.shape[0] == 0 or not config['sg'] and not config['bin']:
                pass
            elif config['sg']:
                window = int(10**(np.log10(a.shape[0]) - 1))
                window = window if config['window'] == 0 else config['window']
                window = window if window % 2 == 1 else window + 1
                arc = savgol_filter(arc, window, 2, axis=-1)
            elif config['bin']:
                window = window if config['window'] else np.median(np.diff(np.sort(vel)))
                vel, arc = binning(vel, arc, windowsize=window)
            # weight with higher velocity = greater weight (disk fitting)
            # normed from 0 to 1
            fit_ar.append(np.array([vel, np.abs(arc)]))

        fit_ar = np.concatenate(fit_ar, axis=-1).T
        yerr = np.zeros(fit_ar.shape[0])

        # data has now been selected and time for fitting
        t_data = np.hstack([fit_ar, yerr[:, None]]) # should be dist, vel
        f_true = .5

        midest = config['v_source']
        if (red_fit_wcs.shape[0] == 0) or (blue_fit_wcs.shape[0] == 0) or config['fixv']:
            try:
                p0 = [config['mass']]
                lbound = [config['mass'] / 2.]
                ubound = [config['mass'] * 2.]
                popt, pcov = optimize.curve_fit(fit, np.abs(t_data[:,0]), np.abs(t_data[:,1]), signa=yerr, p0=p0, bounds=[lbound, ubound])
                # plot points used for fit
                perr = np.sqrt(np.diag(pcov)).tolist()
                popt = popt.tolist()
            except Exception as e:
                print(f'Fit failed: {e}, using fallback')
                popt = [config['mass']]
                perr = [0]
            hr_velocity_wcs_blue = -1. * np.arange(pvwcs.get('vel')['delt'], v_width + pvwcs.get('vel')['delt'], pvwcs.get('vel')['delt']) + v_sys
            hr_velocity_wcs_red = np.arange(pvwcs.get('vel')['delt'], v_width + pvwcs.get('vel')['delt'], pvwcs.get('vel')['delt']) + v_sys
            fit_kepl_wcs_blue = fit(hr_velocity_wcs_blue, *popt)
            fit_kepl_wcs_red = -fit(hr_velocity_wcs_red, *popt)
            params_2_return['vsys'] = config['v_source']
            params_2_return['vsys_err'] = 0
            params_2_return['mass'] = popt[0]
            params_2_return['mass_err'] = perr[0]
            fitfn = fit
        else:
            try:
                p0 = [np.log(config['mass']), midest]
                lbound = [p0[0] - 1., midest - 1]
                ubound = [p0[0] + 1, midest + 1]
                conversion = 1 / 3600 / 180 * np.pi / (d_source * constants.pc)
                #from IPython import embed; embed()
                popt, pcov = optimize.curve_fit(lambda v, logm, vsys: -2.* np.log(np.abs(v - vsys)) + logm + np.log(constants.g * constants.msun * np.sin(inc) ** 2 / conversion), t_data[:,0], np.log(np.abs(t_data[:,1])), p0=p0, bounds=[lbound, ubound])
                popt[0] = np.e ** popt[0]
                perr = np.sqrt(np.diag(pcov))
            except Exception as e:
                print(f'Fit failed: {e}, using fallback')
                popt = [config['mass'], config['v_source']]
                perr = [0, 0]
            hr_velocity_wcs_blue = -1. * np.arange(pvwcs.get('vel')['delt'], v_width + pvwcs.get('vel')['delt'], pvwcs.get('vel')['delt']) + popt[1]
            hr_velocity_wcs_red = np.arange(pvwcs.get('vel')['delt'], v_width + pvwcs.get('vel')['delt'], pvwcs.get('vel')['delt']) + popt[1]
            fit_kepl_wcs_blue = fit2(hr_velocity_wcs_blue, *popt)
            fit_kepl_wcs_red = -fit2(hr_velocity_wcs_red, *popt)
            params_2_return['vsys'] = popt[1]
            params_2_return['vsys_err'] = perr[1]
            params_2_return['mass'] = popt[0]
            params_2_return['mass_err'] = perr[0]
            fitfn = fit2
        # plot points used for fit
        v, o = t_data[:,0], t_data[:,1]
        vp = pvwcs(v, 'pix', 'vel')
        kp = pvwcs(o, 'pix', 'dist')
        #ax.scatter(vp, kp, color='orange', marker='.', zorder=15)

        # compute asym
        params_2_return['fitasym_red/blue'] = (o < 0).sum() / (o > 0).sum()
        fo = o[o < 0]
        fv = v[o < 0]
        fit_red = ((np.abs(fo) - np.abs(fitfn(fv, *popt))) / np.abs(fo)) ** 2
        fit_red = np.sum(fit_red)
        fo = o[o > 0]
        fv = v[o > 0]
        fit_blue = ((np.abs(fo) - np.abs(fitfn(fv, *popt))) / np.abs(fo)) ** 2
        fit_blue = np.sum(fit_blue)
        params_2_return['fit_red'] = fit_red
        params_2_return['fit_blue'] = fit_blue

        
        # fit savgol
        hr_velocity_pix_blue = pvwcs(hr_velocity_wcs_blue, 'pix', 'vel')
        hr_velocity_pix_red = pvwcs(hr_velocity_wcs_red, 'pix', 'vel')
        fit_kepl_pix_blue = pvwcs(fit_kepl_wcs_blue, 'pix', 'dist')
        fit_kepl_pix_red = pvwcs(fit_kepl_wcs_red, 'pix', 'dist')
        params_2_return['mass'] = popt[0]
        params_2_return['mass_err'] = perr[0]
        print(f"{Format('success')}Mass fit: {params_2_return['mass']:0.1e}+-{params_2_return['mass_err']:0.1e}, Vsys fit:{params_2_return['vsys']:0.1e}+-{params_2_return['vsys_err']:0.1e} {Format('reset')}")
        line = pvwcs(config['v_source'], 'pix', 'vel')
        print(line, pvwcs, config['v_source'])
        ax.axvline(line, color='red', linestyle='--')
        line = pvwcs(params_2_return['vsys'], 'pix', 'vel')
        ax.axvline(line, color='cyan', linestyle='--')
        line = pvwcs(0, 'pix', 'dist')
        #ax.axhline(line, color='white', linestyle='--')
        ax.plot(hr_velocity_pix_blue, fit_kepl_pix_blue,  color='cyan', label=r'Fit M$_{*}$=' + f' {params_2_return["mass"]:0.1f}'+r' M$_{\odot}$ @ ' + r'V$_{sys}$=' + f' {params_2_return["vsys"]:0.1f} km ' + r's$^{-1}$', linestyle=':', linewidth=3, zorder=20)
        ax.plot(hr_velocity_pix_red, fit_kepl_pix_red, color='cyan', linestyle=':', linewidth=3, zorder=20)
    if not config['fast'] and (input('[Ret] to continue/anything else to restart: ')) != '':
        return None, None, None
    return summed_image, cbar, params_2_return


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
