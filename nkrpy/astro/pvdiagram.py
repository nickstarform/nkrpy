"""."""
# flake8: noqa
# cython modules

# internal modules
import os
import sys
import itertools
from time import time

# external modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter

# relative modules
from .. import unit
from .. import constants as c
from ..io import fits
from . import WCS
from ..unit.convert import icrs2deg

# global attributes
__all__ = ('center_image', 'rotate_image', 'sum_image', 'keplerian_vel_2_rad',
           'pvdiagram', 'default_config')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


_count = itertools.count()
start = time()


def counter():
    global start
    net = time() - start
    start = time()
    return _count.__next__(), net


def count():
    print(counter())


def center_image(image, ra, dec, wcs):
    xcen = wcs(ra, 'pix', 'ra---sin')
    ycen = wcs(dec, 'pix', 'dec--sin')
    imcen = list(map(lambda x: x / 2, image.shape[1:]))
    print('centers: ', xcen, ycen, imcen)
    center_shift = list(map(int, [imcen[0] - ycen, imcen[1] - xcen]))
    print(center_shift)
    shift = [0]
    shift.extend(center_shift)
    shifted_image = inter.shift(image, shift)
    return shifted_image


def rotate_image(image, deg):
    rotated_image = inter.rotate(image, deg)
    return rotated_image


def sum_image(image, width: int):
    width = int(width / 2)
    center = list(map(lambda x: int(x / 2), image.shape))
    summed_image = np.sum(image[center[0] - width:center[0] + width, ...], axis=0)  # noqa
    return summed_image


def construct_savefile_name(params):
    r = ''.join(e for e in params["title"] if e.isalnum())
    r += f'-{params["positionangle"]:0.2f}-{params["inclination"]:0.2f}-{params["mass"]:0.2f}'
    r += f'-{params["distance_source"]:0.2f}-{params["v_source"]:0.2f}-{params["aswidth"]:0.2f}'
    r += f'-{params["vwidth"]:0.2f}-{params["pixelwidth"]:0.2f}'
    return r


def keplerian_vel_2_rad(mass, velocity, dist, inc):
    """Keplerian rotation.

    velocity in km/s
    mass in solar masses
    dist in pc
    inc in radians
    """
    mask = velocity == 0
    velocity[mask] = 1e-10
    radii = np.sin(inc) ** 2 * c.g * c.msun * mass / (np.abs(velocity) * 100000.) ** 2
    radii *= (180. / c.pi * 3600.) / (unit('pc', 'cm', dist).get_vals()[0])
    radii[velocity < 0] *= -1.
    velocity[mask] = 0
    return radii


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
        'contour_params': 'Dict with the keys start, interval, max to set contours',
        'debug_mode': 'True, False to verbosely output',
        'fit': 'True, False to run fitting routine.'
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
    pa, inc in radians
    width in pixels
    mass in msun
    vsys, vwidth in km/s
    dsource in pc
    image:
        Assuming image is a 3d array with first axis the freq/vel axis."""
    default_config(config)
    wcs = config['wcs']
    ra = config['ra']
    dec = config['dec']
    inc = config['inclination']
    pa = config['positionangle'] - 90.
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
    
    if test:
        count()
    try:
        axis = wcs.get_axis('freq')
        vel_array = wcs(np.arange(image.shape[len(image.shape) - axis], dtype=np.float), 'wcs', 'freq')
        rf = wcs.header[wcs.headlower['restfrq']]
        vel_array -= rf
        vel_array *= (-1. * c.c / 100. / 1000.) / rf
        # now in km / s
    except:
        vel_array = wcs(np.arange(image.shape[-1], dtype=np.float), 'wcs', 'vel')
        uni = [wcs.axis[f].kwargs['uni'] for f in wcs.axis if 'vel' in wcs.header[wcs.headlower[f]]]
        assert len(uni) == 1
        unit = uni[0].strip('/s')
        vel_array = unit(baseunit=uni, convunit='km', vals=vel_array).get_vals()

    if len(image.shape) == 4:
        image = image[0]
    assert len(image.shape) == 3

    coord_width = np.ceil(v_width / 10 / 0.25) * 0.25
    new_vcoord = (np.arange(0, 11) - 5) * coord_width + v_sys  # holds new velocity coord
    vel_array_mask = (vel_array > np.max(new_vcoord))
    vel_array_mask += (vel_array < np.min(new_vcoord))
    # we now have the selection for the 3rd axis, work on other two
    ras = wcs(np.arange(image.shape[1], dtype=np.float), 'wcs', 'ra---sin')
    decs = wcs(np.arange(image.shape[2], dtype=np.float), 'wcs', 'dec--sin')
    ras -= ra
    decs -= dec
    grid = np.meshgrid(ras * 3600., decs * 3600., indexing='ij')
    grid = (grid[0] ** 2 + grid[1] ** 2) ** 0.5
    dist_mask = np.where(grid < (arcsec_width / 2.))
    dist_mask = list(map(lambda x: np.unique(x), dist_mask))
    coord_width = np.ceil(arcsec_width / 10 / 0.25) * 0.25
    new_dcoord = (np.arange(0, 11) - 5) * coord_width  # holds new distance coord
    if test:
        print('New Velocities: ', new_vcoord)
        print('New Offsets   : ', new_dcoord)
        count()
        print(image.shape)
        if not os.path.isdir('testpv'):
            os.mkdir('testpv')
        testfig, testax = plt.subplots()
        vals = np.ravel(np.where(vel_array_mask == False))
        for i in range(np.min(vals) + int(vals.shape[0] / 4), np.min(vals) + int(vals.shape[0] / 1.5)):
            testax.imshow(image[i, :, :], origin='lower', cmap='cividis', interpolation='nearest')
            testfig.savefig(f'testpv/test_orig-{i}.png', bbox_inches='tight')
            testax.cla()
    lvel = wcs(0, 'wcs', 'freq')
    uvel = wcs(image.shape[0], 'wcs', 'freq')
    image = image[~vel_array_mask, ...][:, dist_mask[1], :][:, :, dist_mask[0]]
    if (uvel - lvel) > 0:
        image = np.flip(image, 0)
    if test:
        print(image.shape)
        count()
        for i in range(int(image.shape[0] / 4), int(image.shape[0] / 1.5)):
            testax.imshow(image[i, :, :], origin='lower', cmap='cividis', interpolation='nearest')
            testax.axhline(int(image.shape[1] / 2), color='white', linestyle='--')
            testfig.savefig(f'testpv/test_cut-{i}.png', bbox_inches='tight')
            testax.cla()
    rotated_image = rotate_image(np.nan_to_num(image.T), pa).T
    if test:
        print(rotated_image.shape)
        for i in range(int(rotated_image.shape[0] / 4), int(rotated_image.shape[0] / 1.5)):
            testax.imshow(rotated_image[i, :], origin='lower', cmap='cividis', interpolation='nearest')
            testax.axhline(int(rotated_image.shape[1] / 2), color='white', linestyle='--')
            testfig.savefig(f'testpv/test_rot-{i}.png', bbox_inches='tight')
            testax.cla()
        count()
    summed_image = sum_image(rotated_image.T, width).T
    if test:
        print(summed_image.shape)
        testax.imshow(summed_image, origin='lower', cmap='cividis', interpolation='nearest')
        ax.set_aspect(int(summed_image.shape[1] / summed_image.shape[0]))
        testfig.savefig('testpv/test_sum.png', bbox_inches='tight')
        testax.cla()
        count()
    print(arcsec_width, summed_image.shape)
    pixel_scale = arcsec_width / summed_image.shape[1]

    new_dcoord_pixel = new_dcoord / pixel_scale + summed_image.shape[1] / 2.

    # keplerian overplot
    vmin, vmax = np.min(new_vcoord), np.max(new_vcoord)
    dif = (vmax - vmin) / 1000.
    hr_velocity = np.arange(0, vmax - v_sys + dif, dif)
    hr_keplerian = keplerian_vel_2_rad(mass_est, hr_velocity, d_source, inc)
    mask = np.abs(hr_keplerian) >= np.max(new_dcoord)

    pvwcs = WCS({
        'ctype1': 'vel',
        'crpix1': summed_image.shape[0] / 2,
        'crval1': v_sys,
        'cdelt1': (vmax - vmin) / (summed_image.shape[0]),
        'cunit1': 'km/s',
        'ctype2': 'dist',
        'crpix2': summed_image.shape[1] / 2,
        'crval2': 0,
        'cdelt2': (np.max(new_dcoord) - np.min(new_dcoord)) / summed_image.shape[1],
        'cunit2': 'arcsec',
    })

    # plotting
    cax = ax.imshow(summed_image, origin='lower', cmap='cividis', interpolation='nearest')
    _hr_keplerian = pvwcs(hr_keplerian[~mask], 'pix', 'dist')
    _hr_velocity = pvwcs(hr_velocity[~mask] + v_sys, 'pix', 'vel')
    ax.plot(_hr_keplerian, _hr_velocity, color='white', label='Keplerian rotation', linestyle=':')
    _hr_velocity = pvwcs((-1. * hr_velocity[~mask]) + v_sys, 'pix', 'vel')
    _hr_keplerian = pvwcs(-1. * hr_keplerian[~mask], 'pix', 'dist')
    ax.plot(_hr_keplerian, _hr_velocity, color='white', linestyle=':')

    std_region = np.array([[0, 0], [0, 0]], dtype=int)
    std_region[:, 1] = np.array(summed_image.shape)
    std_region[:,1] = std_region[:, 1] / 4
    std_region = np.ravel(std_region)
    std = np.std(summed_image[std_region[0]:std_region[1],  # noqa
                              std_region[2]:std_region[3]])
    std = np.min([std, np.std(summed_image)])
    if std == 0:
        from IPython import embed
        # embed()
    print(f'STD: {std:0.2e}, {std_region}')
    contour_levels = np.sort(np.array([x * std for x in range(contour_params['max'])
                                       if x >= contour_params['start'] and
                                       ((x - contour_params['start']) %
                                       contour_params['interval']) == 0]))

    # embed()

    # formatting
    x_label = [f"{i:.1f}" for i in new_dcoord]
    y_label = [f"{i:.1f}" for i in new_vcoord]
    ax.set_xticks(np.linspace(0, summed_image.shape[1], 11))
    ax.set_yticks(np.linspace(0, summed_image.shape[0], 11))
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)
    cbar = plt.colorbar(cax)
    try:
        contour = ax.contour(summed_image, contour_levels, colors='white')
    except Exception:
        pass

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

        inc = unit('deg', 'rad', config['inclination']).get_vals()[0]
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
        fig.savefig(savefile + '.png', bbox_inches='tight')
    """
    print(string)

# end of code

# end of file
