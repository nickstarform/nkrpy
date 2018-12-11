"""Reduction Pipeline for ARCSAT."""

# internal modules
from copy import deepcopy
from multiprocessing import Process
import multiprocessing as mp
from glob import glob
import os
import re
from argparse import ArgumentParser
import shutil

# external modules
from astropy.io import fits
import numpy as np
from PIL import Image
from IPython import embed
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# relative modules
from ..fits import read as nkrpy_read
from ..fits import write as nkrpy_write
from ..fits import header as nkrpy_header
from ...functions import list_files, find_nearest_above
from ...load import load_cfg, verify_param, verify_dir
from ...decorators import timing

# global attributes
__all__ = ('test', 'main')
__doc__ = """Handles bulk reduction for ARCSAT. Must have a config file defined and tries to do basic reduction quickly."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1
cpu = mp.cpu_count()


def cut(image, border=100):
    return image[border:-border, border:-border]
    """
    _t = np.zeros((list(map(lambda x: x - 2 * border, image.shape))))
    for i, row in enumerate(image):
        if border <= i < (len(image) - border):
            i -= border
            for j, col in enumerate(row):
                if border <= j < (len(row) - border):
                    j -= border
                    _t[i, j] = col
    return _t"""

@timing
def _gather_filters(files):
    """From list of files, return filters found."""
    """Assuming of style <science_filter_date.fits>
    or <flat_filter_date.fits>"""
    filt = []
    for x in files:
        _t = x.split('_')[1]
        if _t not in filt:
            filt.append(f'_{_t}_')
    filt = set(filt)
    return filt


@timing
def _gather_exp(files):
    """From list of files, return exposures found."""
    exp = {}
    for x in files:
        with fits.open(x) as f:
            header = f[0].header
        exptime = header['EXPTIME'] if header['EXPTIME'] \
            else header['EXPOSURE']
        if exptime not in list(exp.keys()):
            exp[exptime] = [x, ]
        else:
            exp[exptime].append(x)
    return exp


def plot_image(image, title, fig, ax, color=False):
    """General Plotting function."""
    shape = image.shape
    if False:  # color:
        r, g, b = image
        # r, g, and b are 512x512 float arrays with values >= 0 and < 1
        rgbArray = np.zeros((shape[1], shape[2]), 'uint8')
        rgbArray[..., 0] = r * 256.
        rgbArray[..., 1] = g * 256.
        rgbArray[..., 2] = b * 256.
        img = Image.fromarray(np.log10(rgbArray))
        img.save(title + 'log10_color.jpg')
        img = Image.fromarray(np.sqrt(rgbArray))
        img.save(title + 'sqrt_color.jpg')

    oimage = deepcopy(image)
    # image = deepcopy(oimage)
    image = np.abs(cut(image, 400))
    scale = int(image.shape[0] * image.shape[1] * 0.1)
    ii = np.unravel_index(np.argpartition(image.ravel(),
                                          scale)[-scale:],
                          image.shape)
    mask = np.ones(image.shape, dtype=bool)
    mask[ii] = False
    image[~mask] = float('nan')

    maxi = np.nanmax(image)
    med = np.nanmedian(image)
    std = np.nanstd(image)
    img1 = ax.imshow(np.log10(oimage), vmin=np.log10(med - std),
                     vmax=np.log10(med + 5. * std), origin='lower')
    fig.colorbar(img1, ax=ax)
    fig.savefig(title + 'log10.png', dpi=600)
    fig.clear()

    img1 = ax.imshow(np.sqrt(oimage), vmin=np.sqrt(med - std),
                     vmax=np.sqrt(med + 5. * std), origin='lower')
    fig.colorbar(img1, ax=ax)
    plt.savefig(title + 'sqrt.png', dpi=600)

    img1 = ax.imshow(oimage, vmin=med - std,
                     vmax=med + 5. * std, origin='lower')
    fig.colorbar(img1, ax=ax)
    plt.savefig(title + '.png', dpi=600)
    fig.clear()


@timing
def flats(files, header=None, bias_image=None, darks=None):
    """Construct flat frame."""
    al_files = deepcopy(files)
    filts = _gather_filters(files)
    masterflats = {}
    for filt in filts:
        files = [x for x in al_files if filt in x]
        with fits.open(files[0]) as f:
            header = f[0].header
        exptime = header['EXPTIME'] if header['EXPTIME'] \
            else header['EXPOSURE']
        if darks is not None:
            if float(exptime) not in list(darks.keys()):
                dark_image = scale_dark(files=darks,
                                        fin_exposure=exptime)
            else:
                dark_image = darks[float(exptime)]

        fdata = np.zeros((header['NAXIS1'], header['NAXIS2'], len(files)))
        for i, f in enumerate(files):
            ignored, fdata[:, :, i] = nkrpy_read(f)
            if dark_image is not None:
                fdata[:, :, i] -= dark_image
            if bias_image is not None:
                fdata[:, :, i] -= bias_image
        flat_comb = np.median(fdata, axis=2)
        flat_comb /= np.median(flat_comb)
        masterflats[filt] = flat_comb
        print(f'Finished filter: {filt}... med:{np.median(flat_comb)},'
              f'std:{np.std(flat_comb)}')
    return masterflats


@timing
def bias(files, header=None):
    """Construct bias frame."""
    if len(files) > 0:
        if not header:
            with fits.open(files[0]) as f:
                header = f[0].header

        bname = '_'.join(files[0].split('_')[0:2])

        bdata = np.zeros((header['NAXIS1'], header['NAXIS2'], len(files)))
        for i, f in enumerate(files):
            ignored, bdata[:, :, i] = nkrpy_read(f)
        bias_comb = np.median(bdata, axis=2)
    else:
        bias_comb = None
    return bias_comb


@timing
def darks(files, header=None, bias_image=None):
    """Construct dark frame."""
    if len(files) > 0:
        if not header:
            with fits.open(files[0]) as f:
                header = f[0].header

        masterdark_exp = {}
        ddata = np.zeros((header['NAXIS1'], header['NAXIS2']))

        for i, f in enumerate(files):
            ignored, ddata[:, :] = nkrpy_read(f)
            exptime = ignored['EXPTIME'] if ignored['EXPTIME'] \
                else ignored['EXPOSURE']
            if bias_image is not None:
                ddata[:, :] -= bias_image
            if float(exptime) not in list(masterdark_exp.keys()):
                masterdark_exp[float(exptime)] = [deepcopy(ddata),]
            else:
                masterdark_exp[float(exptime)].append(deepcopy(ddata))
            ddata = ddata * 0.
        for x in masterdark_exp:
            dark_comb = np.median(np.array(masterdark_exp[x]), axis=0)
            masterdark_exp[x] = dark_comb
            print(f'Finished exp: {x}')
    else:
        masterdark_exp = None
    return masterdark_exp


@timing
def scale_dark(files, fin_exposure):
    """Scale the dark."""
    masterdarks_exp = list(map(float, files.keys()))
    if fin_exposure in masterdarks_exp:
        return files[fin_exposure]
    else:
        use_dark = find_nearest_above(masterdarks_exp, fin_exposure)
        if use_dark is None:
            print(use_dark, masterdarks_exp, fin_exposure)
        if fin_exposure < use_dark[1]:
            scaler = fin_exposure / use_dark[1]
            _d = deepcopy(files[use_dark[1]])
            print(f'Scaled dark from: {use_dark[1]} to {fin_exposure}')
            files[fin_exposure] = _d * scaler
            return files[fin_exposure]
        else:
            print(f'You don\'t have a sufficient dark for {fin_exposure}.'
                  f'Your darklist: {masterdarks_exp}')
            return files[use_dark[1]]


@timing
def cal(cfg, science, bias_image, darks, flats):
    al_science = deepcopy(science)
    filts = _gather_filters(science)
    for filt in filts:
        if filt in flats:
            files = [x for x in al_science if filt in x]
            files.sort()
            for i, f in enumerate(files):
                header, data = nkrpy_read(f)
                exptime = header['EXPTIME'] if header['EXPTIME'] \
                    else header['EXPOSURE']
                if darks is not None:
                    dark_image = scale_dark(files=darks,
                                            fin_exposure=exptime)
                if bias_image is not None:
                    data -= bias_image
                if dark_image is not None:
                    data -= dark_image
                data /= flats[filt]

                dest = os.path.join(os.getcwd(), cfg.destination)
                _temp = f'{dest}/{"_".join(f.split("_")[0:2]).split("/")[-1]}_{i}.fits'
                if cfg.createfits:
                    while not nkrpy_write(f'{_temp}', header=header,
                                          data=data):
                        continue
                
                if cfg.createplot:
                    os.chdir(os.path.join(cfg._cwd,
                                          cfg.destination))

                    fig = plt.figure(figsize=(10, 10))
                    font0 = FontProperties()
                    font = font0.copy()
                    font.set_weight('bold')
                    ax = fig.add_subplot(1,1,1)
                    ax.tick_params(axis='x', labelsize=25)
                    ax.tick_params(axis='y', labelsize=25)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    plot_image(data, _temp.split('/')[-1], fig,
                               ax, color=False)
                os.chdir(cfg._cwd)
    pass


def main(cfgname, clear=False):
    """Main caller function."""
    # load in configuration
    cfg = load_cfg(cfgname)
    cfg._cwd = os.getcwd()
    if cfg.work:
        os.chdir(cfg.work)
        cfg._cwd = os.getcwd()

    if not verify_param(cfg, f'{__path__}/template_config.py'):
        exit()
    # load up directories
    for x in [cfg.flats, cfg.darks, cfg.bias, cfg.science]:
        verify_dir(x, create=False)
    for x in [cfg.calibration, cfg.destination]:
        if clear and os.path.isdir(x):
            shutil.rmtree(x)
        verify_dir(x, create=True)

    bias_i = glob(f'{cfg.bias}/*.fits')
    dark_i = glob(f'{cfg.darks}/*.fits')
    flat_i = glob(f'{cfg.flats}/*.fits')
    science_i = glob(f'{cfg.science}/*.fits')
    filts = _gather_filters(science_i)

    print('Starting Cals')
    masterbias = bias(bias_i)
    print('Finished Bias')
    masterdarks = darks(dark_i, bias_image=masterbias)
    print('Finished Darks')
    masterflats = flats(flat_i, bias_image=masterbias, darks=masterdarks)
    print('Finished Flats')

    os.chdir(os.path.join(cfg._cwd, cfg.calibration))
    if cfg.createfits:
        while not nkrpy_write(f'master_bias.fits', header=None,
                              data=masterbias):
            continue
        for x in masterdarks:
            while not nkrpy_write(f'masterdark_E{x}.fits', header=None,
                                  data=masterdarks[x]):
                continue
        for x in masterflats:
            while not nkrpy_write(f'masterflat_F{x}.fits', header=None,
                                  data=masterflats[x]):
                continue

    os.chdir(cfg._cwd)
    masterscience = cal(cfg, science_i, bias_image=masterbias,
                        darks=masterdarks, flats=masterflats)
    print('Finished Science')
    pass


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""
    parser = ArgumentParser(__doc__)
    parser.add_argument('-i', '--input', dest='i', type=str, help='configfile')
    parser.add_argument('-g', '--gen', dest='g', default=False,
                        action="store_true", help='generate config')
    parser.add_argument('-r', '--resume', dest='r', default=False,
                        action="store_true", help='pickup reduction where left off')
    args = parser.parse_args()
    if args.g:
        shutil.copyfile(f'{__path__}/template_config.py', 'template_config.py')
        exit()

    if args.i:
        print('Running module')
        if args.r:
            main(args.i)
        else:
            main(args.i, True)
        print('Module Passed')
    else:
        print('Must specify configfile')
else:
    raise(ImportError, 'This program must be run as a file.')

# end of code

"""
WORKFLOW

read in config file
config has flats, darks, bias, science, createplot, createfits, destination, color
the flats, darks, bias, science will all be in separate directories
then make a cal dir and a processed dir
defaults to only use the filters in science.
if corresponding flat not found don't reduce that image

setup multiprocessing

loop through bias and construct
loop through darks and construct
loop through flats and construct
now loop through science and construct
https://stackoverflow.com/questions/31862958/how-to-merge-images-together-in-python-with-pil-or-something-else-with-each-im?rq=1
http://www.noah.org/wiki/Wavelength_to_RGB_in_Python



##############
things to work on 
multiprocessing
ability to resume
plotting

"""
