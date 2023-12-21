#!/usr/bin/env python3

from pdspy.constants.physics import c
from pdspy.constants.astronomy import arcsec
from matplotlib.backends.backend_pdf import PdfPages
import pdspy.interferometry as uv
import pdspy.imaging as im
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import astropy.stats
import numpy as np
import sys
import os
import glob
import argparse
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', dest='overwrite', default=False)
parser.add_argument('--binned', action='store_true', dest='binned', default=False, help='Will only load hdf5 with binned in name')
parser.add_argument('--fit', action='store_true', dest='fit', default=False)
parser.add_argument('--showallfits', action='store_true', dest='showallfits', default=False)
parser.add_argument('--debug', action='store_true', dest='debug', default=False)
parser.add_argument('--fluxerror', dest='fluxerror', default=0, help='flux error to cut below', type=float)
parser.add_argument('--source', default='', dest='source')
parser.add_argument('--mol', default='', dest='mol')
parser.add_argument('--parentdir', dest='parentdir', default='', help='parent directory. Will save like parentdir/science-images/uv1d/...', type=str)
parser.add_argument('--loaddir', dest='loaddir', default='contms,linems', help='loaddir directory. Will load like parentdir/loaddir/*.hdf5. comma separate for multiple', type=str)
args = parser.parse_args()

parent = args.parentdir
if args.parentdir == '':
    parent = os.getcwd()
os.chdir(parent)
savedir = (parent+ '/science-images/uv1d/').replace('//', '/')
for i in range(1, savedir.count('/')):
    path = '/'.join(savedir.split('/')[:i+1])
    if not os.path.exists(path):
        print(path)
        os.mkdir(path)
# A list of sources that need special considerations.



from scipy.optimize import curve_fit
from scipy.special import j0, j1, spherical_jn

# functions in uv
def point_uv(uvdist, amp):
    return np.ones(uvdist.shape, dtype=float) * amp

def gauss_1duv(uvdist, sep, amp, c):
    # assuming sig is in image plane, sigma=sqrt(a*b)  in arcsec
    # uvdist in wavelength
    prod = np.pi * sep * uvdist / 3600 *  np.pi / 180
    return amp * np.e ** (-(prod) ** 2 / (4 * np.log(2))) + c

def uniformdisk_1duv(uvdist, sep, amp, c):
    # assuming sig is in image plane, sigma=sqrt(a*b)  in arcsec
    prod = np.pi * sep * uvdist / 3600 *  np.pi / 180
    return amp * 2 * j1(prod) / prod + c

def ring_1duv(uvdist, sep, width, amp, c):
    # assuming sig is in image plane, sigma=sqrt(a*b)  in arcsec
    # assuming width is in image plane,  in arcsec
    return uniformdisk_1duv(uvdist, sep+width/2, amp, 0) - uniformdisk_1duv(uvdist, sep-width/2, amp, 0) + c

def optthinsphere_1duv(uvdist, sep, amp, c):
    prod = np.pi * sep * uvdist / 3600 *  np.pi / 180
    return 3 / (prod ** 3) * (np.sin(prod) - prod * np.cos(prod)) + c

def gauss_2(uvdist, sep_gauss1, amp_gauss1, sep_gauss2, amp_gauss2, c):
    return gauss_1duv(uvdist, sep_gauss1, amp_gauss1, 0) + gauss_1duv(uvdist, sep_gauss2, amp_gauss2, 0) + c

def gauss_3(uvdist, sep_gauss1, amp_gauss1, sep_gauss2, amp_gauss2, sep_gauss3, amp_gauss3, c):
    return gauss_1duv(uvdist, sep_gauss1, amp_gauss1, 0) + gauss_1duv(uvdist, sep_gauss2, amp_gauss2, 0) + gauss_1duv(uvdist, sep_gauss3, amp_gauss3, 0) + c

def gauss_sph(uvdist, sep_sph, amp_sph, sep_gauss, amp_gauss, c):
    return optthinsphere_1duv(uvdist, sep_sph, amp_sph, 0) + gauss_1duv(uvdist, sep_gauss, amp_gauss, 0) + c
def ring_sph(uvdist, sep_sph, amp_sph, sep_ring, width_ring, amp_ring, c):
    return optthinsphere_1duv(uvdist, sep_sph, amp_sph, 0) + ring_1duv(uvdist, sep_ring, width_ring, amp_ring, 0) + c










fig = plt.figure(figsize=(7,7))
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5, 1], hspace=0, wspace=0)
cax = np.ravel(gs.subplots())
# Loop through the sources and fit.
allfiles = []
for load in args.loaddir.split(','):
    allfiles.extend(glob.glob(f'{parent}/{load}/*.hdf5'))

for sourcems in allfiles:
    filename = sourcems.split('/')[-1]
    if args.source != '' and not filename.lower().startswith(args.source.lower()):
        continue
    source = filename.split('cont')[0].split('calib')[0].strip('.').strip('_')
    rem = filename.replace(source, '')
    if rem.startswith('.contsub'):
        mol = filename.split('.')[-3]
    else:
        mol = 'continuum'
    if args.mol != '' and not mol.lower().startswith(args.mol.lower()):
        continue
    if args.binned and (filename == filename.replace('binned2', '')):
        continue

    for ax in cax:
        ax.cla()
    ax = cax[0]

    if not os.path.exists(savedir+mol):
        os.mkdir(savedir+mol)
    if not os.path.exists(savedir+source):
        os.mkdir(savedir+source)
    if not os.path.exists(savedir+'all'):
        os.mkdir(savedir+'all')

    if not os.path.exists(savedir + f"/all/{source}.{mol}.pdf") or args.overwrite:
        print('Working on ', filename)
        # Read in the data.
        if args.debug:
            print(sourcems)
        data = uv.Visibilities()
        data.read(sourcems)

        if args.debug:
            print('averaging')
        data_1d = uv.average(uv.center(data, [0,0]), gridsize=1024, \
                binsize=None, radial=True, \
                log=True, logmin=data.uvdist[data.uvdist > 0].min()*0.95, \
                logmax=data.uvdist[data.uvdist > 0].max()*1.05)

        if args.debug:
            print('sorting')
        ind = np.argsort(data_1d.uvdist)
        data_1d.uvdist = data_1d.uvdist[ind]
        data_1d.amp[:, 0] = data_1d.amp[ind, 0]
        data_1d.weights[:, 0]= data_1d.weights[ind, 0]
        ind = data_1d.amp[:, 0] > args.fluxerror
        data_1d.uvdist = data_1d.uvdist[ind]
        data_1d.amp = data_1d.amp[ind, ...]
        data_1d.weights= data_1d.weights[ind, ...]
        
        # Create a high resolution model for averaging.

        uvdist_max = data_1d.uvdist.max()

        if args.debug:
            print('modeling')
        u, v = np.meshgrid( np.linspace(-uvdist_max,uvdist_max,10000), \
                np.linspace(-uvdist_max,uvdist_max,10000))
        u = u.reshape((u.size,))
        v = v.reshape((v.size,))

        # Plot the visibilities.
        ax.errorbar(data_1d.uvdist/1000, data_1d.amp[:,0], \
                yerr=np.sqrt(1./data_1d.weights[:,0]),\
                fmt=".", markersize=1, markeredgecolor="black")

        # Plot the best fit model
        if args.fit:
            print('Fitting')
            plotstyles = [['blue', '--'],
                          ['red', '--'],
                          ['green', '--'],
                          ['yellow', '--'],
                          ['cyan', '--'],
                          ['purple', '--'],
                          ['black', '--'],
            ]
            x = np.linspace(data_1d.uvdist.min(), data_1d.uvdist.max(), 1000)
            sigma = np.abs(1./data_1d.weights[:,0])
            for i, [func, p0, bounds] in enumerate((
                    [gauss_1duv, (1, 0.05, 1), ((1e-4, 0, 0), (40, 40, 40))],
                    [uniformdisk_1duv, (1, 0.05, 1), ((1e-4, 0, 0), (40, 40, 40))],
                    [ring_1duv, (1,1, 0.05, 1), ((1e-4, 0,0, 0), (40,40, 40, 40))],
                    [optthinsphere_1duv, (1, 0.05, 1), ((1e-4, 0, 0), (40, 40, 40))],
                    [gauss_2, (1, 10, 0.5, 1, 0), ((1e-4, 0, 0.1, 0, 0), (40, 40, 40, 40, 40))],
                    [gauss_3, (1, 10, 0.5, 1, 0.5, 1, 0), ((1e-4, 0, 0.1, 0, 0.1, 0, 0), (40, 40, 40, 40, 40, 40, 40))],
                    [gauss_sph, (1, 10, 0.5, 1, 0), ((1e-4, 0, 0.1, 0, 0), (40, 40, 40, 40, 40))],
                    [ring_sph, (1, 10, 0.5, 0.1, 1, 1), ((1e-4, 0, 0.1, 0, 0, 0), (40, 40, 40, 40, 40, 40))],
                    )):
                try:
                    poptpcov = curve_fit(f=func,  xdata=data_1d.uvdist, ydata=data_1d.amp[:,0], sigma=sigma, p0=p0, bounds=bounds, method='trf')
                except Exception as e:
                    print(func, 'Failed', e)
                    continue
                y = func(x, *(poptpcov[0]))
                name = str(func)
                name = name.split('at')[0].replace('<function ', '').replace(' ', '')
                model = func(data_1d.uvdist, *(poptpcov[0]))
                residual = (model - data_1d.amp[:,0])
                reduc = model.shape[0] - len(p0)
                var = np.std(data_1d.amp[:,0])
                chi2 = ((residual / var) ** 2).sum() / reduc
                params_str = ','.join([f'{f:0.1e}' for f in poptpcov[0]])
                print(f'    {name}: chi2={chi2:0.1e}:  {params_str}')
                if chi2 > 4e-1 and not args.showallfits:
                    continue
                ax.plot(x/1000, y, linestyle=plotstyles[i%len(plotstyles)][1], color=plotstyles[i%len(plotstyles)][0], label=f'{name}: '+r'$\chi^{2}_{reduc}$='+f'{chi2:0.1e}', alpha=1, linewidth=1, zorder=20)
                cax[-1].plot(data_1d.uvdist/1000, np.abs(residual / model) * 100, linestyle=plotstyles[i%len(plotstyles)][1], color=plotstyles[i%len(plotstyles)][0], alpha=1, linewidth=1, zorder=20)
        for ax in cax:
            ax.label_outer()
        for ax in cax:
            ax.set_xlim((x/1000).min()*0.95,(x/1000).max()*1.1)
        cax[0].set_ylim(1e-5 if not args.fluxerror else args.fluxerror,5)


        for ax in cax:
            ax.set_xscale("log", nonpositive='clip')
            ax.set_yscale("log", nonpositive='clip')

        for ax in cax:
            continue
            print('xtick:', ax.get_xticks())
            print('xticklabel:', ax.get_xticklabels())
            print('ytick:', ax.get_yticks())
            print('yticklabel:', ax.get_yticklabels())

        cax[1].set_xlabel("U-V Distance [k$\lambda$]")
        cax[0].set_ylabel("Amplitude [Jy]")
        cax[1].set_ylabel("Residual [%]")
        cax[0].set_title(f'{source} {mol}')
        cax[0].legend()
        fig.savefig(savedir + f"/all/{source}.{mol}.pdf")
    os.system(f'ln -sf {savedir}all/{source}.{mol}.pdf {savedir}{mol}')
    os.system(f'ln -sf {savedir}all/{source}.{mol}.pdf {savedir}{source}')


plt.close('all')
