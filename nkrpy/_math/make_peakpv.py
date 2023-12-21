import numpy as np
import matplotlib.pyplot as plt
from nkrpy.io import fits
import pickle
from nkrpy import constants
from nkrpy.astro import WCS
from scipy.optimize import curve_fit



def skewgaussian(x, mu, fwhm, flux, c=0, skew = 1):

    sigma = fwhm / 1000. / (2. * np.sqrt(2. * np.log(2)))
    c = constants.c / 100.
    s = mu * sigma / c
    a = flux / np.sqrt(2. * np.pi) / s
    dx = (x - mu) / s
    dx[x > mu] /= skew
    if skew > 1:
        return 2. * a * np.exp(-0.5 * dx ** 2) / (1. + skew)
    return a * np.exp(-0.5 * dx ** 2)


parent = (lambda x: '/net/lovell/myhome3/reynolds/ALMA/ALMAc6-BHR7.2019.1.00463.S/originals/2019.1.00463.S/' + x)
saved = (lambda x: parent('science-images/' + x))


cfg = {}
def load(file, name):
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    wcs = data['wcs']
    pv = data['pv']
    d = data['data']
    std = np.nanstd(d[d < np.percentile(d[~np.isnan(d)], 90)])
    med = np.nanmedian(d[d < np.percentile(d[~np.isnan(d)], 90)])
    beam = wcs.get_beam()
    d[np.isnan(d)] = np.random.normal(loc=med, scale=2*std, size=d.shape)[np.isnan(d)]


    # fit each channel
    offsets = np.arange(0, pv.shape[1], dtype=float) * wcs.axis2['delt'] * 3600
    fits = []
    fitserror = []
    for channel in range(pv.shape[0]):
        flux = pv[channel, :]
        popt, pcov = curve_fit(skewgaussian, offsets, flux, p0=[0, 0.2, flux.max(), med + std, 0])
        perr = np.sqrt(np.diag(pcov))
        peakflux_offset = offsets[np.argmax(flux)]
        fits.append([channel, popt[0], popt[1], peakflux_offset])
        fitserror.append([perr[0], perr[1], 1  * wcs.axis2['delt'] * 3600])



    class mn:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    cfg[name] = mn(h=h, fits=fits, fitserror=fitserror, wcs=wcs, beam=beam, med=med, std=std)
    pass


load(file='pvdiagrams/12co-disk.pickle', name='12co')
load(file='pvdiagrams/13co.pickle', name='13co')
load(file='pvdiagrams/c18o.pickle', name='c18o')
load(file='pvdiagrams/h2co303.pickle', name='h2co1')
load(file='pvdiagrams/h2co321.pickle', name='h2co3')
load(file='pvdiagrams/h2co322.pickle', name='h2co2')
load(file='pvdiagrams/so42.pickle', name='so')
load(file='pvdiagrams/n2hp87.pickle', name='n2hp')
load(file='pvdiagrams/13cs42.pickle', name='13cs')
load(file='pvdiagrams/hnc42.pickle', name='hnc')


from IPython import embed; embed()














