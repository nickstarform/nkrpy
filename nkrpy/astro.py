"""Various Astronomy functions."""

# internal modules

# external modules
import numpy as np

# relative modules
from .constants import h, c, kb, msun, jy
from . import kappa
from . import unit
from .functions import between

# global attributes
__all__ = ('dustmass', 'planck_nu', 'planck_wav')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1

c = c * 1E8  # A/s
h = h * 1E-7  # SI
kb = kb * 1E-7  # SI

"""
def Keplerian_Rotation(mass, velocity, Distance, inclination):
    radii_return =  np.sin(inclination)**2*const.G.value*mass*const.M_sun.value/(velocity*1000)/(velocity*1000)/(Distance*u.pc.to(u.m))*u.rad.to(u.arcsec) 
    #All the positive radii.
    radii_positive = radii_return[velocity < 0]
    #We also have some negative radii, so thats why we have to do this.
    radii_negative = -1*radii_return[velocity > 0]
    return radii_positive, radii_negative
"""
def planck_wav(temp=None, val=None, unit=None):
    """Plank Function in wavelength."""
    _c = unit(baseunit='angstroms', vals=c)('meters')
    _h = h
    wav = unit(baseunit=unit, vals=val)('meters')
    a = 2.0 * _h * _c ** 2
    b = _h * _c / (wav * kb * temp)
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
    # returns in units of watts/ m^2 / steradian / Hz
    return intensity * unit(baseunit='meters', vals=1)('hz')


def planck_nu(temp=None, val=None, unit=None):
    """Plank Function in frequency."""
    _c = unit(baseunit='angstroms', vals=c)('meters')
    _h = h

    nu = unit(baseunit=unit, vals=val)('hz')
    a = 2.0 * _h / _c ** 2
    b = _h * nu / (kb * temp)
    intensity = a * (nu ** 3) / (np.exp(b) - 1.0)
    # returns in units of watts/ m^2 / steradian / inputunit
    return intensity * unit(baseunit='hz', vals=1)(unit)


def dustmass(dist=100, dist_unit='pc', val=0.1,
             val_unit='cm', flux=0, temp=20,
             model_name='oh1994', beta=1.7, gas_density=0, opacity=None):
    """Calculate dust mass.

    @param dist, dist_unit, wavelength, wavelength_unit, flux, temp,model,beta, opacity
    Assuming temp in Kelvin, flux in Janskys"""
    dist = float(unit(baseunit=dist_unit, vals=dist)('cm'))  # to match the opacity units
    wav = float(unit(baseunit=val_unit, vals=val)('microns'))  # to search opcaity models
    intensity = planck_nu(temp, float(unit(baseunit=val_unit, vals=val)('hz')), 'hz') * 1.E26  # in jansky
    from IPython import embed
    #embed()
    if not opacity:
        opacity = kappa(wav, model_name=model_name, density=gas_density, beta=beta)  # cm^2/g
    else:
        opacity = [opacity]
    toret = 'For the various opacities:\n(cm^2/g)...(Msun)\n'
    _ret = []
    for x in opacity:
        _tmp = dist**2 * flux / x / intensity / msun  # noqa in msun units (no ISM assump.)
        # print(x, dist, flux, intensity, _tmp)
        toret += '{}...{}\n'.format(x, _tmp)
        _ret.append(np.array([x, _tmp]))
    return toret, np.array(_ret)


def true_emissive_mass(flux=None,freq=None,lam=None,distance=None,Tex=None):    
    # apply Goldsmith and Langer 1999 to convert each pixel into a mass
    # beam area
    #The below math does this: Jy*km/s / beam to Jy km/s / area to K*km/s / area  
    #  to num_mol/area to num_mol_h2/area to num_mol_h2/pix
    
    theta = 0.27*0.19            # arcsec^2
    theta = theta*np.pi * (1./(60.*60.))**2 * (np.pi/180.)**2# square degrees
    jybm2jy=(lam/10)**2*jy/(2.*theta*kb)*4.*0.693
    w=flux*jybm2jy   #  Jy*km/s / beam to Jy*km/s /area to K km/s / area
    #print('conv: ',jybm2jy)
    
    # Partition Function
    a = 2.321e-06    # s^-1
    B = 56179.99E6  # Hz
    z = kb*Tex / (h * B)
    #print('Partition: ' + str(z))
    j=3
    g = 2*j + 1
    mu = 0.11034
    E = h*B*(j*(j+1))
    #!print(E/k)    
    
    n = (8. * np.pi * kb * freq**2 * w )/(h * c**3 * a)

    #lnN_tot = np.log(n/g) + np.log(z) + (E / (kb * Tex))
    #N_tot=np.exp(lnN_tot)
    N_tot=n*z/g*(np.exp((1.0)*E/(kb*Tex)))* 1.0e5
    #print('N_tot: ',N_tot)
    N_mol=N_tot

    abun_ratio_c18o_c17o = float(4.16)
    abun_ratio_c18o_h2=float(1.7E-7)
    abun_ratio_c17o_co = float(1./2000.)
    abun_ratio_co_h2 = float(10**-4)
    abun_ratio_c17o_h2=1./(abun_ratio_c18o_h2/abun_ratio_c18o_c17o)

    N_mol_h2=N_mol*abun_ratio_c17o_h2
    #print('N_mol_h2: ', N_mol_h2)

    mol_mass = 2.71*mh*pixelarea # mean mol mass / avogadro #3.34E-24  # grams per H2 molecule
    #print('pixelarea: ',pixelarea)
    Mass_h2 = N_mol_h2 * mol_mass
    mass = Mass_h2 
    return mass # grams

def equivalent_width(spectra, blf, xspec0, xspec1, fit='gauss',
                     params=[1, 1, 1]):
    """Compute spectral line eq. width.

    Finds equivalent width of line
    spectra is the full 2d array (lam, flux)
    blf is the baseline function
    xspec0 (xspec1) is the start(end) of the spectral feature


    # PROBABLY EASIER TO JUST ASSUME GAUSSIAN OR SIMPLE SUM


    def gaussc(x, A, mu, sig, C):
        return A*np.exp(-(x-mu)**2/2./sig**2) + C

    from scipy.optimize import curve_fit

    featx, featy = lam[featurei], flux[featurei]
    expected1=[1.3, 2.165, 0.1, 1.]
    params1, cov1=curve_fit(gaussc, featx, featy, expected1)
    sigma=params1[-2]

    # I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)
    # sigma = dw / 2 / sqrt (2 * ln (core/Iw))
    # fwhm = 2.355 * sigma = 2(2ln2)^0.5 * sigma
    # flux = core * sigma * sqrt (2*pi)
    # eq. width = abs (flux) / cont

    fwhm = 2. * (2. * np.log(2.))**0.5 * sigma
    core, center, nsig, cont = params1
    flx = core * sigma * np.sqrt (2.*np.pi)
    eqwidth = abs(flx) / cont
    print(eqwidth)
    print(fwhm)
    """

    specfeatxi = np.array(between(spectra, xspec0, xspec1))[:, 0]
    specfeatxv = np.array(between(spectra, xspec0, xspec1))[:, 1]

    if fit == 'gauss':
        _params2, _cov2 = curve_fit(gauss, specfeatxv, spectra[specfeatxi, 1],
                                    *params)
        _sigma2 = np.sqrt(np.diag(_cov2))
        function = gauss(specfeatxv, *_expected2)
        return
    elif fit == 'ndgauss':
        pass


# end of file
