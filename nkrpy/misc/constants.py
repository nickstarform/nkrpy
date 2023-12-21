"""Useful constants."""
# flake8: noqa
# internal modules

# external modules

# relative modules

# global attributes
__all__ = ['c','h','g','kb','a','sb','qe','ev','na','me','mp','mh2','mn','mh','amu','pi','golden','hour','day','year','calcPi','au','pc','yr','msun','rsun','lsun','msolsys','mmoon','mearth','rmoon','rearth','medd','j2000','kzen','kepler','pole_j2000_ra','pole_j2000_dec','posangle_j2000','pole_b1950_ra','pole_b1950_dec','posangle_b1950','jy','restfreq_hi','restfreq_co','cm2perkkms_hi','abun_ratio_c18o_c17o','abun_ratio_c18o_h2','abun_ratio_c17o_co','abun_ratio_co_h2','abun_ratio_h13cop_h2']
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# CGS PHYSICAL CONSTANTS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
cdef public float c    = 2.99792458e10      # speed of light CGS
cdef public float h    = 6.6260755e-27      # Planck's constant CGS
cdef public float g    = 6.67259e-8         # Grav const CGS
cdef public float kb   = 1.38064852e-16       # Boltzmann's const CGS
cdef public float a    = 7.56591e-15        # Radiation constant CGS
cdef public float sb   = 5.67051e-5         # sigma (stefan-boltzmann const) CGS
cdef public float qe   =  4.803206e-10      # Charge of electron CGS
cdef public float ev   =  1.60217733e-12    # Electron volt CGS
cdef public float na   =  6.0221367e23      # Avagadro's Number
cdef public float me   =  9.1093897e-28     # electron mass CGS
cdef public float mp   =  1.6726231e-24     # proton mass CGS
cdef public float mh2  = 3.3476e-24         # molecular hydrogen mass
cdef public float mn   = 1.674929e-24       # neutron mass CGS
cdef public float mh   = 1.673534e-24       # hydrogen mass CGS
cdef public float amu  =  1.6605402e-24     # atomic mass unit CGS
cdef public double pi   = 3.1415926535897932384626433832795028841
cdef public double golden = 1.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374
cdef public float hour = 3.6e3              # s
cdef public float day  = 8.64e4             # s
cdef public float year = 3.1557600e7        # s



# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# ASTRONOMICAL CONSTANTS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# GENERAL
cdef public float au      = 1.4959787066e13   # astronomical unit CGS
cdef public float pc      = 3.0857e18         # parsec CGS
cdef public float yr      = 3.155815e7        # sidereal year CGS
cdef public float msun    = 1.98900e+33       # solar mass CGS
cdef public float rsun    = 6.9599e10         # sun's radius CGS
cdef public float lsun    = 3.839e33          # sun's luminosity CGS
cdef public float msolsys = 1.00134198 * msun # solar system mas in CGS
cdef public float mmoon   = 7.35000e+25       # moon mass CGS
cdef public float mearth  = 5.97400e+27       # earth mass CGS
cdef public float rmoon   = 1.738e8           # moon radius CGS
cdef public float rearth  = 6.378137e8        # earth's radius CGS
cdef public float medd    = 3.60271e+34       # Eddington mass CGS
cdef public float j2000   = 2451545           # Julian date start j2000
cdef public float kzen    = 0.17              # typical zenith extinction
cdef public float kepler  = 7.495E-6          # [AU^3/day^2/Msun] kepler's constant

# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# ASTRONOMICAL CONSTANTS
# %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# North galactic pole (J2000) -- according to Wikipedia
cdef public float pole_j2000_ra  = 192.859508
cdef public float pole_j2000_dec = 27.128336
cdef public float posangle_j2000 = 122.932-90.0
# North galactic pole (B1950)
cdef public float pole_b1950_ra  = 192.25
cdef public float pole_b1950_dec = 27.4
cdef public float posangle_b1950 = 123.0-90.0

# RADIO SPECIFIC
cdef public float jy            = 1.e-23         # Jansky CGS
cdef public float restfreq_hi   = 1420405751.786 # 21cm transition (Hz)
cdef public float restfreq_co   = 115271201800.  # CO J=1-0 (Hz)
cdef public float cm2perkkms_hi = 1.823e18       # HI column per intensity (thin)

# abundances
cdef public float abun_ratio_c18o_c17o = 4.16
cdef public float abun_ratio_c18o_h2   = 1.7E-7
cdef public float abun_ratio_c17o_co   = 1./1700.
cdef public float abun_ratio_co_h2     = 10**-4
cdef public float abun_ratio_h13cop_h2 = 1.72E-11

# end of code

# end of file
