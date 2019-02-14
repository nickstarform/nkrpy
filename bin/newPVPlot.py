#!/usr/bin/env python

import emcee
import scipy.optimize as op
import corner
import matplotlib.pyplot as plt
import numpy as np
from nkrpy.constants import c, msun
from IPython import embed
from nkrpy.coordinates import checkconv
from astropy import units as u
from astropy import constants as const
from nkrpy.apo import fits
import Nice_Plots_2
from matplotlib.ticker import FormatStrFormatter
from nkrpy.decorators import deprecated

filein = 'PV-Diagram_L1448IRS3B_C17O_image_taper1500k.image.txt'
File = '../L1448IRS3B_C17O_image_taper1500k.image.fits'
title = r'L1448IRS3B C$^{17}$O'
ra_orig = '03 25 36.329'
dec_orig = '30 45 15.042'
rotation_deg = -62 #degrees -62
yerr = 0.05
dis_err = 10
v_err = 0.11
imagemin = -0.5
imagemax = 3
v_width  = 10
arcsec_width = 10
v_source = 4.8
offset = 0
contour_interval = 3
contour_min = 10
contour_max = 24
mass = 1.1
mass_err = 0
d_source = 288.
inclination = 45.
plot_num_fit = True
plot_eye_fit = False
plot_orbital = False
cut = 12
fit_pv = True

ra,dec=checkconv(ra_orig)*15.,checkconv(dec_orig)
PV_Data = np.loadtxt(filein)

print('Everything loaded, now computing PV diagram')

plt.ion()
fig = plt.figure(figsize=(10,10))
Nice_Plots_2.set_style()
ax = fig.add_subplot(111)
cax = ax.imshow(PV_Data, origin = 'lower', cmap = 'magma',vmin=imagemin,vmax=imagemax,interpolation='nearest')
ax.set_title(title)
ax.set_xlabel('Velocity (km s$^{-1}$)')
ax.set_ylabel('Offset ($^{\prime\prime}$)')#'Position [arcsec]')$\Delta$X

#Getting the data.
header, data = fits.read(File)
#The datacubes coming from the thindisk model have 3 dimensions, 
#while the science datacubes have 4 dimension. So we have to account
#for that. 
if len(np.shape(data)) == 4:
    Data = data[0,:,:,:]
else:
    Data = data

#Determining the shape of the data.

Shape_Data = Data.shape


N = header['NAXIS3']
if header['CTYPE3']  == 'VELO-LSR':
    print('True')
    begin_v = header['LSTART']
    delta_v = header['LWIDTH']
elif header['CTYPE3']  == 'FREQ':
    #Reading the data in frequencies. We have to bring this to velocities. 
    begin_freq = header['CRVAL3']
    delta_freq = header['CDELT3']
    begin_pos = header['CRPIX3'] - 1
    rest_freq = header['RESTFRQ']
    #The speed of light is.
    c = c/100/1000.#km s^-1
    #Calculating the begin velocity.
    begin_v = c * (np.square(rest_freq) - np.square(begin_freq - delta_freq*begin_pos)) / ( np.square(rest_freq) + np.square(begin_freq - delta_freq*begin_pos))
    #Now we calculate the delta v
    begin_v_plus_one = c * (np.square(rest_freq) - np.square(begin_freq - delta_freq*(begin_pos + 1))) / ( np.square(rest_freq) + np.square(begin_freq - delta_freq*(begin_pos + 1)))
    delta_v = np.round(begin_v - begin_v_plus_one, 2)
    delta_v =begin_v - begin_v_plus_one

PixelWidth_RA = header['CDELT1']
PixelWidth_DEC = header['CDELT2']

rotation_rad = np.radians(rotation_deg)
y_size = np.round(np.abs(np.cos(np.abs(rotation_rad))*Shape_Data[1]) + np.abs(np.cos(np.pi/2 - np.abs(rotation_rad))*Shape_Data[2]))

length_arcsec_new =  (np.abs(np.cos(np.abs(rotation_rad)))*np.abs(PixelWidth_RA)*Shape_Data[2]+np.abs(np.cos(np.pi/2.0-np.abs(rotation_rad)))*np.abs(PixelWidth_DEC)*Shape_Data[1])*3600


x_values = np.arange(begin_v, begin_v + delta_v*float(Shape_Data[0]), delta_v)

#The total length in arcsec of the y axis in the new image.
length_arcsec_new =  (np.abs(np.cos(np.abs(rotation_rad)))*np.abs(PixelWidth_RA)*Shape_Data[2]+np.abs(np.cos(np.pi/2.0-np.abs(rotation_rad)))*np.abs(PixelWidth_DEC)*Shape_Data[1])*3600
y_values = np.arange(-length_arcsec_new/2.0, length_arcsec_new/2.0 + length_arcsec_new/10.0, length_arcsec_new/10.0)

#Calculating the size of y pixel in the y direction in arcsec.
pixel_size_y_arcsec = length_arcsec_new/y_size

y_arcsec = np.arange(1, PV_Data.shape[0])*pixel_size_y_arcsec - length_arcsec_new/2.0

pix_v_source = float(np.abs((begin_v - v_source)/delta_v))
#Then we determine what half the width of the v slice must be.
pix_v_shift = float(v_width/delta_v/2.0)
#print(pix_v_source, pix_v_shift,v_width,delta_v
#Now we determine the central pixel for the arcsec.
pix_arcsec_central = float(y_size/2.0) - 1.0 + float(offset)
pix_arcsec_shift = float(arcsec_width/pixel_size_y_arcsec/2.0)

start = pix_v_source - pix_v_shift
end = pix_v_source + pix_v_shift
num = 11.
step = (end - start)/num
x = np.arange(start, end + step, step)[:int(num+1)]
start = pix_arcsec_central - pix_arcsec_shift
end = pix_arcsec_central + pix_arcsec_shift
num = 11.
step = (end - start)/num
y = np.arange(start, end+step, step)

x = np.arange(pix_v_source - pix_v_shift, \
    pix_v_source + pix_v_shift + 2.0*pix_v_shift/10.0, \
    2.0*pix_v_shift/10.0)
#y = np.arange(pix_arcsec_central - pix_arcsec_shift, pix_arcsec_central + pix_arcsec_shift + 2*pix_arcsec_shift/10., 2*pix_arcsec_shift/10)
y = np.linspace(pix_arcsec_central - pix_arcsec_shift,\
    pix_arcsec_central + pix_arcsec_shift, 11.0)


x_label = np.round(begin_v + delta_v*x, 1)
#y_label = np.round((y+1)*pixel_size_y_arcsec - length_arcsec_new/2., 2) + 0.01
y_label = np.linspace(-arcsec_width/2.0, arcsec_width/2.0, 11.0)


#ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
y_label = np.array(["%.1f" % i for i in y_label])
ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(x_label)
ax.set_yticklabels(y_label)
#Doing the zooming by limiting the shown x and y coordinates.
#move after the labels, because those functions automatically adjust x and y limits.
#print(pix_v_source - pix_v_shift, pix_v_source + pix_v_shift,pix_v_source
#print('help!'
ax.set_xlim([pix_v_source - pix_v_shift, pix_v_source + pix_v_shift])
print([pix_v_source - pix_v_shift, pix_v_source + pix_v_shift])
print([pix_arcsec_central - pix_arcsec_shift, pix_arcsec_central + pix_arcsec_shift])
ax.set_ylim(pix_arcsec_central - pix_arcsec_shift, pix_arcsec_central + pix_arcsec_shift) 
ax.set_aspect(1.0*pix_v_shift/pix_arcsec_shift)


#Creating an array containing the velocities in km s^-1. 
velocities = (np.arange(begin_v, begin_v + delta_v*float(Shape_Data[0]), delta_v) - v_source)
#If we have correct masses we can calculate the velocity curve.
print('Including the velocities curves.')
#Calculating the extreme masses within the errors, do we can also plot
#those. 

#This function returns for a given mass (solar masses), velocity (in
#km s^-1) and distance to the source (in pc) the radius (in arcsec)  
#assuming Keplerian rotation.
def Keplerian_Rotation(mass, velocity, Distance, inclination):
    radii_return =  np.sin(inclination*np.pi/180.)**2*const.G.value*mass*const.M_sun.value/(velocity*1000)/(velocity*1000)/(Distance*u.pc.to(u.m))*u.rad.to(u.arcsec) 
    #All the positive radii.
    radii_positive = radii_return[velocity < 0]
    #We also have some negative radii, so thats why we have to do this.
    radii_negative = -1*radii_return[velocity > 0]        
    return radii_positive, radii_negative

#Plotting the velocities
if plot_eye_fit:
    mass_min_err = mass - mass_err
    mass_plus_err = mass + mass_err

    #print(velocities)

    #Calculate the radii.
    radii_positive, radii_negative = Keplerian_Rotation(mass, velocities, d_source, inclination)
    radii_positive_min_err, radii_negative_min_err = Keplerian_Rotation(mass_min_err, velocities, d_source, inclination)
    radii_positive_plus_err, radii_negative_plus_err = Keplerian_Rotation(mass_plus_err, velocities, d_source, inclination)

    #Changing the radii to the correct pixel coordinates for correct 
    #plotting. Plus bring the lines to the object. 
    radii_positive_pixel_coor = radii_positive/pixel_size_y_arcsec + (y_size - 1.0)/2.0  + offset
    radii_negative_pixel_coor = radii_negative/pixel_size_y_arcsec + (y_size - 1.0)/2.0  + offset

    radii_positive_min_err_pixel_coor = radii_positive_min_err/pixel_size_y_arcsec + (y_size - 1.0)/2.0 + offset 
    radii_negative_min_err_pixel_coor = radii_negative_min_err/pixel_size_y_arcsec + (y_size - 1.0)/2.0 + offset

    radii_positive_plus_err_pixel_coor = radii_positive_plus_err/pixel_size_y_arcsec + (y_size - 1.0)/2.0 + offset  
    radii_negative_plus_err_pixel_coor = radii_negative_plus_err/pixel_size_y_arcsec + (y_size - 1.0)/2.0 + offset

    ax.plot(np.arange(0,len(radii_positive), 1), radii_positive_pixel_coor, color = 'white', linestyle = ':')
    ax.plot(np.arange(len(radii_positive) , len(velocities), 1), radii_negative_pixel_coor, color = 'white', linestyle = ':')

    ax.plot(np.arange(0,len(radii_positive), 1), radii_positive_min_err_pixel_coor, color = 'white', linestyle = ':')
    ax.plot(np.arange(len(radii_positive) , len(velocities), 1), radii_negative_min_err_pixel_coor, color = 'white', linestyle = ':')

    ax.plot(np.arange(0,len(radii_positive), 1), radii_positive_plus_err_pixel_coor, color = 'white', linestyle = ':')
    ax.plot(np.arange(len(radii_positive) , len(velocities), 1), radii_negative_plus_err_pixel_coor, color = 'white', linestyle = ':')

ax.axhline(np.where(y_arcsec > 0)[0][0] - 1 + offset, color = 'white', linestyle = '--')
ax.axvline(np.where(velocities > 0)[0][0] - 0.33 , color = 'white', linestyle = '--')
#ax.legend(loc = 3)

#---------------------------------------------------------------------------
#Contour lines
#---------------------------------------------------------------------------
PVSHAPE = PV_Data.shape
contour_region = (60,70,1200,1300)
print(contour_region)
plot_c_region = True
std_PV = np.std(PV_Data[contour_region[2]:contour_region[3],contour_region[0]:contour_region[1]]) 
print(f'Std:{std_PV}')

PV_Contour_Levels = np.array([x * std_PV for x in range(contour_max) if x >= contour_min and ((x - contour_min) % contour_interval) == 0])


# plot cyan
cs = ax.contour(PV_Data, PV_Contour_Levels, colors = 'white')

def pix_2_arc(pix):
    return (pix - pix_arcsec_central) * pixel_size_y_arcsec

def pix_2_vel(pix):
    return (pix - pix_v_source ) *delta_v + v_source

def arc_2_pix(arc):
    return (arc / pixel_size_y_arcsec) + pix_arcsec_central

def vel_2_pix(arc):
    return ((arc - v_source) / delta_v) + pix_v_source

def get_dupes(a):
    from collections import Counter
    return np.array([item for item, count in Counter(a).iteritems() if count > 1])

def fit(v, m):
    a = np.sin(inclination*np.pi / 180.)**2*const.G.value*m*const.M_sun.value/(v*1000)/(v*1000)/(d_source*u.pc.to(u.m))*u.rad.to(u.arcsec)
    return a
def inv(x, a, b, c):
    return a / (x - b) ** 2 + c

def maxy(a):
    x,y = a[:,0], a[:,1]
    retx = list(set(x))
    ret = []
    for col in retx:
        idx = np.where(x == col)[0]
        maxy = np.max(y[idx])
        ret.append(np.array([col, maxy]))
    return np.array(ret)

def lnlike(theta, x, y, yerr):
    m, lnf = theta
    model = fit(x, m)
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    m, lnf = theta
    if 0.1 < m < 2. and -10.0 < lnf < 5.0:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def closest_approach(node, nodes):
    a = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(a)

@deprecated
def error_prop(r,v,m):
    a = (v*1000)**2/ const.G.value / (np.sin(inclination * np.pi / 180.) ** 2) / u.rad.to(u.arcsec)
    l1 = (r * dis_err + d_source * yerr) * a * u.pc.to(u.m)
    l2 = 2. * m * const.M_sun.value * (v_err / v)
    return l1 + l2


print('Running EMCEE Fitter')
if fit_pv:
    cuttoff = cut * std_PV
    prev = 0
    fit_pt = []
    for i, row in enumerate(PV_Data):
        row = np.array(row)
        regions = np.where(row > cuttoff)
        if len(regions[0]) > 0:
            if i < pix_arcsec_central:
                # bottom
                loca = np.max(regions[0])
            else:
                # top
                loca = np.min(regions[0])
            prev = loca
            fit_pt.append(np.array([loca,i]))
    rows_cut = len(fit_pt)
    for i, col in enumerate(PV_Data.T):
        col = np.array(col)
        regions = np.where(col > cuttoff)
        if len(regions[0]) > 0:
            if i < pix_v_source:
                # bottom
                loca = np.max(regions[0])
            else:
                # top
                loca = np.min(regions[0])
            prev = loca
            fit_pt.append(np.array([loca,i]))
    fit_pt = np.array(fit_pt,dtype=float)
    fit_o = fit_pt.copy()
    _, idx = np.unique(fit_pt[:,0], return_index=True)
    mask = np.ones(fit_pt[:,0].shape,dtype=bool)
    mask[idx] = False
    fit_pt = fit_pt[mask, :]

    with open('points.txt', 'w') as f:
        fit_ar = np.array(fit_pt,dtype=float)
        fit_ar[:,0] = pix_2_vel(fit_ar[:,0])
        fit_ar[:,1] = pix_2_arc(fit_ar[:,1])
        idx = np.where(fit_ar[:,0] <= 10)[0]
        x = fit_ar[idx,0] 
        y = fit_ar[idx,1]
        fit_ar = np.array([x,y]).T
        idx = np.where(fit_ar[:,1] <= 5)[0]
        x = fit_ar[idx,0] 
        y = fit_ar[idx,1]
        fit_ar = np.array([x,y]).T
        np.savetxt(f, fit_ar, delimiter=';')

    print(f'v_source, pixel_size_y_arcsec, pix_v_source,pix_v_source,delta_v={",".join(list(map(str,[v_source, pixel_size_y_arcsec, pix_v_source,pix_v_source,delta_v])))}')

    t_data = fit_ar.copy()
    t_data[:,1] = np.abs(fit_ar[:,1])
    t_data[:,0] -= v_source

    t_data = maxy(t_data)
    idx = np.where(t_data[:,1] >= yerr)[0]
    t_data = t_data[idx,:]
    x = vel_2_pix(t_data[:,0]+v_source)
    trans = t_data.copy()
    for i,loca in enumerate(x):
        if i < rows_cut:
            c = 'darkcyan'
        else:
            c = 'cyan'
        if loca < pix_v_source:
            tmp = arc_2_pix(t_data[i,1])
            ax.scatter(loca, tmp, color=c,lw=1,marker='.')
        else:
            tmp = arc_2_pix(-1.*t_data[i,1])
        ax.scatter(loca, tmp, color=c,lw=1,marker='.')
        trans[i,:] = np.array([loca,tmp])
    # embed()
    # now convert t_data to fit_ar
    xr = np.linspace(pix_v_source+2, PVSHAPE[1], 1000)
    xl = np.linspace(0, pix_v_source-2, 1000)

    ndim, nwalkers = 2, 100
    x = t_data[:,0]
    y = t_data[:,1]
    m_true = mass
    f_true = yerr
    
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [m_true, np.log(f_true)], args=(x, y, yerr))
    m_ml, lnf_ml = result["x"]
    pos = [result["x"] + yerr*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos, 2000)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    m_true = np.median(samples[:,0])
    f_true = np.median(samples[:,1])
    print('Plotting Emcee Fits')

    c = 0
    for m, lnf in samples[np.random.randint(len(samples), size=100)]:
        ax.plot(xl, inv(xl, m /pixel_size_y_arcsec / delta_v**2, pix_v_source, pix_arcsec_central), color="k", alpha=0.1)
        ax.plot(xr, inv(xr, -1 * m /pixel_size_y_arcsec / delta_v**2, pix_v_source, pix_arcsec_central), color="k", alpha=0.1)
        c = 1E10
    left = np.array([xl,inv(xl, m_true /pixel_size_y_arcsec / delta_v**2, pix_v_source, pix_arcsec_central)]).T
    idx = closest_approach(np.array([pix_v_source, pix_arcsec_central]), left)
    closest = left[idx]
    print(f'Close pix: {closest}\nClose km/s,\'\': {pix_2_vel(closest[0])},{pix_2_arc(closest[1])}')
    ep = error_prop(pix_2_arc(closest[1]),pix_2_vel(closest[0]),m_true)
    print(f'Observational Error M: {ep/msun} Solm')

    cfig = corner.corner(samples, labels=["$m$", "$\ln\,f$"],
                          truths=[m_true, f_true])
    cfig.savefig("triangle.png",dpi=600)


    samples[:, 1] = np.exp(samples[:, 1])
    m_mcmc, f_mcmc = map(lambda v: (v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    print(f'M:{m_true}..{m_mcmc}\nE:{f_true}..{f_mcmc}')
    ax.plot(xl, inv(xl, m_true /pixel_size_y_arcsec / delta_v**2, pix_v_source, pix_arcsec_central), color="C", alpha=1,label=f'Mass: {m_true:.2f} M$_\odot$')
    ax.plot(xr, inv(xr, -1.* m_true /pixel_size_y_arcsec / delta_v**2, pix_v_source, pix_arcsec_central), color="C", alpha=1)
    mass = m_true

if plot_orbital:
    #embed()
    print('Plotting Orbital')
    p = cs.collections[0].get_paths()[1]
    v = p.vertices
    xCen = v[:,0]
    yCen = v[:,1]
    leftX = np.argmin(xCen)#[0]
    rightX = np.argmax(xCen)#[0]
    botY = np.argmin(yCen)#[0]
    topY = np.argmax(yCen)#[0]
    points = np.array([[xCen[leftX],yCen[leftX]],[xCen[rightX],yCen[rightX]],\
              [xCen[botY],yCen[botY]],[xCen[topY],yCen[topY]]])
    points = points.reshape(-1,2)
    print('Points:',leftX,rightX,botY,topY)
    print(PV_Data.shape)
    print(points)

    points[0,:] = [66, 1410]   
    #points[1,:] = [108.0384714, 1383.]
    #points[2,:] = [94., 1174.01457681]  
    #points[3,:] = [84., 1588.57664791]  

    print([points[0,0],points[0,1]],[points[1,0],points[1,1]])

    def findSlope(x1,y1,x2,y2):
        return (y1-y2)/(x1-x2),y2-((y1-y2)/(x1-x2)*x2)

    def findInter(m1,b1,m2,b2):
        return (b2-b1)/(m1-m2),m2*(b2-b1)/(m1-m2) + b2


    slope1 = findSlope(points[0,0],points[0,1],points[1,0],points[1,1])
    slope2 = findSlope(points[2,0],points[2,1],points[3,0],points[3,1])
    print('Slopes:',slope1,slope2)

    newCenter = findInter(*slope1,*slope2)
    print('Center:',newCenter)

    newCenterX = round(velocities[int(newCenter[0])]+v_source,2)
    newCenterY = round(y_arcsec[int(newCenter[1])],3)

    ax.scatter(points[:,0],points[:,1],color='C',marker='o',\
        label=f'Center:({newCenterX} km '+r's$^{-1}$, '+\
              f'{newCenterY}' + r'$^{\prime\prime}$)')
    ax.plot((points[0,0],points[1,0]),(points[0,1],points[1,1]),'--c',lw=1)
    ax.plot((points[2,0],points[3,0]),(points[2,1],points[3,1]),'--c',lw=1)

print('Finished Plotting')

fig.legend(loc=1,fontsize='x-small')
fig.tight_layout()
fig.savefig("PV-Diagram_L1448IRS3B_C17O_image_taper1500k.fit.png",dpi=600)
