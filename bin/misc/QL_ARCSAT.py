"""
Written by Evan Rich, 8/25/2015
This code takes the VHS1256 L and M band images and renders them into a usuable form and plotted.
"""

#---------------------------------------
# Imported libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
#from astropy.wcs import WCS
from matplotlib.font_manager import FontProperties
#from scipy.ndimage import filters 
#from matplotlib.patches import Ellipse

#-------------------------------------
# Make edited flat field
def flats(type, bias_image):
    list = []
    for name in glob.glob('skyflat_' + type + '*.fits'):
        list.append(name)
    fdata = np.zeros((1024,1024,len(list)))
    for i in range(0,len(list)):
        HDU = fits.open(list[i])
        fdata[:,:,i] = HDU[0].data - bias_image
    fcomb = np.median(fdata, axis=2)
    fcomb = fcomb/np.average(fdata)
    hdu = fits.PrimaryHDU(fcomb)
    hdu.writeto('Flat_' + type + '.fits', clobber = True)

def bias():
    list = []
    for name in glob.glob('Bias*.fits'):
        list.append(name)
    bdata = np.zeros((1024,1024,len(list)))
    for i in range(0,len(list)):
        HDU = fits.open(list[i])
        bdata[:,:,i] = HDU[0].data
    print 'making bias image'
    bcomb = np.median(bdata, axis=2)
    hdu = fits.PrimaryHDU(bcomb)
    hdu.writeto('Bias.fits', clobber = True)

# ---------------------------------
#Scale and plot image

def plot_image(image,title):
    print title
    maxi = np.amax(image)
    med = np.median(image)
    fig = plt.figure(figsize=(10,10))
    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
#    image = filters.gaussian_filter(Q_image[149:549,149:549],1)
    img1 = ax.imshow(np.log10(np.abs(image)), origin='lower', vmin = np.log10(np.abs(med)), vmax = np.log10(np.abs(maxi)))
    fig.colorbar(img1, ax=ax)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.savefig(title + 'log10.png', dpi = 600)
    plt.close()
    fig = plt.figure(figsize=(10,10))
    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
#    image = filters.gaussian_filter(Q_image[149:549,149:549],1)
    img1 = ax.imshow(np.abs(image), origin='lower', vmin = np.abs(med), vmax = np.abs(maxi))
    fig.colorbar(img1, ax=ax)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.savefig(title + '.png', dpi = 600)
    plt.close()

#----------------------------
# Main Program

def main():
    while True:
        try:
            HDU = fits.open('Bias.fits')
            bias_image = HDU[0].data
            print 'Bias frame loaded'
            break
        except IOError:
            bias()
            print 'Bias frame created'
    while True:
        try:
            HDU = fits.open('Flat_V.fits')
            flat_V = HDU[0].data
            print 'Flat frame loaded'
            break
        except IOError:
            flats('V', bias_image)
            HDU = fits.open('Flat_V.fits')
            flat_V = HDU[0].data
    while True:
        try:
            HDU = fits.open('Flat_B.fits')
            flat_B = HDU[0].data
            print 'B Flat frame loaded'
            break
        except IOError:
            flats('B', bias_image)
            HDU = fits.open('Flat_B.fits')
            flat_B = HDU[0].data
    while True:
        try:
            HDU = fits.open('Flat_R.fits')
            flat_R = HDU[0].data
            print 'Flat frame loaded'
            break
        except IOError:
            flats('R', bias_image)
            HDU = fits.open('Flat_R.fits')
            flat_R = HDU[0].data
    
    filelist = []
    for name in glob.glob('*.fits'):
        if name.find('Bias') == -1 and name.find('flat') == -1 and name.find('bf_') == -1 and name.find('Flat') == -1:
            print name
            filelist.append(name)
    for i in range(0,len(filelist)):
        HDU = fits.open(filelist[i])
        image = HDU[0].data
        image = image - bias_image
        if filelist[i].find('_V_') != -1:
            print 'V flat applied'
            image = image/flat_V
        elif filelist[i].find('_B_') != -1:
            image = image/flat_B
        elif filelist[i].find('_R_') != -1:
            image = image/flat_R
        title = filelist[i]
        plot_image(image,title)
        hdu = fits.PrimaryHDU(image)
        hdu.writeto('bf_' + filelist[i], clobber = True)
    

if __name__ == '__main__':
    main()
