# Misc Functions
import numpy as np
from math import cos,sin,acos,ceil

def linear(x,a,b):
    return a*x + b

def binning(data,width=3):
    return data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return [idx,array[idx]]

# function for formatting
def addspace(sstring,spacing=20):
    while True:
        if len(sstring) >= spacing:
            sstring = sstring[:-1]
        elif len(sstring) < (spacing -1):
            sstring = sstring + ' '
        else: 
            break
    return sstring + ' '

def list_files(dir):
    '''
    List all the files within a directory
    '''
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        if (len(files) > 0):
            for file in files:
                r.append(subdir + "/" + file)
    return r 

def ang_vec(deg):
    rad = deg*pi/180.
    return (cos(rad),sin(rad))

def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]

def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    while np.abs(cosx) >= 1:
        cosx = cosx/(np.abs(cosx)*1.001)
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees

def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det>0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def gen_angles(start,end,resolution=1,direction='+'):
    if direction == "+":# positive direction
        diff = round(angle_clockwise(ang_vec(start),ang_vec(end)),2)
        numd = int(ceil(diff/resolution))+1
        final = [round((start + x*resolution),2)%360 for x in range(numd)]
    elif direction == "-": # negative direction
        diff = round(360.-angle_clockwise(ang_vec(start),ang_vec(end)),2)
        numd = int(ceil(diff/resolution))+1
        final = [round((start - x*resolution),2)%360  for x in range(numd)]
    return final

def gauss(x,mu,sigma,A):
    '''
    define a single gaussian
    '''
    return A*np.exp(-(x-mu)**2/2./sigma**2)

def ndgauss(x,params):
    '''
    n dimensional gaussian
    assumes params is a 2d list
    ''' 
    for i,dim in enumerate(params):
        if i == 0:
            final = gauss(x,*dim)
        else:
            final = np.sum(gauss(x,*dim),final,axis=0)

    return final

def addconst(func,C):
    return func + C

def polynomial(x,params):
    '''
    polynomial function
    assuming params is a 1d list
    constant + 1st order + 2nd order + ...
    '''
    for i,dim in enumerate(params):
        if i == 0:
            final = [dim for y in x]
        else:
            final = np.sum(dim*x**i,final,axis=0)

    return final

# spectra fitting

def baseline(x,y,order=2):
    '''
    Input the xvals and yvals for the baseline
    Will return the function that describes the fit
    '''
    fit = np.polyfit(x,y,order)
    fit_fn = np.poly1d(fit)
    return fit_fn

def listinvert(total,msk_array):
    '''
    msk_array must be the index values
    '''
    mask_inv = []
    for i in range(len(msk_array)):
        mask_inv = np.append(mask_inv,np.where(total == msk_array[i]))
    mask_tot = np.linspace(0,len(total)-1,num=len(total))
    mask = np.delete(mask_tot,mask_inv)
    mask = [int(x) for x in mask]
    return mask

def equivalent_width(spectra,blf,xspec0,xspec1,fit='gauss',params=[1,1,1]):
    '''
    finds equivalent width of line
    spectra is the full 2d array (lam,flux)
    blf is the baseline function
    xspec0 (xspec1) is the start(end) of the spectral feature


    # PROBABLY EASIER TO JUST ASSUME GAUSSIAN OR SIMPLE SUM


    def gaussc(x,A,mu,sig,C):
        return A*np.exp(-(x-mu)**2/2./sig**2) + C

    from scipy.optimize import curve_fit

    featx,featy = lam[featurei],flux[featurei]
    expected1=[1.3,2.165,0.1,1.]
    params1,cov1=curve_fit(gaussc,featx,featy,expected1)
    sigma=params1[-2]

    # I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)
    # sigma = dw / 2 / sqrt (2 * ln (core/Iw))
    # fwhm = 2.355 * sigma = 2(2ln2)^0.5 * sigma
    # flux = core * sigma * sqrt (2*pi)
    # eq. width = abs (flux) / cont

    fwhm = 2. * (2. * np.log(2.))**0.5 * sigma
    core,center,nsig,cont = params1
    flx = core * sigma * np.sqrt (2.*np.pi)
    eqwidth = abs(flx) / cont
    print(eqwidth)
    print(fwhm)
    '''



    specfeatxi,specfeatxv = np.array(between(spectra,xspec0,xspec1))[:,0],np.array(between(spectra,xspec0,xspec1))[:,1]

    if fit == 'gauss':
        _params2,_cov2=curve_fit(gauss,specfeatxv,spectra[specfeatxi,1],*params)
        _sigma2=np.sqrt(np.diag(_cov2))
        function = gauss(specfeatxv,*_expected2)
        return 
    elif fit == 'ndgauss':
        pass


def between(l1,val1,val2):
    '''
    returns values and index
    '''
    if val1 > val2:
        low = val2
        high = val1
    elif val1 < val2:
        low = val1
        high = val2
    else:
        print('Values are the same')
        return []

    l2 = []
    for j,i in enumerate(l1):
        if(i > low) and (i < high):
            l2.append([j,i])
    return l2


# Utility functions to find minimum 
# and maximum of two elements
def minv(x, y):
    return x if(x < y) else y
     
def maxv(x, y):
    return x if(x > y) else y
 
# Returns length of the longest
# contiguous subarray. Assuming sorted
def findLength(arr, n=0):
    # n is the first n elements of the array to look in
    if n == 0:
        n = len(arr)
     
    # Initialize result
    max_len = 1
    i = 0
    vals = [0,0]
    while i < len(range(n - 1)):
        #print('i:',i)
     
        # Initialize min and max for
        # all subarrays starting with i
        mn = arr[i]
        mx = arr[i]

        # Consider all subarrays starting
        # with i and ending with j
        j = i+1
        while j < len(range(n - 1)):
            #print('j:',j)
            # Update min and max in
            # this subarray if needed
            mn = minv(mn, arr[j])
            mx = maxv(mx, arr[j])
 
            # If current subarray has
            # all contiguous elements
            if ((mx - mn) == (j - i)):
                #print('New Length')
                if max_len < (mx - mn + 1):
                    vals = [mn,mx]
                max_len = maxv(max_len, mx - mn + 1)
            else:
                #print('Keeping ',vals)
                i = j-1
                j += 1
                break
            '''
            print('i:{},j:{},mn:{},mx:{},max_len:{}'\
                  .format(i,j,mn,mx,max_len))
            '''
            j += 1
        i += 1
    # returns length, [min,max],[indexes]
    return max_len,vals,[i for i,x in enumerate(arr) if (vals[0] <= x <= vals[-1]) ]
