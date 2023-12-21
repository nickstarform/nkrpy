import numpy as np
import warnings
from nkrpy import math
import itertools
from nkrpy.astro import WCS
from nkrpy.io import fits
from skimage.transform import resize
from matplotlib.cm import ScalarMappable
from statsmodels.distributions.empirical_distribution import ECDF
from nkrpy import functions
flatten = functions.flatten
from scipy.stats import ks_2samp, anderson_ksamp, epps_singleton_2samp, mannwhitneyu, cramervonmises_2samp
from copy import deepcopy
warnings.filterwarnings('ignore')

def rms(data):
    ret = np.sqrt(np.sum((data - data.mean())**2)/data.shape[0])
    return ret


def sort_corr_mixture_ratio(mxstr_dict):
    # return list of keys sorted
    keys = list(mxstr_dict.keys())
    keys = [k.split('_') for k in keys]
    keys.sort(key=lambda x: float(x[0]))
    return ['_'.join(k) for k in keys][::-1]


# generic functions
def calcinc(s, sampling=False): # find inclination assuming minor == major symmetry
    if 'i' in s and not sampling:
        return s['i']
    mmi = s['minor']
    mma = s['major']
    mmie = s['minor_error']
    mmae = s['major_error']
    if sampling and mmie > 0 and mmae > 0:
        mmi = np.random.normal(loc=mmi, scale=mmie, size=1)[0]
        mma = np.random.normal(loc=mma, scale=mmae, size=1)[0]
        #    mmi = np.random.uniform(low=mmi-mmie, high=mmi + mmie, size=1)[0]
        #    mma = np.random.uniform(low=mma - mmae, high=mma + mmae, size=1)[0]
    if mmi > mma:
        mmi, mma = sorted([mmi, mma])
    return np.arccos(mmi / mma) * 180. / np.pi

def diffinc(s1, s2): # find difference in inclinations
    return calcinc(s1) - calcinc(s2)

def sph_2_xyz(source, sampling=False):
    r = 1
    pa, pae, inc = source['pa'], source['pa_error'], calcinc(source, sampling=sampling)
    if sampling and pae > 0:
        pa = np.random.normal(loc=pa, scale=pae, size=1)[0]
        #    pa = np.random.uniform(low=pa - pae, high=pa + pae, size=1)[0]
    inc_rad = inc * np.pi / 180.
    pa_rad = pa * np.pi / 180.
    # assume pa is east of north
    pa_rad += np.pi / 2
    x = r * np.sin(inc_rad) * np.cos(pa_rad)
    y = r * np.sin(inc_rad) * np.sin(pa_rad)
    z = r * np.cos(inc_rad)
    return x,y,z

def dotprod(s1, s2):
    # find dotproduct of two sources
    p1 = s1['pa']
    p2 = s2['pa']
    i1 = calcinc(s1)
    i2 = calcinc(s2)
    # spherical coord method of dotprod
    return __dotprod(i1=i1,i2=i2,p1=p1,p2=p2)
    #return np.dot(sph_2_xyz(s1, sampling=sampling),sph_2_xyz(s2, sampling=sampling))

def choose(n, k):
    fac = np.math.factorial
    return fac(n) / fac(n - k) / fac(k)

def calcdist(s1, s2):
    ddec_mean = np.mean([s1['dec'], s2['dec']])
    dra = (s1['ra'] - s2['ra']) * np.cos(ddec_mean * np.pi / 180)
    ddec = s1['dec'] - s2['dec']
    ret = (dra ** 2 + ddec ** 2) ** 0.5
    return ret

def calcmid(s1, s2):
    dra = s1['ra'] + s2['ra']
    ddec = s1['dec'] + s2['dec']
    return {'ra': dra / 2, 'dec':ddec / 2}

def arcsec(arc):
    ss = f'{arc:0.2f}'
    ss = ss.split('.')
    return r"\H{.}".join(ss)

def pop_name(src):
    for sn, s in src.items():
        if 'name' not in s:
            s['name'] = sn





def find_min_dist(s, group_comb):
    dist = [None, np.inf]
    for sn1, sn2 in group_comb:
        s1,s2 = s[sn1], s[sn2]
        distpair = calcdist(s1, s2)
        if distpair < dist[-1]:
            dist = [f'{sn1}:{sn2}', distpair]
    return dist


def sort_dist(s, group_comb):
    dist = []
    for sn1, sn2 in group_comb:
        pair = f'{sn1}:{sn2}'
        pair_r = f'{sn2}:{sn1}'
        distpair = calcdist(s1, s2)
        dist.append([sn1, sn2, distpair])
    dist.sort(key=lambda x: x[-1])
    return dist



def pop_distance2(s, association_max_distance_au=1000, association_min_distance_au=0, distance_pc=300):
    # this routine is a little slower, but tries to find the ''geometric center of mass'' for the systems and find all sources within that region.
    s = deepcopy(s)
    dist_dict = {}
    skeys = list(s.keys())
    while len(skeys) > 0:
        group_comb = list(itertools.combinations(skeys, 2))
        mindist = find_min_dist(s, group_comb)
        sn1sn2, distpair = mindist
        if sn1sn2 is None:
            del skeys[0]
            continue
        sn1, sn2 = sn1sn2.split(':')
        s1, s2 = s[sn1], s[sn2]
        if (distpair * 3600 * distance_pc < association_max_distance_au) and (distpair * 3600 * distance_pc > association_min_distance_au):
            dist_dict[f'{sn1}:{sn2}'] = distpair
            s[f'{sn1}___{sn2}'] = {'peak': 0, 'I'
                :0, 'ra': 0, 'dec': 0, 'major':0, 'minor':0, 'pa':0, 'name': f'{s1}___{s2}', **calcmid(s1, s2)}
            del s[sn1]
            del s[sn2]
        else:
            break
        skeys = list(s.keys())
    return dist_dict


def pop_distance(s, association_max_distance_au=1000, association_min_distance_au=0, distance_pc=300):
    # faster routine, this one just calculates the distance from the first source picked. Good for nearly all circumstances unless crowded field
    dist = {}
    group_comb = itertools.combinations(s, 2)
    for sn1, sn2 in group_comb:
        pair = f'{sn1}:{sn2}'
        pair_r = f'{sn2}:{sn1}'
        if pair in dist or pair_r in dist:
            # no duplicates
            continue
        s1,s2 = s[sn1], s[sn2]
        distpair = calcdist(s1, s2)
        if (distpair * 3600 * distance_pc < association_max_distance_au) and (distpair * 3600 * distance_pc > association_min_distance_au):
            # if it is within valid association distance
            dist[pair] = distpair
    return dist


def pop_groupings2(gr, dist, src):
    # to be used with pop_distance2
    temp = {} # will be an array of child: parent
    for dns in dist.keys():
        sn1, sn2 = dns.split(':')
        if sn1 in temp:
            for k in sn2.split('___'):
                temp[k] = temp[sn1]
            continue
        elif sn2 in temp:
            for k in sn1.split('___'):
                temp[k] = temp[sn2]
            continue
        sns = sn1.split('___') + sn2.split('___')
        sns = set(sns)
        if len(sns - temp.keys()) < len(sns):
            # found one of the current group in child of another
            child = sns - (sns - temp.keys())
            for s in sns:
                temp[s] = temp[list(child)[0]]
            continue
        parent = list(sns)[0]
        for s in sns:
            temp[s] = parent
    nt = {}
    for child, parent in temp.items():
        if parent not in nt:
            nt[parent] = [parent, child]
            continue
        nt[parent].append(child)
    for _, childs in nt.items():
        childs = sorted(set(childs))
        gr[childs[0]] = childs[1:]

def pop_groupings(gr, dist):
    # to be used with pop_distance
    sources_grouped = []
    #group_dist = {}
    for dpn in dist.keys():
        sn1,sn2 = dpn.split(':')
        if sn1 in gr:
            sn_parent = sn1
            sn_child = sn2
        elif sn2 in gr:
            sn_parent = sn2
            sn_child = sn1
        else:
            sn_parent = sn1
            sn_child = sn2
        if sn_parent not in gr and sn_parent not in [v1 for _,v in gr.items() for v1 in v]:
            gr[sn_parent] = []
            #group_dist[sn_parent] = []
        found = False
        for gparent, v in gr.items():
            if found:
                break
            for child in v:
                if found:
                    break
                if sn_parent == child:
                    sn_parent = gparent
                    found = True
        if sn_child in sources_grouped:
            continue
        gr[sn_parent].append(sn_child)
        sources_grouped.extend([sn_parent, sn_child])


def sample(mu, sig, numsamples=10000, lowerclip = -np.inf, upperclip=np.inf, lowerclip_val=np.nan, upperclip_val=np.nan):
    samp = np.random.normal(loc=mu, scale=sig, size=numsamples)
    samp[samp < lowerclip] = lowerclip_val
    samp[samp > upperclip] = upperclip_val
    return samp



def pop_sampling(gr, src, NUMBER_SAMPLES=10000, **kwargs):
    ret = []
    ret_error = []
    for sn, s in src.items():
        # sample ra, dec, pa, major, minor
        for mu in ['ra', 'dec', 'pa', 'major', 'minor']:
            if mu in ['major', 'minor']:
                # resample if major or minor < 0
                s[mu+'_sample'] = np.zeros(NUMBER_SAMPLES, dtype=float)
                mask = np.ones(NUMBER_SAMPLES, dtype=int)
                while mask.sum():
                    s[mu+'_sample'][mask] = sample(mu=s[mu],sig=s[mu+'_error'], numsamples=mask.sum())
                    mask = s[mu+'_sample'] <= 0
            else:
                s[mu+'_sample'] = sample(mu=s[mu],sig=s[mu+'_error'], numsamples=NUMBER_SAMPLES)
        incs = np.zeros(NUMBER_SAMPLES, dtype=float)
        mask = s['minor_sample'] > s['major_sample']
        s['minor_sample'][mask],s['major_sample'][mask] = s['major_sample'][mask],s['minor_sample'][mask]
        incs = np.arccos(s['minor_sample'] / s['major_sample']) * 180 / np.pi
        s['inc_sample'] = incs % 90
        s['pa'] = s['pa'] % 180

    # now look through all pairs and take dotprod
    src_order = []
    for gn, g in gr.items():
        # gather names
        group = sorted([gn, *g])
        group_comb = itertools.combinations(group, 2)
        for sn1, sn2 in group_comb:
            src_order.append(f'{sn1}---{sn2}')
            s1,s2 = src[sn1],src[sn2]
            ret.append(dotprod_sample(s1, s2))
            ret_error.append(dotprod_error(s1,s2))
    ret_error = np.array(ret_error).T
    ret = np.array(ret).T # shape of [# dotprods, # samples]
    # now make a dict for every sample for faster referencing
    return src_order, ret_error, ret

def inc_error(s1):
    i1 = s1['inc_sample'] % 90
    i1_rad = s1['inc_sample'] * np.pi / 180.
    sini1 = np.sin(i1_rad)
    rat = s1['minor_sample'] / s1['major_sample']
    di_dmin = -1 / (s1['major_sample'] * sini1)
    di_dmaj = 1 / s1['minor_sample'] * rat ** 2 / sini1
    dminmaj = np.mean([s1['minor_error'], s1['major_error']])
    di2 = (di_dmin * dminmaj) ** 2 + (di_dmaj * dminmaj) ** 2
    return di2 ** 0.5


def __dotprod(i1=0, i2=0, p1=0, p2=0):
    p1_rad, p2_rad, i1_rad, i2_rad = map(lambda x: x * np.pi / 180, [p1, p2, i1, i2])
    dp =  np.sin(i1_rad)*np.sin(i2_rad)*np.cos(p1_rad-p2_rad) + np.cos(i1_rad)*np.cos(i2_rad)
    return np.abs(dp)

def dotprod_sample(s1, s2):
    # find dotproduct of two sources
    # convert each source from spherical to cartesian
    # now vector add
    p1 = s1['pa_sample'] % 180
    p2 = s2['pa_sample'] % 180
    i1 = s1['inc_sample'] % 90
    i2 = s2['inc_sample'] % 90
    return __dotprod(i1=i1,i2=i2, p1=p1, p2=p2)


def dotprod_error(s1, s2):
    p1 = s1['pa_sample'] % 180
    p2 = s2['pa_sample'] % 180
    i1 = s1['inc_sample'] % 90
    i2 = s2['inc_sample'] % 90
    di1 = inc_error(s2)
    di2 = inc_error(s2)
    dp1 = s1['pa_error']
    dp2 = s2['pa_error']
    return __dotprod_error(i1, i2, p1, p2, di1, di2, dp1, dp2)



def __dotprod_error(*args):
    # i1,i2, p1, p2, di1,di2,dp1, dp2
    i1, i2, p1, p2, di1, di2, dp1, dp2 = map(lambda x: x * np.pi / 180, args)
    cosi1 = np.cos(i1)
    sini1 = np.sin(i1)
    cosi2 = np.cos(i2)
    sini2 = np.sin(i2)
    cosp1p2 = np.cos(p1-p2)
    sinp1p2 = np.sin(p1-p2)

    dD_di1 = cosi1*sini2*cosp1p2 - sini1*cosi2
    dD_di2 = sini1*cosi2*cosp1p2 - sini2*cosi1
    dD_dp1 = -sini1*sini2*sinp1p2 + cosi1*cosi2
    dD_dp2 = sini1*sini2*sinp1p2 + cosi1*cosi2

    dD2 = (dD_di1*di1)**2 + (dD_di2*di2)**2 + (dD_dp1*dp1)**2 + (dD_dp2*dp2)**2
    return dD2 ** 0.5


def sample_model_keys(correlation_mixture_ratios, mxstr=None, maxnsamples=100):
    # choose maxnsample random systems of models. 
    if mxstr is None or mxstr not in correlation_mixture_ratios:
        mxstr = np.random.choice(list(correlation_mixture_ratios.keys()), size=1)
    if maxnsamples == len(correlation_mixture_ratios[mxstr]['models']['correlated']):
        return list(correlation_mixture_ratios[mxstr]['models']['correlated'])
    random_groups = np.random.choice(list(correlation_mixture_ratios[mxstr]['models']['correlated']), size=maxnsamples)
    
    return random_groups

def model_dotprod(random_group_nums, src, mdls):
    ret = []
    for system in random_group_nums:
        # gather all sources
        group = list(mdls['correlated'][system]) + list(mdls['uncorrelated'][system])
        group_comb = itertools.combinations(group, 2)
        for sn1, sn2 in group_comb:
            ret.append(dotprod(src[sn1], src[sn2]))
    return np.array(ret)

def binmid(bins):
    diff = np.diff(bins)
    bins = bins[:-1] + diff / 2
    return bins

def ecdf(a):
    modelnums = np.sort(a)
    return ecdf4(modelnums)


def ecdf1(a, bins='auto'):
    # try way 1
    # wrong
    modelpdf, modelbins_edges = np.histogram(a, bins=bins, density=True)
    modelcdf = np.cumsum(modelpdf * np.diff(modelbins_edges))
    return binmid(modelbins_edges), modelcdf

def ecdf2(a, bins='auto'):
    # trying a new function
    # wrong
    modelpdf, modelbins_edges = np.histogram(a, bins=bins, density=True)
    return binmid(modelbins_edges), ECDF(a)(binmid(modelbins_edges))

def ecdf3(a, bins='auto'):
    # okay
    x, counts = np.unique(a, return_counts=True)
    cumsum = np.cumsum(counts)
    y = cumsum / cumsum[-1]
    return x, y

def ecdf4(a, bins='auto'):
    # okay
    y = np.arange(1, a.shape[0] + 1) / a.shape[0]
    return a, y
        


def norm_cuc_to1(mxstr):
    # mxstr is X_Y or cX_ucY
    start = None
    if 'c' in mxstr.lower():
        start = 'c'
    c, uc = mxstr.lower().replace('uc', '').replace('c', '').split('_')
    c, uc = map(float, [c, uc])
    totes = c + uc
    c *= 1 / totes
    uc *= 1 / totes
    if start is None:
        return f'{c:0.2f}_{uc:0.2f}'
    return f'C{c:0.2f}_UC{uc:0.2f}'



def determine_beam_offsets(datashape, bmaj_pix, bmin_pix, bpa, delt=None):
    if delt is not None:
        delt = abs(delt)
        bmaj_pix, bmin_pix = list(map(lambda x: x / delt, [bmaj_pix, bmin_pix]))
    phi_x=(90 + bpa) * np.pi / 180
    off_x = (bmaj_pix ** 2 * np.cos(phi_x)** 2 + bmin_pix ** 2 * np.sin(phi_x)**2)**0.5
    phi_y = bpa * np.pi / 180
    off_y = (bmaj_pix ** 2 * np.cos(phi_y)** 2 + bmin_pix ** 2 * np.sin(phi_y)**2)**0.5
    if max([off_x, off_y]) < max(datashape) / 10:
        off_y, off_x = list(map(lambda x: x / 10, datashape))
    return datashape[1] - off_x, off_y











def make_latex_row(*list_of_cols, number_of_cols=None, centering=None, units=False, header=False, newline=True):
    list_of_cols = flatten(list_of_cols)
    if newline:
        newline = r'\\'
    else:
        newline=''
    if units:
        l, r = '(', ')'
    else:
        l,r = '',''
    ret = []
    if isinstance(centering, str):
        centering = [c for c in centering]
    for i, col in enumerate(list_of_cols):
        numcol = 1
        center = 'c'
        if centering is not None:
            center = centering[i]
        if number_of_cols is not None:
            numcol = number_of_cols[i]
        if numcol == 1:
            if header:
                ret.append(r'\colhead{' + f'{r}{col}{r}' + r'}')
            else:
                ret.append(f'{r}{col}{r}')
        else:
            ret.append(r'\multicolumn{' + f'{numcol}' +r'}{'+f'{center}' + r'}{' +f'{l}{col}{r}' + r'}')
    return ' & '.join(ret) +  newline

def collect_processes(stats, newresult):
    for function in newresult.keys():
        if function not in stats:
            stats[function] = newresult[function]
            continue
        for fitstyle in newresult[function].keys():
            if fitstyle not in stats[function]:
                stats[function][fitstyle] = newresult[function][fitstyle]
                continue
            for mxstr in newresult[function][fitstyle].keys():
                if mxstr not in stats[function][fitstyle]:
                    stats[function][fitstyle][mxstr] = newresult[function][fitstyle][mxstr]

def processrunner(fitstyle, mxstr, datasuites, modeldata):
    stats = {}
    functions = ['kss', 'ads', 'eps', 'cvm', 'mwu']
    for f in  functions:
        if f not in stats:
            stats[f] = {}
        if fitstyle not in stats[f]: 
            stats[f][fitstyle] = {}
        if mxstr not in stats[f][fitstyle]:
            stats[f][fitstyle][mxstr] = []
            stats[f][fitstyle][mxstr+'-prob'] = []
    for i in range(datasuites.shape[0]):
        threadrunner(stats, fitstyle, mxstr, datasuites[i, :], modeldata)
    return stats


# setup multithreading
def threadrunner(stats, fitstyle, mxstr, datasample, modelsample):
    datasample = np.sort(datasample)
    modelsample = np.sort(modelsample)
    ks = ks_2samp(data1=datasample, data2=modelsample, alternative='two-sided', mode='asymp')
    stats['kss'][fitstyle][mxstr].append(ks.statistic)
    stats['kss'][fitstyle][mxstr+'-prob'].append(ks.pvalue)
    # AD
    ad = anderson_ksamp(samples=[datasample, modelsample], midrank=False)
    stats['ads'][fitstyle][mxstr].append(ad.statistic)
    stats['ads'][fitstyle][mxstr+'-prob'].append(ad.significance_level)
    ep = epps_singleton_2samp(x=datasample, y=modelsample, t=(0.4, 0.8))
    stats['eps'][fitstyle][mxstr].append(ep.statistic)
    stats['eps'][fitstyle][mxstr+'-prob'].append(ep.pvalue)
    # mann-whitney-u
    mwu = mannwhitneyu(x=datasample, y=modelsample, alternative='two-sided', method='asymptotic')
    stats['mwu'][fitstyle][mxstr].append(mwu.statistic)
    stats['mwu'][fitstyle][mxstr+'-prob'].append(mwu.pvalue)
    # cramer vonmises
    cvm = cramervonmises_2samp(x=datasample, y=modelsample, method='asymptotic')
    stats['cvm'][fitstyle][mxstr].append(cvm.statistic)
    stats['cvm'][fitstyle][mxstr+'-prob'].append(cvm.pvalue)
    ln = len(stats['kss'][fitstyle][mxstr])
    if ln % 100 == 0:
        print(f'Finished {ln:,} of {fitstyle} {mxstr}', flush=True)



def convstr_uv(s):
    return s.lower().replace('uv fitting', r'\textit{uv}-plane').replace('imfits', r'\textit{imfit}')








def t(mstar, mplanet, sma):
    u = 10 * (mplanet/mstar) / sma # in inverse au
    return mstar ** -0.5 * sma ** -0.5 * (1e4 / u) ** 2 /1e8# 



srcs = [[1.25e-3, 44],
        [4.2e-3, 90],
        [7e-2, 496]]

for src in srcs:
    print(t(1, *src))






# deprojecting the whole visibilities
# useful for deprojecting disks, etc
def deproj(x, y, xmu, ymu, pa, inc):
    """
    Rotate and deproject individual visibility coordinates.
    From Hughes et al. (2007) - "AN INNER HOLE IN THE DISK AROUND 
    TW HYDRAE RESOLVED IN 7 mm DUST EMISSION".
    """
    pa_rad = (pa%180) * np.pi / 180
    inc_rad = (inc%90) * np.pi / 180.
    R = ((x - xmu) ** 2 + (y - ymu) ** 2) ** 0.5
    phi_rad = np.arctan2(x-xmu, y-ymu) - pa_rad
    newx = R * np.cos(phi_rad) * np.cos(inc_rad)
    newy = R * np.sin(phi_rad)
    return newx, newy

def dist(args):
    dra = args[0] * np.cos(args[1] * np.pi / 180)
    ddec = args[1]
    return (dra ** 2 + ddec ** 2) ** 0.5 * 3600 * 300












def findedge(newwcs, xc, yc, ta, inc, major, pa, xoff, yoff):
    # find the starting position on the disk given the center, PA, major minor of the disk and the true anomoly
    # find the ending position given the center and the offsets
    ra_s = xc - major / 2 * np.cos((ta+pa) * np.pi / 180) / 3600
    dec_s = yc + major * np.cos(inc * np.pi / 180) / 2 * np.sin((ta+pa) * np.pi / 180) / 3600
    ra_f = xc + xoff / 3600
    dec_f = yc + yoff / 3600
    ra_s_pix = newwcs(ra_s, 'pix', 'ra---sin')
    ra_f_pix = newwcs(ra_f, 'pix', 'ra---sin')
    dec_s_pix = newwcs(dec_s, 'pix', 'dec--sin')
    dec_f_pix = newwcs(dec_f, 'pix', 'dec--sin')
    return ra_s_pix, dec_s_pix, ra_f_pix, dec_f_pix


def rotation_matrix(x,y, angle):
    # rotate xy by an angle
    cosa = np.cos(angle *np.pi / 180)
    sina = np.sin(angle * np.pi / 180)
    return x*cosa + y * sina, y*cosa-x*sina


def draw_arm(ax, xs_pix, ys_pix, xf_pix, yf_pix, power=2, angle=45, side=1, linestyle='--', color='cyan'):
    xoff = xs_pix
    yoff = ys_pix
    xs_pix, xf_pix = xs_pix-xoff, xf_pix-xoff
    ys_pix, yf_pix = ys_pix-yoff, yf_pix-yoff
    xf_pix, yf_pix = rotation_matrix(xf_pix, yf_pix, angle=-angle)
    x_pix = np.linspace(0, xf_pix, 10)
    m = (yf_pix) / (xf_pix ** power)
    y_pix = side*m*(x_pix) ** power
    x_pix, y_pix = rotation_matrix(x_pix, y_pix, angle=angle)
    ax.plot(x_pix+xoff, y_pix + yoff, linestyle=linestyle, color=color)



def find_vminvmax(cfgs, names):
    vmin = np.inf
    vmax = -np.inf
    for i, source in enumerate(names):
        source = source.replace('+','')
        f = cfgs[source]['imagename']
        _, d = fits.read(f)
        d[np.isnan(d)] = np.nanstd(d)
        std = np.std(d[d < np.percentile(d, 95)])
        peak = np.percentile(d, 99.999)
        if std < vmin:
            vmin = std
        if peak > vmax:
            vmax = peak
    return vmin, vmax


def prep_image(d, wcs, cfg, width=None, pixels=None):
    from scipy.ndimage import spline_filter
    dorig = np.squeeze(d.copy())
    ra, dec = cfg['icrs'].split(',')
    if width is None:
        width = cfg['width']
    if pixels is None:
        pixels = int(round(abs(width / 3600 / wcs.axis1['delt']), 0 ))
    ra = math.icrs2degrees(ra) * 15
    dec = math.icrs2degrees(dec)
    decpix = wcs(dec, 'pix', 'dec--sin')
    rapix = wcs(ra, 'pix', 'ra---sin', declination_degrees=dec)
    ra_width = abs(width / 3600 / wcs.axis1['delt'])
    dec_width = ra_width
    lower_ra = int(round(rapix - ra_width / 2., 0))
    upper_ra = int(round(rapix + ra_width / 2., 0))
    lower_dec = int(round(decpix - dec_width / 2., 0))
    upper_dec = int(round(decpix + dec_width / 2., 0))
    d = dorig[lower_dec:upper_dec+1, lower_ra:upper_ra+1]
    print(f'Image noise:{np.nanstd(dorig[dorig < np.percentile(d, 90)])}')
    d[np.isnan(d)] = np.nanstd(dorig)
    d_show = resize(d, (pixels, pixels))
    d_show = spline_filter(d_show, order=3)
    bma, bmi, bpa = wcs.get_beam()
    newwcs = WCS({
        'cdelt1': -width / 3600 / pixels,
        'cdelt2': width / 3600 / pixels,
        'naxis1': pixels,
        'naxis2': pixels,
        'cunit1': wcs.axis1['unit'],
        'cunit2': wcs.axis2['unit'],
        'ctype1': wcs.axis1['dtype'],
        'ctype2': wcs.axis2['dtype'],
        'crpix1': pixels/2,
        'crpix2': pixels/2,
        'crval1': ra,
        'crval2': dec,
        'bmaj':bma,
        'bmin':bmi,
        'bpa':bpa,

    })
    d_show += abs(np.nanmin(dorig))
    d_show += np.nanmin(np.abs(dorig))
    return newwcs, d_show


def propermotion(ra_shift, dec_shift, wcs):
    ra_shift, dec_shift = map(lambda x: int(round(x, 0)), [ra_shift, dec_shift])
    wcs.shift_axis(val=ra_shift, axis='ra---sin', unit = 'pix')
    wcs.shift_axis(val=-dec_shift, axis='dec--sin', unit = 'pix')
    wcs.refresh_axes()
    #data = np.roll(data, shift=(dec_shift, -ra_shift), axis=(0, 1))
    return wcs


def resolve_dir(pth):
    total = '/'
    if not os.path.exists(pth):
        for p in pth.split('/'):
            total += p + '/'
            if not os.path.exists(total):
                os.mkdir(total)

def valid(comp, wcs):
    if 'ra' not in comp:
        return False
    ra = comp['ra']
    dec = comp['dec']
    decpix = wcs(dec, return_type='pix', axis=wcs.axis2['dtype'])
    rapix = wcs(ra, return_type='pix', axis=wcs.axis1['dtype'], declination_degrees=dec)
    if rapix < 0 or decpix < 0:
        return False
    if rapix > wcs.axis1['axis'] or decpix > wcs.axis2['axis']:
        return False
    return True


def is_first_col(i, ncols):
    return (i % ncols) == 0

def is_last_col(i, ncols):
    return (i % ncols) == (ncols - 1)

def is_last_row(i, nrows, ncols):
    return (nrows - 1) * ncols <= i

def is_first_row(i, ncols):
    return i < ncols

def make_colorbar(fig, data, ax, cax, nrows, ncols, i):
    ticks = np.linspace(0, 1, 5)
    label = np.linspace(np.nanmin(data), np.nanmax(data), 5)
    if not is_last_row(i, nrows=nrows, ncols=ncols):
        ticks = ticks[1:]
        label = label[1:]
    label = [f'{l:0.1e}' for l in label]
    cmap = ScalarMappable(cmap='magma')
    #cmap.set_array()
    cb = fig.colorbar(mappable=cmap, ax=ax, cax=cax, shrink=None)
    cb.ax.get_yaxis().set_ticks(ticks=ticks, direction='inout', color='black', visible=True)
    cb.ax.get_yaxis().set_ticklabels(ticklabels=label, color='black', visible=True)
    cb.ax.tick_params(which='major', axis='both', length=5, colors='white', labelcolor='black',labelbottom=False, labeltop=False, labelleft=False, labelright=True, labelsize='10')
    if is_last_row(i, nrows=nrows, ncols=ncols):
        cb.ax.set_ylabel(r'Jy beam$^{-1}$')
    cb.ax.minorticks_off()
    return



def set_multiplot_ticks_fliplogic(cax, i, nrows, ncols, pixels, width, visible=True):
    firstcol = is_first_col(i, ncols=ncols)
    lastrow = is_last_row(i, ncols=ncols, nrows=nrows)
    cax.label_outer()
    labelcolor = 'black'
    cax.tick_params(axis='both', direction='inout', labelcolor=labelcolor, colors='white', labelsize='12', length=10,zorder=20)
    cax.minorticks_off()
    xtick_pix = np.linspace(0, pixels, 5)
    ytick_pix = np.linspace(0, pixels, 5)
    xtick_val = np.linspace(-width/2, width/2, 5)
    ytick_val = np.linspace(-width/2, width/2, 5)
    if visible:
        xlabels = [f'{x:0.2f}' for x in xtick_val[::-1]]
        ylabels = [f'{x:0.2f}' for x in ytick_val]
    else:
        xlabels = []
        ylabels = []
    cax.set_ylim(0, max(ytick_pix))
    cax.set_xlim(0, max(xtick_pix))
    if lastrow: # is the last row
        if firstcol:
            cax.set_xticks(ticks=xtick_pix)
            cax.set_xticklabels(labels=xlabels, rotation=45)
        else:
            cax.set_xticks(ticks=xtick_pix[1:])
            cax.set_xticklabels(labels=xlabels[1:], rotation=45)
    else:
        cax.set_xticks(ticks=xtick_pix)
        cax.set_xticklabels(labels=[], rotation=45)
    if not firstcol:
        # I cant begin to comprehend why this is backwards. Should be "if firstcol" but this instead gives the correct result. Is it
        if lastrow:
            cax.set_yticks(ticks=ytick_pix)
            cax.set_yticklabels(labels=ylabels)
        else:
            cax.set_yticks(ticks=ytick_pix[1:])
            cax.set_yticklabels(labels=ylabels[1:])
    else:
        cax.set_yticks(ticks=ytick_pix)
        cax.set_yticklabels(labels=[], rotation=45)
    return


def set_multiplot_ticks(cax, i, nrows, ncols, width, pixels, visible=True):
    firstcol = is_first_col(i, ncols=ncols)
    lastrow = is_last_row(i, ncols=ncols, nrows=nrows)
    cax.label_outer()
    labelcolor = 'black'
    cax.tick_params(axis='both', direction='inout', labelcolor=labelcolor, colors='white', labelsize='12', length=10,zorder=20)
    cax.minorticks_off()
    xtick_pix = np.linspace(0, pixels, 5)
    ytick_pix = np.linspace(0, pixels, 5)
    xtick_val = np.linspace(-width/2, width/2, 5)
    ytick_val = np.linspace(-width/2, width/2, 5)
    if visible:
        xlabels = [f'{x:0.2f}' for x in xtick_val[::-1]]
        ylabels = [f'{x:0.2f}' for x in ytick_val]
    else:
        xlabels = []
        ylabels = []
    cax.set_ylim(0, max(ytick_pix))
    cax.set_xlim(0, max(xtick_pix))
    '''
    cax.set_xticks(ticks=xtick_pix)
    cax.set_xticklabels(labels=xlabels, rotation=45)
    cax.set_yticks(ticks=ytick_pix)
    cax.set_yticklabels(labels=ylabels)
    return
    '''
    if lastrow: # is the last row
        if firstcol:
            cax.set_xticks(ticks=xtick_pix)
            cax.set_xticklabels(labels=xlabels, rotation=45)
        else:
            cax.set_xticks(ticks=xtick_pix[1:])
            cax.set_xticklabels(labels=xlabels[1:], rotation=45)
    else:
        cax.set_xticks(ticks=xtick_pix)
        cax.set_xticklabels(labels=[], rotation=45)
    if not firstcol:
        if lastrow:
            cax.set_yticks(ticks=ytick_pix)
            cax.set_yticklabels(labels=ylabels)
        else:
            cax.set_yticks(ticks=ytick_pix[1:])
            cax.set_yticklabels(labels=ylabels[1:])
    else:
        cax.set_yticks(ticks=ytick_pix)
        cax.set_yticklabels(labels=[], rotation=45)
    return

def drawbox(ax, xcen, ycen, width, **plt_kwargs):
    width = abs(width) / 2
    left = xcen + width
    right = xcen - width
    up = ycen + width
    down = ycen - width
    down, up = sorted([down, up])
    left, right = sorted([left, right])
    # bottom
    ax.plot([left, left], [down, up], linestyle='--', **plt_kwargs)
    # right
    ax.plot([right, right], [down, up], linestyle='--',**plt_kwargs)
    # left
    ax.plot([left, right], [up, up], linestyle='--',**plt_kwargs)
    # top
    ax.plot([left, right], [down, down], linestyle='--', **plt_kwargs)
    pass












