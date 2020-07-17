"""."""
# flake8: noqa
# TODO: work on this

# internal modules
import re
import datetime

# external modules
import numpy as np
import matplotlib.pyplot as plt
from nkrpy.apo.fits import read
from nkrpy.apo.reduction import tspec_orders, tspec_noisy_region
from nkrpy.files import list_files
from nkrpy import atomiclines
from nkrpy import constants
from nkrpy import functions as nfun
from nkrpy import miscmath as nmiscmath

# relative modules

# global attributes
__all__ = ('main',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

def _remove_tell(l1, val1, val2):
    mask = np.zeros(l1.shape,dtype=int)
    for i, x in enumerate(l1[0, :]):
        if x > val1 and x < val2:
            mask[:,i] = np.ones(l1.shape[0],dtype=int)
    return np.ma.masked_array(l1, mask=mask)

def _cal_continuum(l1, val1, val2):
    dlim = np.std(l1[1,:][list(map(lambda x: x[0], nfun.between(np.nan_to_num(l1[0, :]), val1, val2)))])
    return dlim
    
def clean_spec(v):
    l = deepcopy(v)
    for x in telluric:
        l = _remove_tell(l, x[0], x[1])
    return l

def plotlines(axis, values):
    labels = lingen.return_label()
    def getpos(v):
        if (np.min(v[0, :]) <= pj <= np.max(v[0, :])):
            index = nfun.find_nearest(v[0, :],pj)[0]
            lower = np.inf
            upper = np.inf
            if index <= 10:
                lower = 0
            if index >= len(val[0, :]) - 10:
                upper = len(val[0, :]) - 1
            if lower == np.inf:
                lower = index - 10
            if upper == np.inf:
                upper = index + 10
            try:
                linepos = np.max(v[1, :][lower:upper])
            except ValueError:
                linepos = 2*np.nanmean(v[1, :])
            return linepos
        else:
            return None
        
    for x in listoflines:
        naming = x
        for pl,pj in enumerate(listoflines[x]):
            linepos = getpos(values)
            if linepos == None:
                continue
            axis.plot((pj,pj),(linepos,linepos*1.05),'r-')
        for pl, pj in enumerate(labels[x]):
            linepos = getpos(values)
            if linepos == None:
                continue
            axis.text(pj, linepos*1.05, naming,
                verticalalignment='bottom',
                horizontalalignment='left',
                fontsize=15, color='red',rotation=35, zorder=20)

def get_name(file: str) -> str:
    regex = re.compile(r'tellcor\.fits$')
    fname = [x for x in file.split('/') if bool(regex.search(x))]
    try:
        return fname[0].strip('tellcor.fits').strip('_').strip('.')
    except:
        return ''


def reduce_name(name):
    return name.lower().strip(' ').strip('-').strip('_').strip('.')\
        .replace(' ', '').replace('-', '').replace('_', '').replace('.', '')


def get_date(file: str) -> str:
    regex = re.compile(r'^UT\d{6,}')
    date = [x for x in file.split('/') if bool(regex.search(x))]
    try:
        return date[0]
    except:
        return ''


def _order_dates(inputs: list, recent: bool = True) -> list:
    files = inputs.copy()
    dates = list(set([int(get_date(f).strip('UT'))
                      for f in files if 'UT' in f]))
    if recent:
        dates.sort(reverse=True)
    else:
        dates.sort()
    ret = []
    for d in dates:
        d = 'UT' + str(d)
        dated = []
        drop = [i for i, f in enumerate(files) if d in f]
        for i in sorted(drop, reverse=True):
            dated.append(files[i])
            del files[i]
        if len(dated) > 0:
            ret.append(dated)
    return ret


def _order_objects(inputs: list) -> list:
    files = inputs.copy()
    names = [x for x in list(set([reduce_name(get_name(f)) for f in files]))
             if x != '' and 'back' not in x]
    names.sort()
    ret = []
    for n in names:
        named = []
        drop = [i for i, f in enumerate(files) if n in reduce_name(get_name(f))]
        for i in sorted(drop, reverse=True):
            named.append(files[i])
            del files[i]
        if len(named) > 0:
            ret.append(named)
    return ret


def order_files(files: list, ordering: str = 'object') -> list:
    ordering = ordering.lower()
    assert ordering in ('date', 'object')
    ret = []
    if ordering == 'date':
        bydate = _order_dates(files)
        for dates in bydate:
            byname = _order_objects(dates)
            ret += byname
    else:
        byname = _order_objects(files)
        for names in byname:
            bydate = _order_dates(names)
            ret += bydate
    return nmiscmath.flatten(ret)


def reset_plot(fig, ax, nr, nc):
    for k in range(nr):
        for i in range(nc):
            ax[k, i].cla()
    pass


def save_plot(fig, pdf, numrows, numcols):
    fig.set_size_inches((numcols * 6, numrows * 3))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.07, \
                wspace=0.0,hspace=0.0)
    fig.set_dpi(300)
    pdf.savefig(fig, bbox_inches='tight')
    pass


def format_plot(ax, row, col, numcols, numrows, o, name):
    orde = col
    col = numcols - 1 - col
    ax[row, col].set_xlim(tspec_orders[orde][0],
                                      tspec_orders[orde][1])
    ax[row, col].tick_params(direction='in', length=3,
                                         width=1,
                                         colors='black',
                                         grid_color='black',
                                         grid_alpha=0.5,
                                         labelsize=4)
    if row == 0:
        ax[row, col].set_title(f"Order: {orde + 3}",
                                           fontsize=5)
    if (col != 0) or (col != (numcols -1)):
        ax[row, col].margins(0.)
    ax[row, 0].set_ylabel(r"Flux (ergs s$^{-1}$ cm$^{-2}$ A$^{-1}$)",
                          fontsize=5)
    pass


def draw_x(ax, k, l):
    ax[k, l].plot([0, 1], [0, 1], color='red', lw=3, transform=ax[k, l].transAxes)
    ax[k, l].plot([0, 1], [1, 0], color='red', lw=3, transform=ax[k, l].transAxes)
    # ax[k, l].axhline(y='0.5', color='red')
    pass

def selector(lam, flux):
    remove = list(np.argwhere(np.isnan(flux)))
    remove += list(np.argwhere(np.isinf(flux)))
    lam = np.array(lam)
    for region in tspec_noisy_region:
        remove += list(np.where(np.logical_and(lam>region[0], lam<=region[1]))[0])
    remove += list(np.argwhere(flux <= 0))
    ret = nmiscmath.listinvert(np.linspace(0, len(lam) - 1, num=len(lam)), remove)
    return np.array(ret, dtype=int)

if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    main()
    print('Test Passed')

# end of code

# end of file
