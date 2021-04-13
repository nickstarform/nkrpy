"""Provides common functions for reducing TSPEC data.

# fit continuum
# fit line
# eqwidth
# pretty plotting
"""

# internal modules

# external modules
import numpy as np
from scipy.optimize import curve_fit

# relative modules
from ...math.fit import linear

# global attributes
__all__ = ('tspec_orders', 'determine_order_tspec', 'tspec_noisy_region',
           'tspec_resolution')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


# make class per target
# allow for a lot of data to be loaded
# save to output file for quick load without loading fits again
tspec_orders = ((1.88, 2.47),  # 3
                (1.40, 1.88),  # 4
                (1.13, 1.49),  # 5
                (0.94, 1.241),  # 6
                (0.94, 1.065))  # 7
tspec_noisy_region = ((2.4, 9),
                      (1.81, 1.94),
                      (1.35, 1.425),
                      (-1, 0.96))
tspec_resolution = 3500.


def split_into_tspec_orders(wav: np.ndarray, conserve: bool = True):
    """Split an array of wavelengths into orders.

    wav must be a numpy array. Will return a list
    of boolean arrays that denote what indexes are
    per order. If conserve is set, will favour long
    wavelength orders more, otherwise will simply yield
    wavelengths in those ranges.
    """
    orders = [np.where(np.logical_and(wav > x[0], wav < x[-1]))[0] for x in tspec_orders]
    return orders


def determine_order_tspec(wav, sort: bool=False):
    """Determine order given wavelength range.

    Specifically for APO Tspec instrument, try
    to determine the order of a wavelength range.
    If no order found, returns -1.

    Parameters
    ----------
    wav : iterable
        Wavelength range with at least 2 elements.
        These should be increasing lower->higher

    Returns
    -------
    list
        List of all the orders spanned

    """
    if sort:
        wav.sort()
    if not isinstance(wav, np.ndarray):
        wav = np.array(wav)
    orders = np.ones(wav.shape, dtype=np.int)
    for i, lims in enumerate(tspec_orders):
        mask = ~(~(wav >= lims[0]) + ~(wav <= lims[1]))
        orders[mask] = i + 3
    return orders


def plotting(ax, xmin, xmax, x, y, tempsource, line, count, start=False):
    colours = ['orange', 'black', 'blue', 'red',
               'green', 'grey', 'purple']
    colour = colours[count % len(colours)]
    y = np.array(y)
    x = np.array(x)
    origx = x.copy()
    origy = y.copy()
    x = x[~np.isnan(origy)]
    y = y[~np.isnan(origy)]
    print("Count: {}, Before: {}, {}".format(count, x.shape, y.shape))
    if not start:
        temp = []
        if count == 0:
            for i, j in enumerate(x):
                if (j < 1.7):
                    temp.append(i)
        elif count == 1:
            for i, j in enumerate(x):
                if ((1.75 < j) or (j < 1.5)):
                    temp.append(i)
        elif count == 2:
            for i, j in enumerate(x):
                if (1.33 < j) or (j < 1.17):
                    temp.append(i)
        elif count == 3:
            for i, j in enumerate(x):
                if ((j < 1.05) or ((1.11 < j)) and (j < 1.17)):
                    temp.append(i)
        elif count == 4:
            for i, j in enumerate(x):
                if (j < 0.95):
                    temp.append(i)
        temp = np.array(temp)
        temp.sort()
        tempx = np.delete(x, temp)
        tempy = np.delete(y, temp)
        expected = [1., 1.]
        params, cov = curve_fit(linear, tempx, tempy, expected)
        # print(params)
        if len(temp) > 0:
            y[temp[0]] = linear(x[temp[0]], *params)
            y[temp[len(temp) - 1]] = linear(x[temp[len(temp) - 1]], *params)
            x = np.delete(x, temp[1:len(temp) - 1])
            y = np.delete(y, temp[1:len(temp) - 1])

        print("After: {}, {}".format(x.shape, y.shape))
        if x.shape[0] == 0:
            x = origx
            y = origy
        count += 1
    ax.plot(x, y, '-', color=colour, label=tempsource[-6:])
    for f in line:
        # print(f)
        if f == 'brg':
            naming = r'Br $\gamma$'
        elif f == 'pab':
            naming = r'Pa $\beta$'
        elif f == 'pag':
            naming = r'Pa $\gamma$'
        else:
            naming = f
        for pl, pj in enumerate(line[f]):
            # print(pl, pj)
            if (pl < (len(line[f]))):
                if pj > 100:
                    pj = pj / 10000
                val = int(int(min(range(len(x)), key=lambda i: abs(x[i] - pj))))  # noqa
                if (xmin <= pj <= xmax) and (min(x) <= pj <= max(x)):
                    if 10 <= val < len(x) - 11:
                        region = y[val - 10:val + 10]
                    elif val > 0:
                        region = y[val:val + 10]
                    elif val < len(x) - 1:
                        region = y[val - 10:val]
                    else:
                        region = y[val]
                    try:
                        linepos = max(region)
                    except ValueError:
                        linepos = 5 * np.nanmean(x)
                    ax.text(pj, linepos * 1.05, naming,
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            fontsize=10, color='red', rotation='vertical')

                    ax.plot((pj, pj), (linepos, linepos * 1.05), 'r-')

# end of code

# end of file
