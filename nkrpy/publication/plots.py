"""."""
# flake8: noqa

# internal modules

# external modules

# relative modules

# global attributes
__all__ = ('set_style',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

import matplotlib.pyplot as plt

# relative modules

fontConv = 3. / 0.04167  # pts/inche

def set_style():
    """Setting the style."""
    # Direct input
    # Options
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}
    params = {'text.usetex': True,
              'font.size': 20,
              'font.family': 'sans-serif',
              'legend.fontsize': 16,
              'legend.handlelength': 2
              }
    plt.rc('font', **font)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rcParams.update(params)

    return


# end of code

# end of file
