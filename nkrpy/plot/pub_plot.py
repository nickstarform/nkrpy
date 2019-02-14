"""Common Nice plotting settings."""

# standard modules

# external modules
import matplotlib.pyplot as plt

# relative modules


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
