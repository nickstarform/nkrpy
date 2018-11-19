"""Common Nice plotting settings."""

# standard modules

# external modules
import matplotlib.pyplot as plt
import matplotlib as mpl

# relative modules


def set_latex():
    """Setting the style."""
    # Direct input
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    # Options
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}
    params = {'text.usetex': True,
              'font.size': 20,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              'legend.fontsize': 8,
              'legend.handlelength': 2
              }
    mpl.rc('font', **font)
    plt.rcParams.update(params)

    return
