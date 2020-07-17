"""Raw Generic Plotter for Mercury."""

# standard modules
import os
from sys import version

# external modules
import numpy as np
import matplotlib.pyplot as plt

# relative modules
from ..load import load_cfg, verify_dir
from .file_loader import parse_aei

__version__ = float(version[0:3])
__cwd__ = os.getcwd()


def choose2Pairs(head):
    """."""
    n = len(head)
    numpairs = n*(n-1)/2.
    pairs = []
    for i, x in enumerate(head):
        for j, y in enumerate(head):
            if j > i:
                pairs.append(x + ' ' + y)
    return numpairs, pairs


def plotting(x, y, xl, yl, title=None, fig=None, ax=None):
    """."""
    if not fig and not ax:
        fig, ax = plt.figure(figsize=(10, 10))
    ax.scatter(x, y)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)


def flip(orig):
    """."""
    toReturn = np.ndarray(orig.shape[::-1])
    for i, row in enumerate(orig):
        for j, col in enumerate(row):
            toReturn[j, i] = col

    return toReturn


def main(configfname):
    """.

    @input : configFname is the name of the configuration file.
    Has a default value just incase
    Loads in all of the values found in the configuration file.
    """
    config = load_cfg(configfname)

    for i, f in enumerate(config['files']):
        oName, header, odata = parse_aei(f)
        data = flip(odata)  # .reshape(len(header), -1)

        for x in choose2Pairs(header)[1]:
            print(x)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            p1, p2 = x.split(' ')
            outfile = config['output'].replace('-N-', oName)\
                                      .replace('-P-', ''.join([p1, p2]))
            direct = outfile.split('/')[0]
            verify_dir(direct)
            x = data[header.index(p1), :]
            y = data[header.index(p2), :]
            plotting(x, y, p1, p2, title=''.join([p1, p2]), fig=fig, ax=ax)
            plt.autoscale(enable=True, axis='both', tight=None)
            plt.tight_layout(pad=1.02, h_pad=None, w_pad=None, rect=None)
            plt.savefig(outfile, dpi=200)
            plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input',
                        type=str, help='input', required=True)

    args = parser.parse_args()
    main(args.input)

# end of file
