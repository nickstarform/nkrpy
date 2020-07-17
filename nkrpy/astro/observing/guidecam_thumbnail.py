"""."""

# internal modules
import os
from multiprocessing import Process
import multiprocessing as mp
from argparse import ArgumentParser

# external modules
import matplotlib.pyplot as plt
import numpy as np

# relative modules
from ...io import fits
from ...io.files import list_files

# global attributes
__all__ = ('main', 'thumbnails')
__doc__ = """Just call this module as a file while
    inside the directory of guidecam images."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
_cwd_ = os.getcwd()
cpu = mp.cpu_count()


def thumbnails(ifile, ax, fig, dpi=200):
    """Create thumbnails."""
    outname = ifile.replace('.fits', '.png')
    header, data = fits.read(ifile)
    header = header[0]
    data = data[0]
    data = np.log10(data)
    med = np.median(data)
    ax.imshow(data / med, cmap='gray', vmin=0.9995, vmax=1.001, origin='left')
    fig.savefig(outname, dpi=dpi)


def main(dest, ifile):
    """."""
    if not ifile:
        _filelist = [x for x in list_files(dest) if 'fits' in x]
    else:
        _filelist = (ifile,)

    processes = [None for x in range(cpu - 1)]
    fig = [None for x in range(cpu - 1)]
    ax = [None for x in range(cpu - 1)]

    i, x = 0, 0
    while i < len(_filelist):
        x = _filelist[i]
        p_count = i % cpu - 1

        if processes[p_count]:
            if not processes[p_count].is_alive():
                a = (x, ax[p_count], fig[p_count])
                ax[p_count].set_title(f'{x}')
                processes[p_count] = Process(target=thumbnails, args=a)
                processes[p_count].start()
                i += 1
        else:
            fig[p_count] = plt.figure(figsize=(4, 4))
            ax[p_count] = fig[p_count].add_subplot(111)
            ax[p_count].set_title(f'{x}')
            a = (x, ax[p_count], fig[p_count])
            processes[p_count] = Process(target=thumbnails, args=a)
            processes[p_count].start()
            i += 1

    for process in processes:
        if process:
            process.join()


if __name__ == "__main__":
    """Directly Called."""
    parser = ArgumentParser()
    parser.add_argument('-f', dest='f', help='Only a single file', default='')
    args = parser.parse_args()

    print('Running Module')
    main(_cwd_, args.f)
    print('Finished')


# end of file
