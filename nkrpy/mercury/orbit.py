"""Handle Orbital 3D Plotting using multi-core."""

# internal modules
from multiprocessing import Process
import multiprocessing as mp
import pickle
import os

# external modules
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') # noqa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# relative modules
from ..load import load_cfg, verify_dir
from .file_loader import parse_aei

# global attributes
__all__ = ('main',)
__doc__ = """This packages tries to be fairly robust and efficient, utilizing the speedups offered via numpy where applicable and multicore techniques. To get started, simply need a config file and call orbit.main(config). Inside the config should be mostly 3 things: files<input file list> out_dir<outputdirectory> and out_name<unique output name>. A lot of files will be generated (sometimes tens of thousands). The end goal is matplotlib libraries are ineffient for animation creation, so static thumbnails are created and then a imagmagick shell script is created to utilize a more efficient program."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1
mpl_colours = ('black', 'blue', 'green', 'red', 'purple', 'Cyan')
cpu = mp.cpu_count()


def plot_orbit(ax, data, name, color='black'):
    """Plot the orbit of a given target."""
    x, y, z = data

    if x * y > 0:
        ax.scatter(x, y, zs=z, zdir='z', marker='^', label=name,
                   c=color, s=2)
    else:
        ax.scatter(x, y, zs=z, zdir='z', marker='v', label=name,
                   c=color, s=2)
    pass


def draw(a):
    """."""
    names, index, x, meta, dest, cfg, num, ax, fig = a

    ax.scatter([0], [0], zs=[0], label='center', marker='+',
               c='firebrick', s=10)
    x_s = np.linspace(meta[0][0], meta[1][0], 100)
    y_s = np.linspace(meta[0][1], meta[1][1], 100)
    X, Y = np.meshgrid(x_s, y_s)
    Z = np.zeros((X.shape))

    ax.plot_surface(X, Y, Z , alpha=0.2,
                    linewidth=0, antialiased=False)
    for j, name in enumerate(names):
        plot_orbit(ax, (x[j, index[0]], x[j, index[1]], x[j, index[2]]),
                   name, mpl_colours[j % len(mpl_colours)])
    ax.set_xlim3d(meta[0][0], meta[1][0])
    ax.set_ylim3d(meta[0][1], meta[1][1])
    ax.set_zlim3d(meta[0][2], meta[1][2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=10.)
    ax.set_ylim3d(ax.get_ylim()[::-1])
    fig.legend()
    fig.savefig(f'{dest}/{cfg.out_name}_{num}.png', dpi=200)
    fig.clf()
    print(f'Finished {num}')


def main(cfgname):
    # load
    print('Loading...')
    cfg = load_cfg(cfgname)
    dest = f"{cfg.cur_dir}/{cfg.out_dir}"
    verify_dir(dest)
    names = []
    bodies = []
    # read in data and reshape
    if os.path.isfile(f'{dest}/{cfg.out_name}.dat') and os.path.isfile(f'{dest}/{cfg.out_name}.head'):
        with open(f'{dest}/{cfg.out_name}.head', 'rb') as r_h: 
            header = list(pickle.load(r_h))
        with open(f'{dest}/{cfg.out_name}.dat', 'rb') as r_d: 
            bodies = pickle.load(r_d)
        names = [x.strip('.aei').split('/')[-1] for x in cfg.files]
    else:
        for x in cfg.files:
            name, header, data = parse_aei(x)
            names.append(name)
            bodies.append(data)
        bodies = np.array(bodies).reshape(-1, len(names), len(header))
    # gather useful meta
    datashape = bodies.shape
    index = tuple([header.index('x'), header.index('y'), header.index('z')])
    meta = ((np.min(bodies[:, :, index[0]]) / 3.,
             np.min(bodies[:, :, index[1]]) / 3.,
             np.min(bodies[:, :, index[2]]) / 3.),
            (np.max(bodies[:, :, index[0]]) / 3.,
             np.max(bodies[:, :, index[1]]) / 3.,
             np.max(bodies[:, :, index[2]]) / 3.))
    # bodies.shape (#objects, # integrations, #header)
    # now begin plotting
    print('Plotting...')
    processes = [None for x in range(cpu - 1)]
    fig = [None for x in range(cpu - 1)]
    ax = [None for x in range(cpu - 1)]
    kick = None
    i, x = 0, 0
    while i < len(bodies):
        x = bodies[i]
        p_count = i % cpu - 1

        num = ''.join(['0' for _t in range(len(str(datashape[0])) -
                                           len(str(i)))]) +\
              str(i)

        if processes[p_count]:
            if not processes[p_count].is_alive():
                a = (names, index, x, meta, dest, cfg, num,
                     ax[p_count], fig[p_count])
                ax[p_count].set_title(f'{num}/{datashape[0]}')
                processes[p_count] = Process(target=draw, args=(a,))
                processes[p_count].start()
                i += 1
        else:
            fig[p_count] = plt.figure(figsize=(4, 4))
            ax[p_count] = fig[p_count].add_subplot(111, projection='3d') 
            ax[p_count].set_title(f'{num}/{datashape[0]}')
            a = (names, index, x, meta, dest, cfg, num,
                 ax[p_count], fig[p_count])
            processes[p_count] = Process(target=draw, args=(a,))
            processes[p_count].start()
            i += 1

    for process in processes:
        process.join()
    print('Writing data...')
    with open(f'{dest}/{cfg.out_name}.dat', 'wb') as f_b:
        pickle.dump(bodies, f_b, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{dest}/{cfg.out_name}.head', 'wb') as f_h:
        pickle.dump(header, f_h, protocol=pickle.HIGHEST_PROTOCOL)

    print_cmd(dest, cfg, datashape[0])

def print_cmd(dest, cfg, length, chunk=100):
    """."""
    _t = ''
    count = 0
    with open(f'{cfg.out_name}_gifgen.sh', 'w') as f:
        for x in range(length):
            num = ''.join(['0' for _d in range(len(str(length)) -
                                               len(str(x)))]) +\
                  str(x)
            _t += f'{dest}/{cfg.out_name}_{num}.png '
            if ((x % chunk) == (chunk - 1)) or (x == (length - 1)) and (_t != ''):
                f.write(f'convert -loop 0 -delay 20 {_t} {dest}/chunk_{count}.gif &\n')
                if (count % cpu) == (cpu - 1):
                    f.write(f'echo -ne "Progress: [{count + 1}/{length / chunk}]"\\\\r\n')
                    f.write('wait\n')
                _t = ''
                count += 1
        f.write(f'wait\n')
        f.write(f'echo "Only Chunks of gifs are made, run the last command yourself: `echo {cfg.out_name}_gifgen.sh | tail -n1`"\n')
        f.write('# The next command is commented out because can eat RAM\n')
        f.write(f'# convert -loop 0 -delay 20 {dest}/chunk_*.gif {dest}/{cfg.out_name}.gif\n')
    print(f'Run the command: {cfg.out_name}_gifgen.sh')
    print('I commented out the last line as it can take a shocking amount of RAM')
    pass


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
