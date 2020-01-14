"""."""
# flake8: noqa

# internal modules

# external modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

# relative modules

# global attributes
__all__ = ('set_style', 'Arrow3D')
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


class Arrow3D(FancyArrowPatch):
    """Hack to add fancy arrow heads in 3D.

    Example
    -------
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = figadd_subplot(111, projection='3d')
    a = Arrow3D([x1, x2], [y1, y2], [z1, z2],
                mutation_scale=20, lw=3, arrowstyle='-|>',
                color='red')
    ax.add_artist(a)
    This will draw
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Dunder."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = (xs, ys, zs)

    def draw(self, renderer):
        """Draw method for patches."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# end of code

# end of file
