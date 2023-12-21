"""."""
# flake8: noqa

# internal modules

# external modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

# relative modules
from ..misc import constants
print(constants)
golden = constants.golden

# global attributes
__all__ = ['set_style', 'Arrow3D', 'ppi']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

ppi = 3. / 0.04167  # pts/inche

styles = {
    'mnras': f'{__path__}/mnras.mplsty',
}


def set_style(style: str = None, usetex: bool = False):
    """Setting the style."""
    # Direct input
    # Options
    if style is not None:
        style = style.lower()
        plt.style.use(styles[style])
        rcParams['text.usetex'] = usetex
        rcParams['axes.titlesize']      = 'large'
        return
    font = {'family': 'sans-serif',
            'weight': 'bold'}
    plt.rc('font', **font)
    rcParams['axes.titlesize']      = 48
    rcParams['axes.labelsize']      = 12
    rcParams['axes.labelweight']     = 'bold'
    rcParams['xtick.labelsize']     = 12
    rcParams['ytick.labelsize']     = 12
    rcParams['legend.fontsize']     = 12
    rcParams['axes.linewidth']      = 1.25
    rcParams['xtick.major.size']    = 2.5
    rcParams['xtick.minor.size']    = 1.5
    rcParams['xtick.major.width']   = 1.25
    rcParams['xtick.minor.width']   = 1.25
    rcParams['ytick.major.size']    = 2.5
    rcParams['ytick.minor.size']    = 1.5
    rcParams['ytick.major.width']   = 1.25
    rcParams['ytick.minor.width']   = 1.25
    #rcParams['text.usetex']         = True
    rcParams['xtick.major.pad']     = 6
    rcParams['ytick.major.pad']     = 6
    rcParams['ytick.direction']     = 'in'
    rcParams['xtick.direction']     = 'in'
    rcParams['figure.figsize']      = 10., 10./golden


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
