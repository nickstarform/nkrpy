"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transform
import numpy as np

# relative modules
from .._types import PlotClass
from ..misc.functions import dict_split

# global attributes
__all__ = ('Plot', )
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class Plot(PlotClass):
    """Generalized plotting class."""
    def __init__(self, *args, fig = None, ax = None, **kwargs):
        kwargs, pltcfg = dict_split(kwargs, ['fig', 'ax'])
        if fig is None:
            self.__fig, self.__ax = plt.subplots(*args, **kwargs)
        else:
            self.__fig, self.__ax = fig, ax
        self.axis_to_data = self.__ax.transAxes + self.__ax.transData.inverted()
        self.data_to_axis = self.axis_to_data.inverted()
        pass

    def ellipse(self, xcen: float, ycen: float, major: float, minor: float, angle: float = 0, **plt_kwargs):
        """

        In data coords
        """
        major, minor if major < minor else minor, major
        ellipse = Ellipse(xy=(xcen, ycen), width=major, height=minor, angle=angle, **plt_kwargs)
        self.__ax.add_patch(ellipse)

    def circle(self, xcen: float, ycen: float, radius: float, **plt_kwargs):
        self.ellipse(xcen=xcen, ycen=ycen, major=radius, minor=radius, angle=0, **plt_kwargs)

    def square(self, xcen: float, ycen: float, width: float, angle: float, **plt_kwargs):
        self.rectangle(xcen=xcen, ycen=ycen, xwidth=width, ywidth=width, angle=angle, **plt_kwargs)

    def rectangle(self, xcen: float, ycen: float, xwidth: float, ywidth: float, angle: float = 0, **plt_kwargs):
        radians = angle * np.pi / 180.
        left = xcen + xwidth
        right = xcen - xwidth
        up = ycen + ywidth
        down = ycen - ywidth
        newx = lambda x, y, theta: (x * np.cos(-theta) - y * np.sin(-theta)) + xcen
        newy = lambda x, y, theta: (x * np.sin(-theta) + y * np.cos(-theta)) + ycen
        newxy = lambda x, y, theta: (newx(x, y, theta), newy(x, y, theta))
        urx, ury = newxy(xwidth / 2., ywidth / 2., radians)
        ulx, uly = newxy(-xwidth / 2., ywidth / 2., radians)
        lrx, lry = newxy(xwidth / 2., -ywidth / 2., radians)
        llx, lly = newxy(-xwidth / 2., -ywidth / 2., radians)
        # bottom
        self.__ax.plot([lrx, llx], [lry, lly], **plt_kwargs)
        # right
        self.__ax.plot([lrx, urx], [lry, ury], **plt_kwargs)
        # left
        self.__ax.plot([llx, ulx], [lly, uly], **plt_kwargs)
        # top
        self.__ax.plot([urx, ulx], [ury, uly], **plt_kwargs)

    def scale(self, ratio='same'):
        if isinstance(ratio, float) or isinstance(ratio, int):
            self.__ax.set_aspect(ratio)
        else:
            self.__ax.set_aspect(1./self.__ax.get_data_ratio())

    def xlabel(self, *args, **kwargs):
        self.__ax.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        self.__ax.set_xlabel(*args, **kwargs)

    def colorbar(self, *args, **kwargs):
        cb = self.__fig.colorbar(*args, ax=self.__ax,  **kwargs)
        return cb

    def xlim(self, *args):
        self.__ax.set_xlim(args)

    def ylim(self, *args):
        self.__ax.set_ylim(args)

    def cla(self):
        self.__ax.cla()

    def plot(self, *args, **kwargs):
        self.__ax.plot(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        if 'cmap' in kwargs:
            self.__cmap = kwargs['cmap']
        self.__ax.imshow(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        self.__ax.scatter(*args, **kwargs)

    def contour(self, *args, **kwargs):
        self.__ax.contour(*args, **kwargs)

    def vector(self, xcen: float, ycen: float, length: float,angle: float = 0, **plt_kwargs):
        radians = angle * np.pi / 180.
        self.__ax.plot([xcen, xcen + length * np.cos(radians)], [ycen, ycen + length * np.sin(radians)], **plt_kwargs)

    def get_colorbar(self):
        return self.__fig

    def get_fig(self):
        return self.__fig

    def get_ax(self):
        return self.__ax


    def marker(self, *args, **kwargs):
        self.__ax.scatter(*args, **kwargs)

    def save(self, fname: str, dpi: int = 150):
        self.__fig.savefig(fname, dpi=dpi)


# end of code

# end of file
