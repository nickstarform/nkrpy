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
from ..._types import PlotClass
from ...misc.functions import dict_split

# global attributes
__all__ = ['Plot', ]
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class Plot(PlotClass):
    """Generalized plotting class."""
    def __init__(self, *args, fig = None, ax = None, **kwargs):
        kwargs, pltcfg = dict_split(kwargs, ['fig', 'ax'])
        if fig is None:
            self.fig, self.axes = plt.subplots(*args, **kwargs)
        else:
            self.fig, self.axes = fig, ax
        if not isinstance(self.axes, np.ndarray):
            self.ax = self.axes
            self.axes = np.array(self.axes)
        else:
            self.ax = self.axes[0]
        self.__iteration = 0
        self.axis_to_data = self.ax.transAxes + self.ax.transData.inverted()
        self.data_to_axis = self.axis_to_data.inverted()
        pass

    def next(self):
        if self.__iteration == (self.axes.shape[0] - 1):
            self.ax = self.axes[0]
            self.__iteration = 0
            return
        self.ax = self.axes[self.__iteration + 1]
        self.__iteration += 1

    def ellipse(self, xcen: float, ycen: float, major: float, minor: float, angle: float = 0, **plt_kwargs):
        """

        In data coords, east of north
        """
        major, minor if major < minor else minor, major
        ellipse = Ellipse(xy=(xcen, ycen), width=major, height=minor, angle=angle, **plt_kwargs)
        self.ax.add_patch(ellipse)
        return ellipse

    def circle(self, xcen: float, ycen: float, radius: float, **plt_kwargs):
        self.ellipse(xcen=xcen, ycen=ycen, major=radius, minor=radius, angle=0, **plt_kwargs)

    def square(self, xcen: float, ycen: float, width: float, angle: float, **plt_kwargs):
        self.rectangle(xcen=xcen, ycen=ycen, xwidth=width, ywidth=width, angle=angle, **plt_kwargs)

    def cone(self, xcen: float, ycen: float, r1: float, r2: float, angle: float, pa: float = 0, capped: bool = True, **plt_kwargs):
        # first section
        self.vector(xcen=xcen, ycen=ycen, length=r1,angle=pa + angle / 2, **plt_kwargs)
        # first section
        self.vector(xcen=xcen, ycen=ycen, length=r2,angle=pa - angle / 2, **plt_kwargs)
        if capped:
            angle = angle % 360
            radians = -1. * (pa + angle / 2)  * np.pi / 180. + np.pi # originally angle is west of south, swap to east of north
            coradians =  -1. * (pa - angle / 2) * np.pi / 180. + np.pi # originally angle is west of south, swap to east of north
            left = (xcen + r1 * np.sin(radians), ycen + r1 * np.cos(radians))
            right = (xcen + r1 * np.sin(coradians), ycen + r1 * np.cos(coradians))
            self.ax.plot([left[0], right[0]], [left[1], right[1]], **plt_kwargs)

    def conic_section(self, xcen: float, ycen: float, r1: float, r2: float, angle: float, pa:float, num_points: int = 100, **plt_kwargs):
        # a cylindrical conic section of radii r1 > r2 and angle
        # cylinder 1
        #    one corner
        rad1 = np.pi / 180 * (pa - angle / 2) - np.pi / 2
        rad2 = np.pi / 180 * (pa + angle / 2) - np.pi / 2
        a = np.linspace(rad1, rad2, num_points, dtype=float)
        for r in [r1, r2]:
            x = np.cos(a) * r + xcen
            y = np.sin(a) * r + ycen
            self.ax.scatter(x, y, **plt_kwargs)
        self.vector(xcen=xcen + np.cos(rad1) * r2, ycen=ycen + np.sin(rad2) * r2, length=r1-r2,angle=pa - angle / 2, **plt_kwargs)
        # first section
        self.vector(xcen=xcen + np.cos(rad2) * r2, ycen=ycen + np.sin(rad2) * r2, length=r1-r2,angle=pa + angle / 2, **plt_kwargs)

    def rectangle(self, xcen: float, ycen: float, xwidth: float, ywidth: float, angle: float = 0, **plt_kwargs):
        """Plot a vector, east of north"""
        if 'color' not in plt_kwargs:
            color = 'black'
        else:
            color = plt_kwargs['color']
        radians = -1. * angle * np.pi / 180. + np.pi # originally angle is west of south, swap to east of north
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
        self.ax.plot([lrx, llx], [lry, lly], color=color, **plt_kwargs)
        # right
        self.ax.plot([lrx, urx], [lry, ury], color=color, **plt_kwargs)
        # left
        self.ax.plot([llx, ulx], [lly, uly], color=color, **plt_kwargs)
        # top
        self.ax.plot([urx, ulx], [ury, uly], color=color, **plt_kwargs)

    def marker(self, xcen: float = None, ycen: float = None, axis: str = 'x', both: bool = True, invert: bool = False, label: str = '', text_invert: bool= True, axis_fraction: float = 0.025, rotation: float = 45, **matplotlib_plot_kwargs):
        # plot a marker on either the x or y axis. If both, then plot both. If not both then invert will decide which side to plot

        plotting_sequence = [0, 1] if both else [1] if invert else [0]
        for i in plotting_sequence:
            xvals = [xcen, xcen] if axis == 'x' else [-axis_fraction + i, axis_fraction + i]
            yvals = [ycen, ycen] if axis == 'y' else [-axis_fraction + i, axis_fraction + i]
            transform = self.ax.get_xaxis_transform() if axis == 'x' else self.ax.get_yaxis_transform()
            self.ax.plot(xvals, yvals, transform=transform, **matplotlib_plot_kwargs)
        i = 0 if not text_invert else 1
        xvals = [xcen, xcen] if axis == 'x' else [-axis_fraction + i, axis_fraction + i]
        yvals = [ycen, ycen] if axis == 'y' else [-axis_fraction + i, axis_fraction + i]
        transform = self.ax.get_xaxis_transform() if axis == 'x' else self.ax.get_yaxis_transform()
        x = xvals[1] if text_invert else xvals[0]
        y = yvals[1] if text_invert else yvals[0]
        self.ax.text(x=x, y=y, s=label, rotation=rotation, transform=transform)
        label = ''

    def scale(self, height_width: float = 1):
        self.ax.set_aspect(height_width/self.ax.get_data_ratio())

    def xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)

    def colorbar(self, *args, **kwargs):
        cb = self.fig.colorbar(*args, ax=self.ax,  **kwargs)
        return cb

    def xlim(self, *args):
        self.ax.set_xlim(args)

    def ylim(self, *args):
        self.ax.set_ylim(args)

    def cla(self):
        for ax in self.axes:
            ax.cla()

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        if 'cmap' in kwargs:
            self.__cmap = kwargs['cmap']
        self.ax.imshow(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        self.ax.scatter(*args, **kwargs)

    def contour(self, *args, **kwargs):
        self.ax.contour(*args, **kwargs)

    def vector(self, xcen: float, ycen: float, length: float,angle: float = 0, **plt_kwargs):
        """Plot a vector, east of north"""
        angle = angle % 360
        radians = -1. * angle * np.pi / 180. + np.pi # originally angle is west of south, swap to east of north
        self.ax.plot([xcen, xcen + length * np.sin(radians)], [ycen, ycen + length * np.cos(radians)], **plt_kwargs)

    def get_colorbar(self):
        return self.fig

    def get_fig(self):
        return self.fig

    def get_ax(self):
        return self.ax

    def save(self, fname: str, dpi: int = 150, **plt_kwargs):
        self.fig.savefig(fname, dpi=dpi, **plt_kwargs)

    def savefig(self, *args, **kwargs):
        self.save(*args, **kwargs)


# end of code

# end of file
