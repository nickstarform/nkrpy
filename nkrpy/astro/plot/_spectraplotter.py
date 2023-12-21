"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# relative modules
from ...publication._plots import set_style, ppi
from ...misc.functions import typecheck
from ..._unit import Unit as unit
from ..._math import _convert as nkrpy__convert
from ..._math._miscmath import ellipse_distance
from ..._math._miscmath import rms as math_rms
icrs2degrees, degrees2icrs = nkrpy__convert.icrs2degrees, nkrpy__convert.degrees2icrs
from ...misc import constants
from ...publication.cmaps import mainColorMap, mapper
from ._plot import Plot
from ..._types import WCSClass
from .._wcs import WCS
from ..._unit import Unit
from ...io import fits
from ..._math import gauss, binning, sigma_clip_fit, polynomial
# global attributes
__all__ = ['SpectraPlot', 'auto_spectra_plot']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
#set_style()


def auto_spectra_plot(filename: str):
    h, d = fits.read(filename)
    wcs = WCS(h)
    if len(d.shape) > 2:
        m = np.sum(np.squeeze(d), axis=0)
    else:
        m = d
    w = WCSImagePlot(wcs=wcs, data=m, cmap='cividis')
    w.set_image(m)
    wcs = w.get_wcs()
    ra = wcs(wcs.axis1['axis'] / 2, 'wcs', wcs.axis1['type'])
    dec = wcs(wcs.axis2['axis'] / 2, 'wcs', wcs.axis2['type'])
    w.add_source(ra, dec, marker='.',color='red', s=4)
    w.add_beam(edgecolor='black', lw=1, facecolor='white')
    scale = wcs(wcs.axis1['axis'] / 10., 'wcs', wcs.axis1['type']) - wcs(0, 'wcs', wcs.axis1['type'])
    w.add_scalebar(scale, wcs.axis1['unit'], wcsaxis=wcs.axis1['type'], facecolor='white', edgecolor='black', lw=1)
    w.add_contour(start=5, stop=100, plot_negative=False)
    w.xlabel(wcs.axis1['type'])
    w.ylabel(wcs.axis2['type'])
    w.set_colorbar(colortitle=wcs.get_head('bunit'), pad=0)
    w.add_title(wcs.get_head('field'), fontsize='larger')
    w.set_ticks(xy='x', rotation=45)
    w.set_ticks(xy='y')
    w.set_limits()
    w.save('test.pdf', bbox_inches='tight')


class SpectraPlot(Plot):
    """Generic Spectra plotter.

    Methods
    -------
    add_gaussian: plot beams
    add_spectra: plot background (data, color)
    add_horizontal_line: plot contours  (contour params)
    add_vertial_line: plot outflows (vectors, xcen, ycen, angle, color, length)
    add_title: plot sources (marker styles)
    add_ticks: plot regions (ellipse, rectangle, xcen, ycen, major/xwidth, minor/ywidth, angle)
    add_transitions: plot custom scalebars
    fit_line
    fit_spectra
    fit_confinuum
    invert_axis
    set_limits
    set_yscale
    convert_xaxis
    convert_yaxis:
    refresh_data: reslices the data from the original using the loaded wcs

    read in data. Either as 1d wcs + in order brightness or 2x1d array and ignore wcs
    Then only handle the frequeny and brightness
    """

    def __init__(self, *args, xdata: np.ndarray=None,ydata: np.ndarray=None, xerror: np.ndarray=None, yerror: np.ndarray=None, **pltcfg):
        # assert some needed params
        # initalize parent plot
        super(SpectraPlot, self).__init__(*args, **pltcfg)
        self.matplotlib_renderer = self.fig.canvas.get_renderer()
        # define some tracking bariables

        self.xdata = xdata
        self.ydata = ydata
        self.xerror = xerror
        self.yerror = yerror
        # generate new data
        self.reset_axes()

    def invert_axis(self, xy='x'):
        if 'x' in xy:
            self.ax.invert_xaxis()
        if 'y' in xy:
            self.ax.invert_yaxis()

    def set_xlim(self, lower: float, upper: float):
        self.xlim = [lower, upper]
        self.ax.set_xlim(lower, upper)

    def set_ylim(self, lower: float, upper: float):
        self.ylim = [lower, upper]
        self.ax.set_ylim(lower, upper)

    def poststamp(self):
        """Format the plot for tight layouts."""
        self.ax.grid(False)
        self.ax.set_axis_on()
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')

    def reset_all_axes(self):
        for _ in self.axes:
            self.reset_axes()
            self.next()

    def reset_axes(self):
        self.ax.cla()
        self.__last_gaussian_height = 0
        self.__last_title_height = 0
        self.__gaussians = []
        self.__errors = []

    def add_title(self, title: str, xcen: float = None, ycen: float = None, verticalalignment: str = 'top', horizontalalignment: str = 'center', **text_kwargs):
        if ycen is None:
            if self.__last_title_height != 0:
                ycen = self.ylim - self.__last_title_height
            else:
                ycen = self.ylim[-1]
        if xcen is None:
            xcen = np.diff(self.xlim) / 2. + self.xlim[0]
        sba = self.ax.text(s=title, x=xcen, y=ycen, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, zorder=20, **text_kwargs)
        bb = sba.get_window_extent(renderer = self.matplotlib_renderer)
        self.__last_title_height += bb.height

    """
    from nkrpy .io import File; import numpy as np; from nkrpy.astro import SpectraPlot                                             
    file = File('UT180205_hops65.tellcor.fits'); h,d = file.data(); data =np.concatenate(d, axis=1)                                
    plot = SpectraPlot(xdata=data[0, :], xerror=np.zeros(data.shape[-1], dtype=float), ydata=data[1, :], yerror=data[2, :])                       
    plot.add_spectra()                                                   
    plot.set_ylim(1e-18, 5e-16)                               
    plot.set_xlim(1.2, 1.8)                                   
    plot.add_vertical_line(1.5, color='red')                  
    plot.add_horizontal_line(1.5e-16, color='red')               
    plot.refresh_data()                                        
    plot.add_gaussian(sigma=0.04, ampl=1, yfrac = 0.1, color='black')     
    plot.add_gaussian(sigma=0.02, ampl=1.1, yfrac = 0.05,  color='orange') 
    plot.add_gaussian(sigma=0.03, ampl=0.8, yfrac = 0.05,  color='green')   
    plot.add_gaussian(sigma=0.03, ampl=0.8, yfrac = 0.05,   location='upper-left', color='blue')          
    plot.set_ticks(xy='x', xminor_num = 0, xmajor_num=11, xfmt=(lambda val, tick: f'{val:.2f}'), rotation=45)                       
    plot.set_ticks(xy='y', ymajor_num=11, yminor_num = 0,  yfmt=(lambda val, tick: f'{val:0.1e}'))        
    plot.save('test.pdf')                    

    """
    def set_ticks(self, xy= 'auto', major_num = 11, minor_num = 4, fmt = (lambda val, tick: f'{val:.0f}'), **plt_kwargs):
        """Set the axis labels and ticks.

        Parameters
        ----------
        xy: str
            Default: auto. If set to aut will try to guess the axis based on the WCS. Else set 'x', 'y', 'z' to set the specific axis. If 'auto' `wcs` must be set
            Comma separated. For twin axis use twinx, twiny
        wcs: WCSClass
        """
        def set(ax, ma, mi, fm, **kwargs):
            ax.set_major_locator(plt.MaxNLocator(ma))
            ax.set_minor_locator(plt.MaxNLocator(ma * (mi + 1)))
            ax.set_major_formatter(plt.FuncFormatter(fm))
            ax.set_tick_params(**kwargs)
        xyaxis = []
        if isinstance(xy, str):
            xy = xy.lower().replace(' ', '')
            if xy == 'auto' or 'x' in xy.split(','):
                xyaxis += [self.ax.xaxis]
            if xy == 'auto' or 'y' in xy.split(','):
                xyaxis += [self.ax.yaxis]
            if 'ax_x_twin' in dir(self) and 'twinx' in xy.split(','):
                xy += [self.ax_x_twin]
            if 'ax_y_twin' in dir(self) and 'twiny' in xy.split(','):
                xy += [self.ax_y_twin]
        elif isinstance(xy, matplotlib.axes.Axes):
            xyaxis = [ax]
        if not isinstance(major_num, (list, tuple)):
            major_nums, minor_nums, fmts = map(lambda x: [x] * len(xyaxis), [major_num, minor_num, fmt])
        else:
            major_nums, minor_nums, fmts = major_num, minor_num, fmt
        for i in range(len(xyaxis)):
            xyax, major_num, minor_num, fmt = map(lambda x: x[i], [xyaxis, major_nums, minor_nums, fmts])
            set(ax=xyax, ma=major_num, mi=minor_num, fm=fmt, **plt_kwargs)


    def add_horizontal_line(self, y, *args, **kwargs):
        # transform WCS to pixel
        self.ax.axhline(y=y, *args, zorder=2, **kwargs)

    def add_vertical_line(self, x, *args, **kwargs):
        # transform WCS to pixel
        self.ax.axvline(x=x, *args, zorder=2, **kwargs)

    def add_secondary_x_axis(self, forward_func, reverse_func, location='top', new_label='', **kwargs):
        self.ax_x_twin = self.ax.secondary_xaxis(location, functions=(forward_func, reverse_func))
        ax = self.ax_x_twin
        ax.set_xlabel(new_label)

    def add_secondary_y_axis(self, forward_func, reverse_func, location='right', new_label='', **kwargs):
        self.ax_y_twin = self.ax.secondary_yaxis(location, functions=(forward_func, reverse_func))
        ax = self.ax_y_twin
        ax.set_ylabel(new_label)

    def add_gaussian(self, sigma: float, ampl: float = 1, yfrac: int = 0.1, location: str = 'lower-right', *plt_args, **plt_kwargs):
        # transform WCS to pixel
        fwhm = 2. * np.sqrt(2. * np.log(2.)) * sigma
        xd = 2 * fwhm
        df = xd / 100. 
        if 'right' in location:
            starting = self.xlim[-1] - xd - 2 * df
        elif 'left' in location:
            starting = self.xlim[0] + df
        x = np.arange(starting, starting + xd + df, df)
        xm = np.nanmedian(x)
        df = np.diff(self.ylim[::-1]) / 100.
        y = gauss(x, xm, sigma, 1)
        y *= yfrac * abs(np.diff(self.ylim))
        ymax = np.nanmax(y)
        if 'lower' in location:
            starting = self.ylim[0] + df
        elif 'upper' in location:
            starting = self.ylim[-1] - df - ymax
        self.plot(x, y + starting, *plt_args, **plt_kwargs)
        pass

    def add_spectra(self, xdata: np.ndarray, ydata: np.ndarray, *plt_args, plottype: str = 'all', xerror: np.ndarray = None, yerror: np.ndarray = None, yoffset: float = 0, xoffset: float = 0, **plt_kwargs):
        if xerror is None:
            xerror = np.zeros(xdata.shape)
        if yerror is None:
            yerror = np.zeros(ydata.shape)
        if plottype in ['errorbar', 'all', 'auto']:
            self.ax.errorbar(*plt_args, x=xdata + xoffset, y=ydata + yoffset, xerr=xerror, yerr=yerror, **plt_kwargs)
        if plottype in ['step', 'all', 'auto']:
            self.ax.step(*plt_args, xdata + xoffset, ydata + yoffset, **plt_kwargs)
        if plottype in ['scatter', 'all', 'auto']:
            self.ax.scatter(*plt_args, xdata + xoffset, ydata + yoffset, **plt_kwargs)
        if plottype in ['plot', 'all', 'auto']:
            self.ax.plot(*plt_args, xdata + xoffset, ydata + yoffset, **plt_kwargs)

    def add_transitions(self, name: str):
        if name == 'all':
            pass
        pass

    def set_aspect(self, height_width: float = 10):
        self.scale(height_width)

# end of file
