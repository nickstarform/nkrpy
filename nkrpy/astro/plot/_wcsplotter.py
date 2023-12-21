"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.colors as mplcolors
plt = mpl
import matplotlib
import math
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

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

# global attributes
__all__ = ['WCSImagePlot', 'auto_wcs_plot']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
#set_style()


def auto_wcs_plot(filename: str, savefile: str = ''):
    h, d = fits.read(filename)
    wcs = WCS(h)
    if len(np.squeeze(d).shape) > 2:
        m = np.sum(np.squeeze(d), axis=0)
    else:
        m = np.squeeze(d)
    m[m<=0] = 1e-10
    fig, ax = plt.subplots()
    print(filename)
    w = WCSImagePlot(wcs=wcs, data=np.sqrt(m), cmap='magma', ax=ax, fig=fig)
    #w.set_image(np.sqrt(m), cmap='magma', origin='lower')#, norm=mplcolors.LogNorm(vmin=np.nanmin(np.abs(d)), vmax=np.percentile(d, 99.5)), vmin=None, vmax=None)
    wcs = w.get_wcs()
    ra = wcs(wcs.axis1['axis'] / 2, 'wcs', wcs.axis1['dtype'])
    dec = wcs(wcs.axis2['axis'] / 2, 'wcs', wcs.axis2['dtype'])
    w.add_beam(edgecolor='white', lw=1, facecolor='white')
    scale = wcs(wcs.axis1['axis'] / 10., 'wcs', wcs.axis1['dtype']) - wcs(0, 'wcs', wcs.axis1['dtype'])
    w.add_scalebar(scale, wcs.axis1['unit'], facecolor='white', edgecolor='black', lw=1)
    #w.add_contour(start=5, stop=100, plot_negative=False)
    w.xlabel(wcs.axis1['dtype'])
    w.ylabel(wcs.axis2['dtype'])
    w.set_colorbar(colortitle=wcs.get_head('bunit'), pad=0)
    w.add_title(wcs.get_head('field'), fontsize='larger')
    w.set_ticks(xy='x', rotation=45)
    w.set_ticks(xy='y')
    #w.set_limits()
    if savefile == '':
        w.save('test.pdf', bbox_inches='tight')
    else:
        if not savefile.endswith('.pdf'):
            w.save(savefile+'.pdf', bbox_inches='tight', format='pdf')
        else:
            w.save(savefile, bbox_inches='tight', format='pdf')
    plt.close(w.fig)


class WCSImagePlot(Plot):
    """Generic WCS Image plotter.
    
    Methods
    -------
    add_beam: plot beams
    set_image: plot background (data, color)
    add_contour: plot contours  (contour params)
    add_outflow: plot outflows (vectors, xcen, ycen, angle, color, length)
    add_source: plot sources (marker styles)
    add_region: plot regions (ellipse, rectangle, xcen, ycen, major/xwidth, minor/ywidth, angle)
    add_scalebar: plot custom scalebars
    set_colorbar
    poststamp: ready the plot for compact plotting
    add_title
    set_ticks
    refresh_wcs: recreates the image wcs based off of new centers and widths
    refresh_data: reslices the data from the original using the loaded wcs
    """

    def __init__(self, filename: str = None, wcs: WCSClass = None, ra_cen: float = None, dec_cen: float = None, ra_width: float = None, dec_width: float = None, *args, data: np.ndarray = None, cmap = 'viridis', **pltcfg):
        # assert some needed params
        if filename is None:
            if wcs is None or data is None:
                print('Incorrect configuration. You need to specify both (wcs and data) or filename.')
        # initalize parent plot
        super(WCSImagePlot, self).__init__(*args, **pltcfg)
        self.matplotlib_renderer = self.fig.canvas.get_renderer()
        # define some tracking bariables
        self.__cmap = cmap
        self.__cbar_axes = None
        self.reset_axes()
        # lets niceify some things
        if filename is not None:
            h, data = fits.read(filename)
            wcs = WCS(h)
        if dec_cen is None:
            dec_cen = wcs(wcs.axis2['axis'] / 2., return_type='wcs', axis=wcs.axis2['dtype'])
        if ra_cen is None:
            ra_cen = wcs(wcs.axis1['axis'] / 2., return_type='wcs', axis=wcs.axis1['dtype'], declination_degrees=dec_cen)
        if ra_width is None:
            ra_width = wcs.axis1['axis'] * wcs.axis1['delt']
        if dec_width is None:
            dec_width = wcs.axis2['axis'] * wcs.axis2['delt']
        ra_width, dec_width = map(np.abs, [ra_width, dec_width])
        # generate new wcs
        self.__original_wcs = wcs
        self.refresh_wcs(ra_cen, dec_cen, ra_width, dec_width)
        # generate new data
        self.__original_data = data
        self.refresh_data()

    def set_limits(self, xy='xy', lower: float = None, upper: float = None):
        for axis in xy:
            if 'x' == axis:
                axis = self.__wcs.axis1
                fun = self.ax.set_xlim
            else:
                axis = self.__wcs.axis2
                fun = self.ax.set_ylim
            if lower is None:
                lower = 0
            if upper is None:
                upper = axis['axis']
            lower, upper = (lower, upper) if lower < upper else (upper, lower)
            fun(lower, upper)

    def get_owcs(self):
        return self.__original_wcs

    def get_wcs(self):
        return self.__wcs

    def refresh_wcs(self, ra_cen, dec_cen, ra_width, dec_width):
        wcs = WCS(self.__original_wcs)
        nwcs = self.__create_image_wcs(wcs, ra_cen, dec_cen, ra_width, dec_width)
        self.__wcs = nwcs

    def __create_image_wcs(self, wcs:WCSClass, ra_cen, dec_cen, ra_width, dec_width):
        ra_width, dec_width = map(abs, [ra_width, dec_width])
        image_wcs = wcs
        # calculate widths
        image_wcs.center_axis_wcs(axis=wcs.axis2['dtype'], val=dec_cen, width=dec_width)
        image_wcs.center_axis_wcs(axis=wcs.axis1['dtype'], val=ra_cen, width=ra_width, declination_degrees=dec_cen)
        self.__wcs = image_wcs
        return image_wcs

    def refresh_data(self, data: np.ndarray = None, owcs: WCSClass = None):
        if owcs is None:
            owcs = self.__original_wcs
        if data is None:
            data = self.__original_data
        wcs = self.__wcs
        # get limits for new wcs
        dec_upper = wcs(wcs.axis2['axis']-1, 'wcs', wcs.axis2['dtype'])
        dec_lower = wcs(0, 'wcs', wcs.axis2['dtype'])
        ra_upper = wcs(wcs.axis1['axis']-1, 'wcs', wcs.axis1['dtype'], declination_degrees=np.mean([dec_upper, dec_lower]))
        ra_lower = wcs(0, 'wcs', wcs.axis1['dtype'], declination_degrees=np.mean([dec_upper, dec_lower]))
        # now find these new limits within the old wcs
        dec_upper_in_old = owcs(dec_upper, 'pix', owcs.axis2['dtype'])
        dec_lower_in_old = owcs(dec_lower, 'pix', owcs.axis2['dtype'])
        ra_upper_in_old = owcs(ra_upper, 'pix', owcs.axis1['dtype'], declination_degrees=np.mean([dec_upper, dec_lower]))
        ra_lower_in_old = owcs(ra_lower, 'pix', owcs.axis1['dtype'], declination_degrees=np.mean([dec_upper, dec_lower]))
        # sort to be safe
        ra_upper_in_old, ra_lower_in_old = sorted([ra_upper_in_old, ra_lower_in_old])
        dec_lower_in_old, dec_upper_in_old = sorted([dec_lower_in_old, dec_upper_in_old])
        ra_upper_in_old, ra_lower_in_old, dec_lower_in_old, dec_upper_in_old= map(lambda x: int(round(abs(x),0)), [ra_upper_in_old, ra_lower_in_old, dec_lower_in_old, dec_upper_in_old])
        #print('bounds:',ra_upper_in_old, ra_lower_in_old, dec_lower_in_old, dec_upper_in_old, data.shape)
        #print('wcs:', wcs.axis2['axis'], wcs.axis1['axis'])
        # now check if padding is needed
        if ra_upper_in_old < 0:
           # pad left side
           #print('pad left')
           ra_upper_in_old = int(round(abs(ra_upper_in_old),0))
           data = np.roll(data, shift=ra_upper_in_old, axis=1)
           data[:, :ra_upper_in_old+1] = np.nan
           ra_lower_in_old += ra_upper_in_old
           ra_upper_in_old = 0
        if dec_lower_in_old < 0:
           # pad bottom side
           #print('pad bottom')
           dec_lower_in_old = int(round(abs(dec_lower_in_old),0))
           data = np.roll(data, shift=dec_lower_in_old, axis=0)
           data[:dec_lower_in_old+1, :] = np.nan
           dec_upper_in_old += dec_lower_in_old
           dec_lower_in_old = 0
        if ra_lower_in_old >= owcs.axis1['axis']:
           # pad right side
           #print('pad right')
           ra_lower_in_old = int(round(abs(ra_lower_in_old),0))
           data = np.roll(data, shift=-ra_lower_in_old, axis=1)
           data[:, -ra_lower_in_old:] = np.nan
           ra_lower_in_old = owcs.axis1['axis']
        if dec_upper_in_old >= owcs.axis2['axis']:
           # pad up side
           #print('pad up')
           dec_upper_in_old = int(round(abs(dec_upper_in_old),0))
           data = np.roll(data, shift=-dec_upper_in_old, axis=0)
           data[-dec_upper_in_old:, :] = np.nan
           dec_upper_in_old = owcs.axis2['axis']
        ra_upper_in_old, ra_lower_in_old, dec_lower_in_old, dec_upper_in_old= map(lambda x: int(round(abs(x),0)), [ra_upper_in_old, ra_lower_in_old, dec_lower_in_old, dec_upper_in_old])
        sliced_data = data[dec_lower_in_old:dec_upper_in_old + 1, ra_upper_in_old:ra_lower_in_old + 1]

        if sliced_data.shape != (int(wcs.axis2['axis']), int(wcs.axis1['axis'])):
            print(f'Error in formating data: Output data {sliced_data.shape} != ({int(wcs.axis2["axis"])}, {int(wcs.axis1["axis"])})')
        self.__data = sliced_data
        return sliced_data

    def poststamp(self, restrict: bool = False):
        """Format the plot for tight layouts."""
        self.ax.grid(False)
        self.ax.set_axis_on()
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.remove_colorbars()
        if restrict:
            self.reset_axes()

    def reset_axes(self):
        self.remove_colorbars()
        self.ax.cla()
        self.__contours = []
        self.__scalebars = []
        self.__last_scalebar_height = 0
        self.__last_beam_height = 0
        self.__last_title_height = 0
        self.__beams = []
        self.__slicing_params = None
        self.__cbar_axes = None

    def remove_colorbars(self):
        if self.__cbar_axes is None:
            return
        for axis in self.__cbar_axes:
            axis.cla()
            axis.set_axis_off()
        self.__cbar_axes = None

    def add_title(self, title: str, xcen: float = 0.5, ycen: float = None, verticalalignment: str = 'top', horizontalalignment: str = 'center', **text_kwargs):
        if ycen is None:
            ycen = 1.
            if self.__last_title_height == 0:
                ycen = self.__wcs.axis2['axis'] * ycen
            else:
                ycen = self.__wcs.axis2['axis'] - self.__last_title_height
        else:
            ycen = self.__wcs.axis2['axis'] * ycen
        xcen *= self.__wcs.axis1['axis']
        sba = self.ax.text(s=title, x=xcen, y=ycen, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, zorder=20, **text_kwargs)
        bb = sba.get_window_extent(renderer = self.matplotlib_renderer)
        bb = bb.transformed(self.ax.transData)
        self.__last_title_height = self.__wcs.axis2['axis'] - (ycen - bb.height)

    def set_colorbar(self, colortitle: str = '', **plt_kwargs):
        if 'cax' in plt_kwargs:
            cax = plt_kwargs['cax']
            plt_kwargs = dict(plt_kwargs)
            del plt_kwargs['cax']
        else:
            if self.__cbar_axes is None:
                self.add_colorbar(colortitle, **plt_kwargs)
                return
            else:
                cax = self.__cbar_axes[-1]
        cb = self.fig.colorbar(label=colortitle, mappable=ScalarMappable(cmap=self.__cmap), cax=cax, **plt_kwargs)

    def add_colorbar(self, colortitle: str = '', **plt_kwargs):
        if 'ax' in plt_kwargs:
            ax = plt_kwargs['ax']
            plt_kwargs = dict(plt_kwargs)
            del plt_kwargs['ax']
        else:
            ax = self.ax

        if 'cax' in plt_kwargs:
            cax = plt_kwargs['cax']
            plt_kwargs = dict(plt_kwargs)
            del plt_kwargs['cax']
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if self.__cbar_axes is None:
                self.__cbar_axes = [cax]
            else:
                self.__cbar_axes.append(cax)

        cmap = ScalarMappable(cmap=self.__cmap)
        cmap.set_array([])
        cb = self.fig.colorbar(label=colortitle, mappable=cmap, cax=cax, **plt_kwargs)

    def set_ticks(self, xy, wcs, number: int = 7, skipfirst: bool = True, skiplast: bool=False, **plt_kwargs):
        """Set the axis labels and ticks.

        Parameters
        ----------
        xy: str
            Default: auto. If set to aut will try to guess the axis based on the WCS. Else set 'x', 'y', 'z' to set the specific axis. If 'auto' `wcs` must be set
        wcs: WCSClass
        """
        def fmt(label, ra=True):
            nlabel = label.tolist()
            starting_icrs = degrees2icrs(label[0])
            starting_icrs[-1] = round(starting_icrs[-1], 2)
            starting_format = ':'.join(map(str, starting_icrs))
            nlabel = [starting_format]
            last = starting_icrs
            for i in range(1, len(label)):
                current = degrees2icrs(label[i])
                newlabel = ''
                if abs((current[0] - last[0])) >= 1:
                    current[-1] = round(current[-1], 3)
                    newlabel = ':'.join(map(str, current))
                elif abs((current[1] - last[1])) >= 1:
                    current[-1] = round(current[-1], 3)
                    newlabel = ':'.join(map(str, current[1:]))
                else:
                    current[-1] = round(current[-1], 3)
                    newlabel = f'{current[-1]}'
                nlabel.append(newlabel)
                last = current
            return nlabel

        # generate both based on wcs
        xaxis = wcs.axis1['dtype']
        yaxis = wcs.axis2['dtype']
        xticks_icrs = wcs.array(size=number, return_type='wcs', axis=xaxis)[::-1] / 15.
        yticks_icrs = wcs.array(size=number, return_type='wcs', axis=yaxis)
        if skiplast:
            xticks_icrs = xticks_icrs[:-1]
            yticks_icrs = yticks_icrs[:-1]
        if skipfirst:
            xticks_icrs = xticks_icrs[1:]
            yticks_icrs = yticks_icrs[1:]

        xlabels = fmt(xticks_icrs)
        ylabels = fmt(yticks_icrs)
        xticks_pix = wcs(xticks_icrs*15, return_type='pix', axis=xaxis, declination_degrees=yticks_icrs)
        yticks_pix = wcs(yticks_icrs, return_type='pix', axis=yaxis)
        xticks_pix = np.around(xticks_pix,4)
        yticks_pix = np.around(yticks_pix,4)
        if xy == 'auto':
            xy = 'xy'
        #from IPython import embed; embed()
        if 'x' in xy:
            self.ax.get_xaxis().set_ticks(ticks=xticks_pix)
            self.ax.get_xaxis().set_ticklabels(ticklabels=xlabels, **plt_kwargs)
        if 'y' in xy:
            self.ax.get_yaxis().set_ticks(ticks=yticks_pix)
            self.ax.get_yaxis().set_ticklabels(ticklabels=ylabels, **plt_kwargs)


    def add_region(self, xcen: float, ycen: float, angle: float, x_width: float, y_width: float, rectangle: bool = True, **plot_kwargs):
        # transform WCS to pixel
        wcs = self.__wcs
        ycen = wcs(ycen, return_type='pix', axis=wcs.axis2['dtype'])
        xcen = wcs(xcen, return_type='pix', axis=wcs.axis1['dtype'], declination_degrees=ycen)
        x_width = abs(x_width / (wcs.axis1['delt'] / np.cos(ycen * np.pi / 180)))
        y_width = abs(y_width / wcs.axis2['delt'])
        if rectangle:
            self.rectangle(xcen=xcen, ycen=ycen, angle=angle,xwidth=x_width, ywidth=y_width, zorder=15, **plot_kwargs)
            return
        major, minor = x_width, y_width
        if major is None:
            major, minor = wcs.get_head('bmaj'), wcs.get_head('bmin') # in deg
            angle = wcs.get_head('bpa') # in deg
            major, minor = map(lambda x: abs(x / wcs.axis1['delt']), [major, minor])
        if minor is None or minor == 0:
            minor = major
        if angle is None:
            angle = 0
        # east of north units
        angle += 90
        major, minor if major > minor else minor, major
        ellipse = self.ellipse(xcen, ycen, major=major, minor=minor, angle=angle, zorder=15, **plot_kwargs)

    def add_line(self, x, y, *args, **kwargs):
        # transform WCS to pixel
        ycen_pix = self.__wcs(y, return_type='pix', axis=self.__wcs.axis2['dtype'])
        xcen_pix = self.__wcs(x, return_type='pix', axis=self.__wcs.axis1['dtype'], declination_degrees=ycen)
        self.ax.plot(xcen_pix, ycen_pix, *args, zorder=2, **kwargs)

    def add_source(self, xcen, ycen, *args, **kwargs):
        # transform WCS to pixel
        ycen_pix = self.__wcs(ycen, return_type='pix', axis=self.__wcs.axis2['dtype'])
        xcen_pix = self.__wcs(xcen, return_type='pix', axis=self.__wcs.axis1['dtype'], declination_degrees=ycen)
        self.ax.scatter(xcen_pix, ycen_pix, *args, zorder=15, **kwargs)

    def add_outflow(self, xcen, ycen, length, *args, **kwargs):
        """Alias for vector."""
        # transform WCS to pixel
        ycen_pix = self.__wcs(ycen, return_type='pix', axis=self.__wcs.axis2['dtype'])
        xcen_pix = self.__wcs(xcen, return_type='pix', axis=self.__wcs.axis1['dtype'], declination_degrees=ycen)
        length = length / self.__wcs.axis1['delt']
        self.vector(xcen_pix, ycen_pix, *args, length = length, zorder=15, **kwargs)

    def add_disk(self, xcen, ycen, length, *args, **kwargs):
        """Alias for vector."""
        # transform WCS to pixel
        xcen = self.__wcs(xcen, return_type='pix', axis=self.__wcs.axis1['dtype'])
        ycen = self.__wcs(ycen, return_type='pix', axis=self.__wcs.axis2['dtype'])
        length = length / self.__wcs.axis1['delt']
        self.vector(xcen, ycen, *args, length = length, zorder=15, **kwargs)

    def add_contour(self, *args,  data: np.ndarray = None,start: int = 3, stop: int = 10, interval: int = 1, rms: float = None, plot_negative: bool = True, **kwargs):
        """Plot contour based on the data."""
        if data is None:
            if len(args) > 0:
                data = args[0]
                args = list(args)
                del args[0]
            else:
                data = self.__data
        start, stop if start < stop else stop, start
        if rms is None:
            rms = np.std(data[data > 0])
        if rms < 0:
            rms = np.abs(rms)
        levels = np.arange(start, stop + interval, interval).astype(np.float64)
        levels *=  rms
        if plot_negative:
            levels = np.concatenate([-1. * levels, levels])
        levels = np.sort(np.ravel(levels))
        cont = self.ax.contour(data, *args, levels=levels, zorder=2,**kwargs)
        self.__contours.append(cont)
        pass

    def add_scalebar(self, length: float, scaletype: str, omit_text: bool=False, scalebar_inline:bool = True, text_kwargs: dict = {}, **bar_kwargs: dict):
        length = abs(length)
        units = Unit(vals=length, baseunit=scaletype, convunit=self.__wcs.axis1['unit'])
        wcs = self.__wcs
        scale_pltsize = abs(units.get_vals()/ wcs.axis1['delt'])
        if scale_pltsize > (wcs.axis1['axis'] / 2.):
            print('Your scalebar size is larger than 50\% of the axis.')
        cscalebar = len(self.__scalebars) + 1
        xstart = wcs.axis1['axis'] / 50.
        if len(str(length)) > 5:
            length = (f'{length: 0.2e}')
        height = 0
        sba = ''
        if not omit_text:
            sba = self.ax.text(s=f'{length} {scaletype}', x=xstart, y=self.__last_scalebar_height, horizontalalignment='left', verticalalignment='bottom', zorder=20, **text_kwargs)
            bb = sba.get_window_extent(renderer = self.matplotlib_renderer).transformed(self.ax.transData)
            if not scalebar_inline:
                self.__last_scalebar_height = bb.y1
                height = abs(bb.height)
        if height == 0:
            height = wcs.axis1['axis']/50
        sb = self.ax.fill_between(x=[xstart, xstart + scale_pltsize], y1=self.__last_scalebar_height, y2=self.__last_scalebar_height + height, zorder=19, **bar_kwargs)
        self.__last_scalebar_height += height
        self.__scalebars.append({'scalebar': sb, 'annotation': sba})
        #from IPython import embed; embed()

    def set_image(self, *args, data: np.ndarray = None, scale_fn = None, fill: float = None, **kwargs):
        """Plot the background image."""

        if data is None:
            if len(args) > 0:
                data = args[0]
                args = list(args)
                del args[0]
            else:
                data = self.__data
        if scale_fn is not None:
            data = scale_fn(data)
        if 'vmax' not in kwargs or 'vmin' not in kwargs:
            """ Estimate the background level and rms. """
            good = ~np.isnan(data)
            if np.any(good):
                omin = np.percentile(data[good], 65)
                # remove lowest 2% to get rid of any outliers
                omax = np.percentile(data[good][data[good] > omin], 99.999)
            else:
                omin, omax = None, None
        kwargs = dict(kwargs)
        if 'vmax' not in kwargs:
            vmax = omax
        else:
            vmax = kwargs['vmax']
            del kwargs['vmax']
        if 'vmin' not in kwargs:
            vmin = omin
        else:
            vmin = kwargs['vmin']
            del kwargs['vmin']

        if 'origin' not in kwargs:
            kwargs['origin'] = 'lower'
        good = ~np.isnan(data)
        data[~good] = vmin if fill is None else fill

        image = self.ax.imshow(data, *args, vmin=vmin, vmax=vmax, **kwargs, zorder=1)
        return image

    def add_beam(self, major: float = None, minor: float = 0, beamunits: str = 'arcsec', angle: float = 0, **plt_kwargs):
        """
        Parameters
        ----------
        major, minor : float
            major minor axis in deg (total length not half)
        angle: float
            angle in degrees, east of north
        wcs: WCSClass
            wcs of the object to use instead
        """
        wcs = self.__wcs
        if major is None:
            major, minor, angle = wcs.get_beam() # in radians
            beamunits = wcs.axis1['unit']
        if minor is None or minor == 0:
            minor = major
        if angle is None:
            angle = 0
        # converted to data units
        major, minor = Unit(vals=[major, minor], baseunit=beamunits, convunit=wcs.axis1['unit']).get_vals()
        # convert to pixels
        major, minor = map(lambda x: abs(x / wcs.axis1['delt']), [major, minor])
        # east of north units
        radians = np.pi/2 + angle * np.pi / 180.
        angle = 180 / np.pi * radians
        major, minor = (major, minor) if major > minor else (minor, major)
        # spacing of beams. There is a base spacing of axis_width / 100.. Then additional spacing of the ellipse apoapsis
        xspacing = wcs.axis1['axis'] / 100.
        yspacing = wcs.axis2['axis'] / 100.
        xoff = abs(major * np.sin(radians)) / 2.
        xoff += abs(minor * np.cos(radians)) / 2.
        yoff = abs(major * np.cos(radians)) / 2.
        yoff += abs(minor * np.sin(radians)) / 2.
        xcen = wcs.axis1['axis']
        xcen -= xoff
        xcen -= xspacing
        ycen = self.__last_beam_height if self.__last_beam_height != 0 else yspacing
        ycen += yoff
        el = self.ellipse(xcen=xcen, ycen=ycen, major=major, minor=minor, angle=angle, zorder=20, **plt_kwargs)
        bb = el.get_window_extent(renderer = self.matplotlib_renderer).transformed(self.ax.transData)
        self.__last_beam_height = ycen + yoff + yspacing
        self.__beams.append(el)
