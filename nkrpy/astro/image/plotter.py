"""."""
# flake8: noqa
# cython modules

# internal modules
import importlib
import os
from sys import version
import re
import pickle
import glob
import argparse

# external modules
import aplpy
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib
from matplotlib.cm import ScalarMappable

# relative modules
from ...publication.plots import set_style, ppi
from ...misc.functions import typecheck
from ...io import Config
from ..._unit import Unit as unit
from ...math import _convert as nkrpy__convert
icrs2deg, deg2icrs = nkrpy__convert.icrs2degrees, nkrpy__convert.degrees2icrs
from ...misc import constants
from ...publication.cmaps import mainColorMap, mapper
from ._config import default
from .._plot import Plot
from ..._types import WCSClass
from ...math import ellipse_distance
from ...math import rms as math_rms

# global attributes
__all__ = ('Plotter', 'WCSImagePlot')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
assert aplpy.__version__ == '1.1.1'
set_style()


class WCSImagePlot(Plot):
    """Generic Image plotter.
    
    Methods
    -------
    beam: plot beams
    image: plot background (data, color)
    contour: plot contours  (contour params)
    outflow: plot outflows (vectors, xcen, ycen, angle, color, length)
    sources: plot sources (marker styles)
    regions: plot regions (ellipse, rectangle, xcen, ycen, major/xwidth, minor/ywidth, angle)
    scalebar: plot custom scalebars
    """
    SCALEBAR_PLTSCALING = 0.1

    def __init__(self, ra_cen: float, dec_cen: float, ra_width: float, dec_width: float, width_unit: str, wcs: WCSClass, *args, cmap = 'viridis', **pltcfg):
        self.__cmap = cmap
        super(ImagePlot, self).__init__(*args, **pltcfg)
        self.__beams = {0.0: {'x': 0., 'y': 0., 'major': 0, 'minor': 0, 'angle': 0}} # [dist] x, y, dist, obj
        self.pltcfg = pltcfg
        self.__contours = []
        self.__scalebars = []
        self.___original_wcs = wcs

        image_wcs = wcs.deepcopy()
        image_wcs.drop_axis('stokes')
        # shift axis to new center and slice
        raval = racen - wcs(wcs.axis1['size'] / 2., 'wcs', wcs.axis1['name'])
        decval = deccen - wcs(wcs.axis2['size'] / 2., 'wcs', wcs.axis2['name'])
        image_wcs.shift_axis(axis=wcs.axis1['name'], unit='wcs', val=raval)
        image_wcs.shift_axis(axis=wcs.axis2['name'], unit='wcs', val=decval)
        # new total length
        rapixwidth = 2. * Unit(vals=ra_width, baseunit=width_unit, convunit=image_wcs.axis1['unit']) / image_wcs.axis1['delt']
        decpixwidth = 2. * Unit(vals=dec_width, baseunit=width_unit, convunit=image_wcs.axis1['unit']) / image_wcs.axis2['delt']
        image_wcs.get(image_wcs.axis1['name'])['axis'] = rapixwidth
        image_wcs.get(image_wcs.axis2['name'])['axis'] = decpixwidth
        self.__wcs = image_wcs

    def __set_color_axis(self, colortitle: str = '', **plt_kwargs):
        self.__cbar = self.__fig.colorbar(mappable=ScalarMappable(cmap=self.__cmap), cax = self.__ax, ax = self.__ax, **plt_kwargs)

    def __set_xy_axis_ticks(self, xy: str = 'auto', xlocations: list = [], xlabels: list = [], ylocations: list = [], ylabels: list = [], xtitle: str = '', ytitle: str = '', wcs: WCSClass = None, **plt_kwargs):
        """Set the axis labels and ticks.

        Parameters
        ----------
        xy: str
            Default: auto. If set to aut will try to guess the axis based on the WCS. Else set 'x', 'y', 'z' to set the specific axis. If 'auto' `wcs` must be set
        wcs: WCSClass
        """
        def fmt(label, axis):
            starting = np.min(label)
            startingidx = label.index(starting)
            label = [label, label]
            starting = ':'.join(map(str, degrees2icrs(starting)))
            label[1][startingidx] = starting
            for i, v in enumerate(label[0]):
                icrs = degrees2icrs(v if 'dec' in axis else v / 15.)
                icrs[-1] = round(icrs[-1], 2)
                if np.abs(v - label[0][startingidx]) > 1.:
                    icrs = ':'.join(map(str, icrs))
                    label[1][i] = icrs
                elif np.abs(v - label[0][startingidx]) * 60 > 1.:
                    icrs = ':'.join(map(str, icrs[1:]))
                    label[1][i] = icrs
                elif np.abs(v - label[0][startingidx]) * 3600 > 1.:
                    icrs = f'{icrs[-1]}'
                    label[1][i] = icrs
            label.sort(key= lambda x: x[0])
            return label[1]

        if wcs is not None:
            # generate both based on wcs
            xaxis = wcs.axis1['type']
            yaxis = wcs.axis2['type']
            xticks = wcs.array(11, return_type='wcs', axis=xaxis)
            yticks = wcs.array(11, return_type='wcs', axis=yaxis)
            xlabels = fmt(xticks, xaxis)
            ylabels = fmt(yticks, yaxis)
            xy = 'xy'
        if 'x' in xy:
            self.__ax.get_xaxis().ticks(ticks=xticks, labels=xlabels, **plt_kwargs)
        if 'y' in xy:
            self.__ax.get_yaxis().ticks(ticks=yticks, labels=ylabels, **plt_kwargs)

    def region(self, xcen: float, ycen: float, angle: float, x_width: float, y_width: float, rectangle: bool = True, **kwargs):
        radians = angle * np.pi / 180.
        if rectangle:
            self.rectangle(xcen=xcen, ycen=ycen, angle=angle,xwidth=x_width,ywidth=y_width, **kwargs)
        else:
            major, minor = x_width, y_width if x_width > y_width else y_width, x_width
            ellipse = Ellipse(xy=(xcen, ycen), width=major, height=minor, angle=angle, **kwargs)
            self.__ax.add_patch(ellipse)

    def source(self, *args, **kwargs):
        self.__ax.scatter(*args, zorder=15, **kwargs)

    def outflow(self, *args, **kwargs):
        """Alias for vector."""
        self.vector(*args, zorder=10, **kwargs)

    def contour(self, data: np.ndarray,*args,  start: int = 3, stop: int = 10, interval: int = 1, rms: float = None, plot_negative: bool = True, **kwargs):
        """Plot contour based on the data."""
        start, stop if start < stop else stop, start
        if rms is None:
            rms = np.std(data[data > 0])
        if rms < 0:
            rms = np.abs(rms)
        levels = np.arange(start, stop + interval, interval) * rms
        if plot_negative:
            levels = np.concatenate([-1. * levels, levels])
        levels = np.sort(np.ravel(levels))
        self.__contours.append(cont)
        cont = self.__ax.contour(data, *args, levels=levels, zorder=2,**kwargs)
        pass

    def scalebar(self, length: float, scaletype: str, wcsaxis, **plt_kwargs):
        units = Unit(vals=length, baseunit=scaletype, convunit=self.__wcs.axis1['unit'])
        scale_pltsize = units.get_vals()/ (wcs(wcs.get(wcsaxis)['axis'], 'wcs', wcs.get(wcsaxis)['type']) - wcs(0, 'wcs', wcs.get(wcsaxis)['type']))
        scale_pltsize = abs(scale_pltsize)
        if scale_pltsize > 0.5:
            print('Your scalebar size is larger than 50\%.')
        cscalebar = len(self.__scalebars)
        sb = self.__ax.fill_between(x=[0.05, 0.05 + scale_pltsize], y1=0.1 + self.SCALEBAR_PLTSCALING * cscalebar, y2=0.15 + self.SCALEBAR_PLTSCALING * cscalebar, hatch='/', transform=self.__ax.transAxes)
        sba = self.__ax.text(text=f'{length} {scaletype}', x=0.05 + scale_pltsize / 2., y=0.1 + self.SCALEBAR_PLTSCALING * cscalebar, xycoords='figure fraction', horizontalalignment='center', verticalalignmnet='center', **plt_kwargs)
        self.__scalebars.append({'scalebar': sb, 'annotation': sba})

    def image(self, data: np.ndarray, *args, **kwargs):
        """Plot the background image."""
        self.__ax.imshow(data, *args, origin='lower-left', **kwargs, zorder=1)

    def beam(self, major: float = None, minor: float = None, angle: float = None, label: str = None, wcs: WCSClass = None, facecolor: str = 'white'):
        """
        Parameters
        ----------
        major, minor : float
            major minor axis in deg
        angle: float
            angle in degrees
        wcs: WCSClass
            wcs of the object to use instead
        """
        if label is None:
            label = f'{len(self.__beams.keys())}'
        if wcs is not None:
            major, minor = wcs.get_head('bmaj'), wcs.get_head('bmin') # in deg
            angle = wcs.get_head('bpa') # in deg

        radians = angle * np.pi / 180.
        major, minor if major > minor else minor, major
        # need to get display width instead
        width_points = self.data_to_axis.transform((-major, 0), (major, 0))
        width_points = np.abs(width_points[1][0] - width_points[0][1]) / 2.
        major_pixel = width_points
        minor_pixel = width_points * minor / major
        xcen = 1. - (major_pixel * np.cos(radians)  + 0.01)
        ycen = major_pixel * np.sin(radians) + 0.01

        # for first beam
        if len(self.__beams.keys()) == 0:
            center = (np.sin(radians) * major, np.cos(radians) * major)
            dist = ellipse_distance(major=major_pixel, minor=minor_pixel, ellipse_pa=angle, vector_pa = 45., vector_ref=False)[0] + (center[0] ** 2 + center[1] ** 2) ** 0.5
            self.__beam[dist] = {
                'x': center[0],
                'y': center[1],
                'major': major_pixel,
                'minor': minor_pixel,
                'angle': angle,
            }
        else: # subsequent beams
            max_idx = max(self.__beams.keys())
            el = ellipse_distance(major=major_pixel, minor=minor_pixel, ellipse_pa=angle, vector_pa = 45. + 270., vector_ref=False)[0]
            eu = ellipse_distance(major=major_pixel, minor=minor_pixel, ellipse_pa=angle, vector_pa = 45., vector_ref=False)[0]
            d = max_idx - el
            dist = max_idx - el + eu
            self.__beam[dist] = {
                'x': d * np.sin(radians),
                'y': d * np.cos(radians),
                'major': major_pixel,
                'minor': minor_pixel,
                'angle': angle,
            }
        # plot beam of self.__beam[dist]
        xcen, ycen, major, minor, angle = self.__beam[dist].values()
        self.ellipse(xcen=self.__beam[dist]['x'], ycen=self.__beam[dist]['y'], major=self.__beam[dist]['major'], minor=self.__beam[dist]['minor'], angle=self.__beam[dist]['angle'], facecolor=facecolor, hatch='/', zorder=20)


class Plotter():
    """
    """
    __default_config = default

    def __init__(self, cfg: dict, fig = None):
        self.config = Config(default=self.__default_config, target=cfg, strict=False)
        if fig is None:
            self.fig = mpl.figure(figsize=self.config['figuresize'])
        else:
            self.fig = fig
        self.fontspacereq = (self.config['textsize'] / ppi) / self.config['figuresize'][-1]  # in % of y axis
        self.mols = self.__gather_molecules()
        self.markers = self.__gather_markers()
        self.outflows = self.__gather_outflows()
        self.labels = []
        self.__tmp_plot = mpl.figure(figsize=(1, 1))
        if 'ds9' in self.config['colormap']:
            mainColorMap('ds9')

    def __gather_outflows(self):
        """Gather outflows from config.

        Return
        ------
        dict
            This dict is {#: [keys]} denoting each of the outflows to plot and their corresponding keys.
        """
        keys = [k.strip('_ra').strip('outflow') for k in self.config if k.startswith('outflow') and not k.startswith('outflowX_') and k.endswith('_ra')]
        keys = set(keys)
        _compare_against = [k.strip('outflowX_') for k in self.config if k.startswith('outflowX_')]
        if not len(keys):
            return {}
        if self.config['debug']:
            print(f'Loaded {len(keys)} outflows')
        return keys

    def __gather_molecules(self):
        """Gather molecules from config.

        Return
        ------
        dict
            This dict is {#: [keys]} denoting each of the molecules to plot and their corresponding keys.
        """
        keys = [k for k in self.config if k.startswith('molecule') and not k.startswith('moleculeX_')]
        _compare_against = [k.strip('moleculeX_') for k in self.config if k.startswith('moleculeX_')]
        if not len(keys):
            return {}
        ret = {}
        for k in keys:
            mol = k.strip('molecule').split('_')[0]
            for i in _compare_against:
                mol = mol.strip(i)
            if mol not in ret:
                ret[mol] = []
            ret[mol].append(k)

        print(f'Loaded {len(ret)} molecules')
        return ret

    def __gather_markers(self):
        """Gather markers from config.

        Return
        ------
        dict
            This dict is {#: [keys]} denoting each of the markers to plot and their corresponding keys.
        """
        keys = [k for k in self.config if k.startswith('marker') and not k.startswith('markerX_')]
        _compare_against = [k.strip('markerX_') for k in self.config if k.startswith('markerX_')]
        if not len(keys):
            return {}
        ret = {}
        for k in keys:
            mol = k.strip('marker').split('_')[0]
            if mol not in ret:
                ret[mol] = []
            ret[mol].append(k)
        print(f'Loaded {len(ret)} markers')
        return ret

    def drawBackground(self):
        """
        """
        if not self.config['backgroundfile']:
            return

        self.__gc1 = aplpy.FITSFigure(self.config['backgroundfile'], figure=self.fig, dimensions=(0, 1))
        #print(dir(self.__gc1))
        self.__data = self.__gc1._data
        vmin = np.nanmin(np.abs(self.__data.flatten()))
        vmax = np.nanmax(np.abs(self.__data.flatten()))
        vrms = math_rms(self.__data.flatten())
        vstd = np.nanstd(self.__data)
        #from IPython import embed; embed()
        self.config['backgroundplotvmin'] = float(vmin / vstd)  if self.config['backgroundplotvmin'] is None else self.config['backgroundplotvmin']
        self.config['backgroundplotvmax'] = float(vmax / vstd) if self.config['backgroundplotvmax'] is None else self.config['backgroundplotvmax']
        self.config['backgroundplotrms'] = vstd if self.config['backgroundplotrms'] is None else self.config['backgroundplotrms']
        try:
            _cmap = mpl.get_cmap(self.config['colormap'])
        except:
            _cmap = mpl.get_cmap('twilight')
        if self.config['colormapinvert']:
            def invert(cmap):
                return mapper(lambda x: 1. - x, cmap)
            _cmap = invert(_cmap)
        cmap = _cmap
        vmin = vrms

        if self.config['backgroundplotcolor']:
            self.__gc1.show_colorscale(vmin=vmin, vmax=vmax,
                                       stretch=self.config['fluxstretch'], cmap=cmap)
        else:
            self.__gc1.show_grayscale(
                vmin=vmin, vmax=vmax,
                stretch=self.config['fluxstretch'],
                invert=True)
        if not self.config['backgroundcontour']:
            return
        color = self.config['backgroundcontourcolor'][0],
        file = self.config['backgroundfile'],
        negative = self.config['backgroundnegcontour'],
        start = self.config['backgroundcontourstart'],
        interval = self.config['backgroundcontourint'],
        end = self.config['backgroundcontourend'],
        rms = self.config['backgroundcontourrms'],
        manual = self.config['backgroundcontourmanual']
        #print(color, file, negative, start, interval, end, rms, manual)
        self.__setup_contours(color, file, negative, start, interval, end, rms, manual)

    def addMoleculeContours(self):
        for num in self.mols:
            print(num)
            if f'molecule{num}_file' in self.config and self.config[f'molecule{num}_file']:
                continue
                file = self.config[f'molecule{num}_file'][0]
                color = self.config[f'molecule{num}_plotcolor'][0]
                cmap = self.config[f'molecule{num}_plotcolorcmap'][0]
                manual = self.config[f'molecule{num}{col}_contourmanual'][0]
                rms = self.config[f'molecule{num}{col}_rms'][0]
                # setup single contour plot
                self.__setup_contours(color, file, negative, start, interval, end, rms, manual)
            else:
                # setup red blue plot
                for col in ['red', 'blue']:
                    if f'molecule{num}_{col}file' not in self.config:
                        continue
                    name = self.config[f'molecule{num}_{col}name'][0]
                    manual = self.config[f'molecule{num}_{col}contourmanual'][0]
                    file = self.config[f'molecule{num}_{col}file'][0]
                    negative = self.config[f'molecule{num}_{col}neg'][0]
                    start = self.config[f'molecule{num}_{col}start'][0]
                    interval = self.config[f'molecule{num}_{col}int'][0]
                    end = self.config[f'molecule{num}_{col}end'][0]
                    rms = self.config[f'molecule{num}_{col}rms'][0]
                    color = self.config[f'molecule{num}_{col}contourcolor'][0]
                    negative = self.config[f'molecule{num}_{col}neg']
                    self.__setup_contours(color, file, negative, start, interval, end, rms, manual)


    def __setup_contours(self, color, imagename, negative, start, interval, end, rms, contourmanual=[]):
        """
        """
        print(imagename)
        linesize = self.config['contourlinesize']
        if len(contourmanual) > 0:
            contours = np.array(contourmanual)
        else:
            contours = self.__gather_contours(negative, start, interval, end, rms)
        ind = contours < 0
        imagename = imagename if isinstance(imagename, str) else imagename[0]
        self.__gc1.show_contour(imagename, levels=contours[~ind], colors=color, linewidths=linesize)
        if negative:
            ind = np.ravel(np.where(contours > 0))
            self.__gc1.show_contour(imagename, levels=contours[ind], colors=color, linewidths=linesize, linestyles='--')

        if 'contour_set_2' in self.__gc1._layers:
            CS = self.__gc1._layers['contour_set_2']

            # Recast levels to new class
            if self.config['backgroundcontourlvlfmt']:
                nf = lambda x: self.config['backgroundcontourlvlfmt'] % float(x)
                CS.levels = [nf(val) for val in CS.levels]
                fmt = self.config['backgroundcontourlvlfmt']
                self.__gc1._figure._get_axes()[0].clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)

    def __gather_contours(self, neg=False, contstart=None, continterval=None, contend=None, contnoise=None):
        """
        contours1=np.arange(contstart, contstart*10.0, continterval, dtype='float32')
        contours2=np.arange(contstart*10.0, contstart*40.0, continterval*3.0, dtype='float32')
        contours3=np.arange(contstart*40.0, contstart*100.0, continterval*10.0, dtype='float32')
        poscontours=np.concatenate((contours1, contours2, contours3))
        """
        poscontours = np.arange(contstart[0], contend[0] + 1, continterval[0], dtype=np.float)
        contours = poscontours
        if neg[0]:
            negcontours = poscontours[::-1] * -1.0
            contours = np.concatenate((poscontours, negcontours)).astype(np.float)
        contours.sort()
        contours *= contnoise
        return contours

    def drawBeam(self):
        """
        """
        self.__tmp_plot.clf()
        if 'beams' not in dir(self):
            self.beams = []
        try:
            if self.config['backgroundfile']:
                self.beams.append(aplpy.FITSFigure(
                    self.config['backgroundfile'],
                    figure=self.__tmp_plot,
                    dimensions=(0, 1)))
            if len(self.mols) > 0:
                for m in self.mols:
                    f = None
                    if f'molecule{m}_redfile' in self.config:
                        f = f'molecule{m}_redfile'
                    elif f'molecule{m}_bluefile' in self.config:
                        f = f'molecule{m}_bluefile'
                    elif f'molecule{m}_file' in self.config:
                        f = f'molecule{m}_file'
                    if f is not None:
                        self.beams.append(aplpy.FITSFigure(
                            self.config[f],
                            figure=self.__tmp_plot,
                            dimensions=(0, 1)))
            for b in self.beams:
                b.add_beam()
                self.__gc1.add_beam()
        except Exception as e:
            print('Cant add beam', e)
            pass

    def plotFormat(self):
        """
        """
        # centering
        width = self.config['imagewidthra'] / 3600.0
        height = self.config['imagewidthdec'] / 3600.0
        self.__gc1.recenter(x=self.config['imagecenterra'], y=self.config['imagecenterdec'], width=width, height=height)

        #from IPython import embed; embed()
        #print(self.__gc1.world2pix)

        # axis labels
        if self.config['showAxisLabels']:
            self.__gc1.axis_labels.set_font(size=self.config['textsize'])
            self.__gc1.tick_labels.set_style('colons')
            self.__gc1.tick_labels.set_xformat('hh:mm:ss.s')
            self.__gc1.tick_labels.set_yformat('dd:mm:ss.s')
            self.__gc1.tick_labels.set_font(size=self.config['textsize'])
            self.__gc1.axis_labels.set_ypad(-2*self.config['textsize'])
            self.__gc1.ticks.set_color('black')
            self.__gc1.ticks.show()
        else:
            self.__gc1.axis_labels.hide()
            self.__gc1.tick_labels.hide()
            self.__gc1.ticks.hide()

        # scalebar
        if self.config['showscalebar']:
            self.scalebar = unit(vals=self.config['scalebarlength'], baseunit=self.config['scalebarunit']).convert('arcsec')
            self.__gc1.add_scalebar(self.scalebar / 3600., color=self.config['textcolor'])
            self.__gc1.scalebar.set_corner('bottom left')
            if self.scalebar > 2000:
                val = float(f'{self.scalebar * self.config["distance"] / (constants.pc / constants.au):1.1f}')
                u = 'pc'
            else:
                val = float(f'{self.scalebar * self.config["distance"]:1.1f}')
                u = 'au'
            self.__gc1.scalebar.set_label(f'{self.scalebar:1.1f}" ({val} {u})')
            self.__gc1.scalebar.set_linewidth(self.config['textsize']/3)
            self.__gc1.scalebar.set_font_size(self.config['textsize'])

        # beam
        if self.config['showbeam']:
            try:
                if typecheck(self.__gc1.beam):
                    for i, h in enumerate(self.__gc1.beam):
                        if i == 0:
                            for x in range(len(self.beams)):
                                self.__gc1.beam[i]._base_settings = {**self.__gc1.beam[i]._base_settings, **self.beams[x].beam._base_settings}

                        self.__gc1.beam[i].set_corner('bottom right')
                        self.__gc1.beam[i].set_color(self.config['textcolor'])
                        self.__gc1.beam[i].set_hatch('+')
                        self.__gc1.beam[i].set_alpha(1.0)
                        self.__gc1.beam[i].set_pad(self.config['beampad'] * i)
                else:
                    self.__gc1.beam.set_corner('bottom right')
                    self.__gc1.beam.set_color(self.config['textcolor'])
                    self.__gc1.beam.set_hatch('+')
                    self.__gc1.beam.set_alpha(1.0)
            except Exception as e:
                try:
                    for x in self.beams[0]._base_settings:
                        self.__gc1.beam._base_settings[x] = self.beams[0]._base_settings[x]
                except Exception as e:
                    pass

        if self.config['showlegend']:
            self.__gc1._ax1.legend(loc=self.config['legendlocation'])

    def drawOutflows(self):
        """draws outflows
        @input : xlength draws an outflow of this xlength
        @input : ylength "" ylength
        @input : ra places outflow at this pos
        @input : dec ""
        @input : paR and paB position angles of Red and blue outflow
        """
        for num in self.outflows:
            ra = icrs2deg(self.config[f'outflow{num}_ra']) * 15.
            dec = icrs2deg(self.config[f'outflow{num}_dec'])
            if self.config[f'outflow{num}_redlength'] is not None:
                dxred = self.config[f'outflow{num}_redlength'] * np.cos(self.config[f'outflow{num}_redangle'])
                dyred = self.config[f'outflow{num}_redlength'] * np.sin(self.config[f'outflow{num}_redangle'])
                self.__gc1.show_arrows(ra, dec, dxred, dyred, color='red')
            if self.config[f'outflow{num}_bluelength']:
                dxblue = self.config[f'outflow{num}_bluelength'] * np.cos(self.config[f'outflow{num}_blueangle'])
                dyblue = self.config[f'outflow{num}_bluelength'] * np.sin(self.config[f'outflow{num}_blueangle'])
                self.__gc1.show_arrows(ra, dec, dxblue, dyblue, color='blue')

    def showSources(self):
        """
        """
        for num in self.markers:
            ra = self.config[f'marker{num}_ra']
            dec = self.config[f'marker{num}_dec']
            label = self.config[f'marker{num}_name']
            w = self.config[f'marker{num}_width']
            h = self.config[f'marker{num}_height']
            a = self.config[f'marker{num}_angle']
            style = self.config[f'marker{num}_style'].lower()
            color = self.config[f'marker{num}_color']
            size = self.config[f'marker{num}_size']
            display = self.config[f'marker{num}_display']
            self.labels.append({'label': label, 'color': color, 'style': style,
                               'display': display, 'ra': ra, 'dec': dec, 'size': size})
            if style == 'rectangle':
                if a % 180 == 0:
                    self.__gc1.show_rectangles(ra, dec, w / 3600., h / 3600.,
                                               edgecolor=color, zorder=20,
                                               label=label,
                                               linewidth=self.config['contourlinesize'])
                else:
                    a *= np.pi / 180.
                    left = ra + w / 3600.
                    right = ra - w / 3600.
                    up = dec + h / 3600.
                    down = dec - h / 3600.
                    newx = lambda x, y, theta: (x * np.cos(-theta) - y * np.sin(-theta)) + ra
                    newy = lambda x, y, theta: (x * np.sin(-theta) + y * np.cos(-theta)) + dec
                    ur = (newx(-w/3600, h/3600., a), newy(-w/3600, h/3600., a))
                    ul = (newx(w/3600, h/3600., a), newy(w/3600, h/3600., a))
                    lr = (newx(-w/3600, -h/3600., a), newy(-w/3600, -h/3600., a))
                    ll = (newx(w/3600, -h/3600., a), newy(w/3600, -h/3600., a))
                    self.__gc1.show_polygons([np.array([ur, ul, ll, lr]).T], color=color)
            elif style == 'circle':
                self.__gc1.show_ellipses(ra, dec, w / 3600., h / 3600.,
                                         angle=a, edgecolor=color, zorder=20,
                                         linewidth=self.config['contourlinesize'])
                self.__gc1.show_markers(ra, dec, c=color, marker='_', zorder=-1, label=label)
            elif style:
                self.__gc1.show_markers(ra, dec, c=color, marker=style,
                    zorder=20, s=size,
                    linewidths = self.config['contourlinesize'],
                    label=label)
            else:
                continue

    def showColorbar(self):
        """
        """
        self.__gc1.add_colorbar()
        self.__gc1.colorbar.set_width(0.15)
        self.__gc1.colorbar.set_location('right')
        self.__gc1.colorbar.set_axis_label_text(self.config['coloraxislabel'])
        self.__gc1.colorbar.set_axis_label_font(size=self.config['textsize'], weight='black', style='oblique')
        self.__gc1.colorbar.set_font(size=self.config['textsize'], weight='bold')


    def addLabels(self):
        """
        """
        return
        if not self.config['showlegend']:
            return
        print(self.labels)
        for num, label in enumerate(self.labels):
            pos = 1. - (num + 1.) * self.fontspacereq
            if label['display'] is not None:
                ra = label['ra']
                dec = label['dec']
                self.__gc1.add_label(ra - label['display'] / 3600.,
                                     dec, f'{label["label"]}',
                                     relative=False,
                                     size=self.config['textsize'],
                                     color=label['color'])

    def save(self, dpi=150): 
        """plotting module
        @input extension tuple of extension types
        @input dpi, 400 is pretty high quality
        """ 
        self.__gc1.list_layers()
        print(dpi)
        #self.fig.set_facecolor('black')
        # self.fig.tight_layout()
        #from IPython import embed; embed()
        outfilename = self.config["outputfname"]
        if not typecheck(self.config['ext']):
            self.config['ext'] = [self.config['ext']]
        for ext in self.config['ext']:
            print(f'Saving: {outfilename}.{ext}')
            self.fig.savefig(f'{outfilename}.{ext}', dpi=dpi, format=ext,
                facecolor=self.fig.get_facecolor(), edgecolor='none',
                bbox_inches = 'tight', pad_inches = 0)
            print(f'saved: {outfilename}.{ext}')

    def main(self):
        """Runs through the main process."""
        self.drawBackground()
        self.addMoleculeContours()
        self.drawOutflows()
        self.showColorbar()
        self.showSources()
        self.drawBeam()
        self.addLabels()
        self.plotFormat()
        self.save()



def main(config):
    '''Main calling function for the program
    @input : configFname is the name of the configuration file. Has a default value just incase
    Loads in all of the values found in the configuration file
    '''
    plot = plotter(config)
    plot.main()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plots contours based on parameters found in the config file')
    parser.add_argument('--input', type=str, help='name of the config file')
    args = parser.parse_args()
    if not args.input:
        print('No input')
        exit()
    main(Config(target_config_file=args.input))

# end of code

# end of file
