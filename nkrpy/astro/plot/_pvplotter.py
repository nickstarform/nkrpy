"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib
import math
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
__all__ = ['PVImagePlot', 'auto_pv_plot']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
#set_style()


def auto_pv_plot(filename: str):
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


class PeakPVImagePlot(Plot):
    """Generic PV DIagram plotter.
    
    Methods
    -------
    add_keplerian: add keplerian curve
    set_disk_image: plot background (data, color)
    add_infall: plot contours  (contour params)
    add_contour: plot outflows (vectors, xcen, ycen, angle, color, length)
    set_outflow_image: plot sources (marker styles)
    add_region: plot regions (ellipse, rectangle, xcen, ycen, major/xwidth, minor/ywidth, angle)
    set_colorbar
    add_title
    set_ticks
    refresh_wcs: recreates the image wcs based off of new centers and widths
    refresh_data: reslices the data from the original using the loaded wcs
    """

    pass

class PVImagePlot(Plot):
    """Generic PV DIagram plotter.
    
    Methods
    -------
    add_keplerian: add keplerian curve
    set_disk_image: plot background (data, color)
    add_infall: plot contours  (contour params)
    add_contour: plot outflows (vectors, xcen, ycen, angle, color, length)
    set_outflow_image: plot sources (marker styles)
    add_region: plot regions (ellipse, rectangle, xcen, ycen, major/xwidth, minor/ywidth, angle)
    set_colorbar
    add_title
    set_ticks
    refresh_wcs: recreates the image wcs based off of new centers and widths
    refresh_data: reslices the data from the original using the loaded wcs
    """

    pass



# end of code


# end of file
