"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import matplotlib as mpl
from matplotlib.cm import register_cmap, get_cmap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# relative modules

# global attributes
__all__ = ('mainColorMap', 'mapper')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def mapper(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])  # noqa
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return LinearSegmentedColormap('colormap', cdict, 1024)


def mainColorMap(colorMap):
    cdict = {}
    if colorMap == 'ds9':
        for c, v in {
            'ds9b': {'red': lambda v : 4 * v - 1, 
              'green': lambda v : 4 * v - 2,
              'blue': lambda v : np.select([v < 0.25, v < 0.5, v < 0.75, v <= 1],
                                            [4 * v, -4 * v + 2, 0, 4 * v - 3])},

      # Note that this definition slightly differs from ds9cool, but make more sense to me...
            'ds9cool': {'red': lambda v : 2 * v - 1, 
                 'green': lambda v : 2 * v - 0.5,
                 'blue': lambda v : 2 * v},

            'ds9a': {'red': lambda v : np.interp(v, [0, 0.25, 0.5, 1],
                                              [0, 0, 1, 1]),
               'green': lambda v : np.interp(v, [0, 0.25, 0.5, 0.77, 1],
                                                [0, 1, 0, 0, 1]),
               'blue': lambda v : np.interp(v, [0, 0.125, 0.5, 0.64, 0.77, 1],
                                               [0, 0, 1, 0.5, 0, 0])},

            'ds9i8': {'red': lambda v : np.where(v < 0.5, 0, 1), 
              'green': lambda v : np.select([v < 1/8., v < 0.25, v < 3/8., v < 0.5,
                                             v < 5/8., v < 0.75, v < 7/8., v <= 1],
                                            [0, 1, 0, 1, 0, 1, 0, 1]),
              'blue': lambda v : np.select([v < 1/8., v < 0.25, v < 3/8., v < 0.5,
                                            v < 5/8., v < 0.75, v < 7/8., v <= 1],
                                            [0, 0, 1, 1, 0, 0, 1, 1])},

            'ds9aips0': {'red': lambda v : np.select([v < 1/9., v < 2/9., v < 3/9., v < 4/9., v < 5/9.,
                                              v < 6/9., v < 7/9., v < 8/9., v <= 1],
                                              [0.196, 0.475, 0, 0.373, 0, 0, 1, 1, 1]), 
                  'green': lambda v : np.select([v < 1/9., v < 2/9., v < 3/9., v < 4/9., v < 5/9.,
                                              v < 6/9., v < 7/9., v < 8/9., v <= 1],
                                              [0.196, 0, 0, 0.655, 0.596, 0.965, 1, 0.694, 0]),
                  'blue': lambda v : np.select([v < 1/9., v < 2/9., v < 3/9., v < 4/9., v < 5/9.,
                                              v < 6/9., v < 7/9., v < 8/9., v <= 1],
                                              [0.196, 0.608, 0.785, 0.925, 0, 0, 0, 0, 0])},

            'ds9rainbow': {'red': lambda v : np.interp(v, [0, 0.2, 0.6, 0.8, 1], [1, 0, 0, 1, 1]),
                    'green': lambda v : np.interp(v, [0, 0.2, 0.4, 0.8, 1], [0, 0, 1, 1, 0]),
                    'blue': lambda v : np.interp(v, [0, 0.4, 0.6, 1], [1, 1, 0, 0])},

      # This definition seems a bit strange...
            'ds9he': {'red': lambda v : np.interp(v, [0, 0.015, 0.25, 0.5, 1],
                                              [0, 0.5, 0.5, 0.75, 1]),
               'green': lambda v : np.interp(v, [0, 0.065, 0.125, 0.25, 0.5, 1],
                                                [0, 0, 0.5, 0.75, 0.81, 1]),
               'blue': lambda v : np.interp(v, [0, 0.015, 0.03, 0.065, 0.25, 1],
                                               [0, 0.125, 0.375, 0.625, 0.25, 1])},

            'ds9heat': {'red': lambda v : np.interp(v, [0, 0.34, 1], [0, 1, 1]),
                 'green': lambda v : np.interp(v, [0, 1], [0, 1]),
                 'blue': lambda v : np.interp(v, [0, 0.65, 0.98, 1], [0, 0, 1, 1])}}.items():

            cdict[c] = LinearSegmentedColormap(c, segmentdata=v)

      # Set aliases, where colormap exists in matplotlib
    for c, v in cdict.items():
      # Register all other colormaps
        register_cmap(c, cmap=v)
    for s, f in [['ds9bb','afmhot'], ['ds9grey', 'gray']]:
        register_cmap(s, cmap=get_cmap(f))


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    print('Test Passed')

# end of code

# end of file
