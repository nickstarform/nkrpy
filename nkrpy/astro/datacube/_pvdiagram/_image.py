"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules

# relative modules

# global attributes
__all__ = ('test', 'main')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def binning(x, y, windowsize=0.1):
    newx, newy = [], []
    for xi in x:
        mask = np.abs(x - xi) < windowsize
        xmed = np.median(x[mask])
        if xmed in newx:
            continue
        newx.append(xmed)
        newy.append(np.median(y[mask]))
    return np.array(newx), np.array(newy)


def center_image(image, ra, dec, wcs):
    xcen = wcs(ra, 'pix', 'ra---sin')
    ycen = wcs(dec, 'pix', 'dec--sin')
    imcen = list(map(lambda x: x / 2, image.shape[1:]))
    center_shift = list(map(int, [imcen[0] - ycen, imcen[1] - xcen]))
    shift = [0]
    shift.extend(center_shift)
    shifted_image = inter.shift(image, shift)
    return shifted_image


def rotate_image(image, deg):
    rotated_image = inter.rotate(image, deg)
    return rotated_image


def sum_image(image, width: int):
    width = int(width / 2)
    center = list(map(lambda x: int(x / 2), image.shape))
    summed_image = np.sum(image[center[0] - width:center[0] + width, ...], axis=0)  # noqa
    return summed_image



def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
