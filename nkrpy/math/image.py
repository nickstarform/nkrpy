"""."""
# flake8: noqa

# internal modules

# external modules
import numpy as np

# relative modules

# global attributes
__all__ = ('raster_matrix', 'gen_angles', 'rotate_points', 'rotate_matrix')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def raster_matrix(*args, auto=False, **kwargs):
    """Return a matrix of a raster track.

    Assuming all units are the same!

    Parameters
    ----------
    Cen: iterable[float]
        center points
    width: float
        total width (evenly split)
    height: float
        total height (evenly split)
    fov: float
        field of view of window
    auto: bool
        if auto is set will construct a double grid, one of specified
    plot: boolean
        if should plot the output
    direction and the next grid of opposite to maximize sensitivity
    main: <h/v> determine which is the major track
    h: str [+ | -]
        direction of starting horizontal track
    v: str  [+ | -]
        direction of starting vertical track
    sample: float
        Amount of overlap. > 0 with 1 being exactly no overlap and
        infinity being complete overlap.
    """
    if auto:
        firstn, firstm = _raster_matrix_con(*args, **kwargs)
        secondn, secondm = _raster_matrix_con(*args, rev=True, **kwargs)
        totaln = firstn + secondn
        totalm = np.concatenate((firstm, secondm))
    else:
        totaln, totalm = _raster_matrix_con(*args, **kwargs)
    return totaln, totalm


def gen_angles(start, end, resolution=1, direction='+'):
    """Generate angles between two designations."""
    if direction == "+":  # positive direction
        diff = round(angle_clockwise(ang_vec(start), ang_vec(end)), 2)
        numd = int(ceil(diff / resolution)) + 1
        final = [round((start + x * resolution), 2) % 360 for x in range(numd)]
    elif direction == "-":  # negative direction
        diff = round(360. - angle_clockwise(ang_vec(start), ang_vec(end)), 2)
        numd = int(ceil(diff / resolution)) + 1
        final = [round((start - x * resolution), 2) % 360 for x in range(numd)]
    return final


def rotate_points(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def rotate_matrix(origin, matrix, angle):
    """Rotate a matrix counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = matrix[:, 0], matrix[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.concatenate((qx.reshape(qx.shape[0], 1),qy.reshape(qy.shape[0], 1)), axis=1)


def _raster_matrix_con(fov, cen=(0, 0), width=1, height=1, main='h',
                       theta=0, h='+', v='-', sample=2.,
                       box_bounds=None, rev=False, plot=False):
    """Main constructor."""
    if rev:
        h = '+' if h == '-' else '-'
        v = '+' if v == '-' else '-'
        main = 'h' if main == 'v' else 'v'
    print(h, v)
    if not box_bounds:
        box_bounds = ((cen[0] - width / 2., cen[1] + height / 2.),
                      (cen[0] + width / 2., cen[1] + height / 2.),
                      (cen[0] - width / 2., cen[1] - height / 2.),
                      (cen[0] + width / 2., cen[1] - height / 2.))

    num_centers_w = int(np.ceil(2. * width / (fov / sample) - 4))
    num_centers_h = int(np.ceil(2. * height / (fov / sample) - 4))
    vertrange = np.linspace(box_bounds[2][1], box_bounds[0][1], endpoint=True,
                            num=num_centers_h)
    horirange = np.linspace(box_bounds[0][0], box_bounds[1][0], endpoint=True,
                            num=num_centers_w)

    if v == '-':
        vertrange = np.array(list(vertrange)[::-1])
    if h == '-':
        horirange = np.array(list(horirange)[::-1])

    alldegrees = []
    count = 0
    if main == 'h':
        for i, v in enumerate(vertrange):
            _t = list(horirange)

            if count % 2 == 1:  # negative direction
                _t = _t[::-1]

            for x in _t:
                alldegrees.append(np.array([x, v]))
            count += 1
    else:
        for i, h in enumerate(horirange):
            _t = list(vertrange)

            if count % 2 == 1:  # negative direction
                _t = _t[::-1]

            for x in _t:
                alldegrees.append(np.array([h, x]))
            count += 1
    alldegrees = np.array(alldegrees)
    if theta % 360. != 0.:
        _t = deepcopy(alldegrees)
        alldegrees = rotate_matrix(cen, _t, theta)

    if plot:
        _plot_raster(alldegrees)
    return num_centers_w * num_centers_h, alldegrees


def _plot_raster(matrix):
    """Plotter for the raster matrix."""
    plt.figure(figsize=[16, 16])

    plt.plot(matrix[:, 0], matrix[:, 1], 'r-')
    plt.plot(matrix[:, 0], matrix[:, 1], 'b.')
    plt.plot(matrix[0, 0], matrix[0, 1], '*', color='black', label='start')
    plt.plot(matrix[-1, 0], matrix[-1, 1], '*', color='purple', label='end')
    a = plt.legend()
    plt.title(f'Raster Scan: {matrix[0]} to {matrix[-1]}')
    plt.show()

# end of code

# end of file
