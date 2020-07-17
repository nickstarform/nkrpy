"""."""
# flake8: noqa

# internal modules
import numbers

# external modules
import numpy as np

# relative modules
from .miscmath import inner_angle
# global attributes
__all__ = ('Vector', 'BaseVectorArray')
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class BaseVectorArray(object):
    """Main base vector type."""
    def __new__(cls, point: np.ndarray):
        if isinstance(point, (BaseVectorArray)):
            return point
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        if point.shape[0] != 3:
            point = (np.pad(np.array(point), (2,), 'constant',
                            constant_values=0))[2:5]
        return point


class Vector(object):
    """Assuming Cartesian of 1->3D.

    General vector class.
    These vectors point from p1 -> p2.
    Several methods are supported including subtracting, scalar
    multiplying, adding, cross product (*),
    and dot product (^).

    Example
    -------
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from nkrpy.math import vector
    from nkrpy.publication.plots import Arrow3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a = vector([0,0,0], [0,1,0])
    b = vector([0,0,0], [0,0,1])
    c = a * b
    d = b * a
    colors = ('blue', 'black', 'red', 'green')
    labels = ('Original Y', 'original Z', '+ X', '- X')
    for i, x in enumerate((a, b, c, d)):
        form = x.get_plotable()
        arrow = Arrow3D(*form, mutation_scale=20, lw=3, arrowstyle='-|>',
                        color=colors[i], label=labels[i])
        ax.add_artist(arrow)

    ax.legend()
    plt.show()
    """

    def __init__(self, point1: BaseVectorArray,
                 point2: BaseVectorArray, radius=None):
        """Vector class setup.

        These vectors point from p1 -> p2.
        Several methods are supported including subtracting, scalar
        multiplying, adding, cross product (*),
        and dot product (^).

        Parameters
        ----------
        point1: np.ndarray
            The first vertex
        point2: np.ndarray
            The second vertex

        """
        point1 = BaseVectorArray(point1)
        point2 = BaseVectorArray(point2)
        sub = (point2 + point1)
        self.__vec = {
            'sub': sub,
            'unitvec': self.unit_vector(sub),
            'points': [point1, point2],
            'radius': radius,
        }
        params = self.__calculate()
        self.__vec.update(params)

    @staticmethod
    def distance(point1: BaseVectorArray, point2: BaseVectorArray):
        point1 = BaseVectorArray(point1)
        point2 = BaseVectorArray(point2)
        return np.sqrt(np.prod((point2 - point1) ** 2))

    def vertex_has_point(self, point1: BaseVectorArray):
        point1 = BaseVectorArray(point1)
        for vertex in self.__vec['points']:
            p1_d = self.distance(vertex, point1)
            if p1_d <= self.__vec['radius']:
                return True
        return False

    @staticmethod
    def unit_vector(vector: BaseVectorArray):
        """Return the unit vector of the vector."""
        vector = BaseVectorArray(vector)
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between(v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'.

        Examples
        --------
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

        """
        if isinstance(v1, Vector):
            v1 = v1.__vec['sub']
        if isinstance(v2, Vector):
            v2 = v2.__vec['sub']
        print(v1, v2)
        return inner_angle(v1, v2)

    def angle(self, v2):
        if isinstance(v2, Vector):
            v2 = v2.__vec['sub']
        return self.angle_between(self.__vec['sub'], v2)

    def new(self, point1: BaseVectorArray,
            point2: BaseVectorArray) -> 'Vector':
        """Generate a new class instance."""
        point1 = BaseVectorArray(point1)
        point2 = BaseVectorArray(point2)
        return self.__class__(point1, point2)

    def get_sub(self) -> np.ndarray:
        """Get the corresponding vector from internal points."""
        return self.__vec['points'][-1] - self.__vec['points'][0]

    def get_vec(self) -> dict:
        """Return the vector info."""
        return self.__vec

    """ # work on this
    cpdef rotate(double[:, :] data: np.ndarray, long angle, float cval: float = np.NaN):
        cdef Py_ssize_t rows, cols
        rows, cols = np.asarray(data).shape
        cdef double theta = np.radians(angle)
        cdef double c, s
        c = np.cos(theta)
        s = np.sin(theta)
        cdef cnp.ndarray[double, ndim=2, mode='c'] r_matrix = np.array(((c, -s),
                                                                            (s, c)))
        print(r_matrix)                                                                    
        return np.dot(r_matrix, np.asarray(data))
    """

    def get_plotable(self) -> list:
        """Return list for 3D plotting.

        Reads the points set and returns a xs, ys, zs list of the points.
        You can neglect the zs if you want simply a 2d plot.
        """
        y = (np.vstack(self.__vec['points']).T).tolist()
        return y

    @staticmethod
    def __sph2cart(r, t, p) -> tuple:
        x = r * np.sin(p) * np.cos(t)
        y = r * np.cos(p) * np.cos(t)
        z = r * np.cos(p)
        return x, y, z

    @staticmethod
    def __cart2sph(x, y, z) -> tuple:
        s = x ** 2 + y ** 2
        r = s + z ** 2
        theta = 0
        if x != 0 and y != 0:
            theta = np.arctan2(y, x)
        phi = 0
        if z != 0:
            phi = np.arctan2(s ** 0.5, z)
        return np.sqrt(r), theta, phi

    def __calculate(self, vec: BaseVectorArray = None) -> dict:
        if vec is None:
            vec = self.__vec['points'][-1].copy()
        vec = BaseVectorArray(vec)
        subvec = vec + self.__vec['points'][0]
        points = subvec.tolist()
        ndim = subvec.shape[0]
        cdef int r = 0
        cdef int t = 0
        cdef int p = 0
        cdef int y = 0
        cdef int z = 0
        if ndim < 3:
            z = 0
        if ndim < 2:
            y = 0
            r, t, p = self.__cart2sph(*points, y, z)
        elif ndim == 2:
            r, t, p = self.__cart2sph(*points, z)
        elif ndim == 3:
            r, t, p = self.__cart2sph(*points)

        t = t % (2. * np.pi)
        p = p % (np.pi)

        return {'ndim': ndim,
                'distance': r,
                'theta': t,
                'phi': p,
                'sub': subvec,
                'unitvec': self.unit_vector(subvec),
                'points': (self.__vec['points'][0], vec)}

    def __mul__(self, value):
        assert isinstance(value, Vector) or isinstance(value, np.ndarray)
        if isinstance(value, np.ndarray):
            vec = BaseVectorArray(value)
        else:
            vec = value.get_sub()
        cross = np.cross(self.get_sub(), vec)
        return self.new(self.__vec['points'][0],
                        self.__vec['points'][0] + cross)

    def __rmul__(self, value):
        assert isinstance(value, Vector) or isinstance(value, np.ndarray)
        if isinstance(value, np.ndarray):
            if value.shape[0] != 3:
                value = (np.pad(value, (2,), 'constant',
                                constant_values=0))[2:5]
            vec = (self.new(self.__vec['points'][0], value)).get_sub()
        else:
            vec = value.get_sub()
        cross = np.cross(vec, self.get_sub())
        return self.new(self.__vec['points'][0],
                        self.__vec['points'][0] + cross)

    def __abs__(self):
        """Dunder."""
        return self.__vec['distance']

    def __add__(self, value: 'Vector'):
        """Dunder."""
        assert isinstance(value, Vector)
        # now do vector add.Only takes mag and angles
        diff = self.__vec['points'][-1] - value.__vec['points'][0]
        newvec = value.__vec['points'][-1] + diff
        return self.new(self.__vec['points'][0], newvec)

    def __radd__(self, value: 'Vector'):
        """Dunder."""
        assert isinstance(value, Vector)
        # now do vector add.Only takes mag and angles
        diff = value.__vec['points'][-1] - self.__vec['points'][0]
        newvec = self.__vec['points'][-1] + diff
        return self.new(value.__vec['points'][0], newvec)

    def __sub__(self, value: 'Vector'):
        """Dunder."""
        assert isinstance(value, Vector)
        v1 = self.__vec['points'][-1]
        v2 = value.sub()
        newvec = v1 - v2
        return self.new(self.__vec['points'][0], newvec)

    def __rsub__(self, value: 'Vector'):
        """Dunder."""
        assert isinstance(value, Vector)
        v2 = self.sub()
        v1 = value.__vec['points'][-1]
        newvec = v1 - v2
        return self.new(value.__vec['points'][0], newvec)

    def __divmod__(self, value):
        """Dunder."""
        assert isinstance(value, numbers.Number)
        assert value > 0
        dist = self.__vec['distance'] / value
        x, y, z = self.__sph2cart(dist, self.__vec['theta'],
                                  self.__vec['phi'])
        newvec = np.array([x, y, z])
        newvec += self.__vec['points'][0]
        return self.new(self.__vec['points'][0], newvec)

    def __truediv__(self, *args, **kwargs):
        """Dunder."""
        return self.__divmod__(*args, **kwargs)

    def __xor__(self, value):
        """Dunder.
        
        If scalar assume distance mult. If vector
        assume dot product. If prior returns vector
        if latter returns scalar.
        """
        if value is self:
            return self.__vec['distance']
        if isinstance(value, Vector):
            newval = np.sum(self.__vec['sub'] * value.__vec['sub'])
            return newval
        assert isinstance(value, numbers.Number)
        dist = self.__vec['distance'] * value
        x, y, z = self.__sph2cart(dist, self.__vec['theta'],
                                  self.__vec['phi'])
        newvec = np.array([x, y, z])
        newvec += self.__vec['points'][0]
        return self.new(self.__vec['points'][0], newvec)

    def __rxor__(self, *args, **kwargs):
        """Dunder."""
        return self.__mul__(*args, **kwargs)

    def __pow__(self, value):
        """Dunder.
        
        if value power of 2 then returns scalar.
        """
        assert isinstance(value, numbers.Number)
        even = value // 2
        remain = value % 2
        scalar = 1
        vec = None
        if even > 0:
            scalar = (self.__xor__(self)) ** even
        if remain > 0:
            vec = self
        if even == 0:
            return self
        if remain == 0:
            return scalar
        if vec is not None:
            return scalar * vec

    def __gt__(self, value):
        """Dunder."""
        assert isinstance(value, Vector)
        return self.__vec['distance'] > value.__vec['distance']

    def __lt__(self, value):
        """Dunder."""
        assert isinstance(value, Vector)
        return self.__vec['distance'] < value.__vec['distance']

    def __le__(self, value):
        """Dunder."""
        assert isinstance(value, Vector)
        return self.__vec['distance'] <= value.__vec['distance']

    def __ge__(self, value):
        """Dunder."""
        assert isinstance(value, Vector)
        return self.__vec['distance'] >= value.__vec['distance']

    def __eq__(self, value):
        """Dunder."""
        assert isinstance(value, Vector)
        return self.__vec['sub'] == value.__vec['sub']

    def __ne__(self, value):
        """Dunder."""
        assert isinstance(value, Vector)
        return self.__vec['sub'] != value.__vec['sub']


# end of code

# end of file
