"""."""
# flake8: noqa

# internal modules
import itertools

# external modules
import numpy as np

# relative modules
from ..math import vector
from ..math.vector import BaseVectorArray

# global attributes
__all__ = ('Triangle',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

"""
Define a base image
Define the points (min 2, max 3) to compare
load in all images
find all gaussians in images
make triangles from all points
"""

class Triangle(object):
    AreaError = 0.1  # amount of error allowed for area.
    AngleError = 0.005  # amount of error allowed in angle
    SideError = 0.005  # amount of error allowed in side length

    def __init__(self, vertex1: BaseVectorArray, vertex2: BaseVectorArray,
                 vertex3: BaseVectorArray, radius=0):
        """Dunder."""
        vectors = itertools.combinations((vertex1, vertex2, vertex3), 2)
        vectors = [vector(*x, (radius, radius, radius)) for x in vectors]
        vectors.sort(key=lambda x: x.get_vec()['distance'])
        self.vectors = vectors
        self.sides = self.get_sides()
        self.area = self.get_area()
        self.angles = self.get_angles()
        self.radius = radius

    def get_sides(self):
        """Generate the sides from vectors."""
        sides = np.array([x.get_vec()['distance'] for x in self.vectors])
        return sides

    def get_area(self):
        """Generate the area from the sides using Hernon's Formula."""
        cumvec = self.sides
        vecsum = np.sum(cumvec) / 2.
        area = np.sqrt(vecsum * np.prod(vecsum - cumvec))
        return area

    def get_angles(self):
        """Generate the angles from side combinations."""
        veccomb = itertools.combinations(self.vectors, 2)
        angles = [x[0].angle(x[1]) for x in veccomb]
        angles.sort()
        return np.array(angles, dtype=np.float)

    def __eq__(self, value: 'Triangle'):
        """Dunder."""
        if self.simi_aaaa(self, value):
            return True
        if self.simi_sss(self, value):
            return True
        if self.simi_sas(self, value):
            return True
        return False

    def __ne__(self, value: 'Triangle'):
        """Dunder."""
        return not (self == value)

    # Function for SAS similarity
    @classmethod
    def simi_sas(cls, t1: 'Triangle', t2: 'Triangle') -> bool:
        """Side-Angle-Side similar triangle test.

        Parameters
        ----------
        t1: 'Triangle'
            The first triangle to compare against.
        t2: 'Triangle'
            The second triangle to compare against.

        Returns
        -------
        bool:
            True/False if similar.

        """
        s1 = t1.sides
        s2 = t2.sides
        a1 = t1.angles
        a2 = t2.angles
        diff_s1s2 = np.abs(s1 - s2)
        similar_s = np.logical_or(np.logical_or(diff_s1s2 <= t1.radius,
                                                diff_s1s2 <= t2.radius),
                                  diff_s1s2 <= cls.SideError)
        diff_a = a1 / a2
        similar_a = np.logical_and((1. - cls.AngleError) <= diff_a,
                                   diff_a <= (1. + cls.AngleError))

        # angle b / w two smallest sides is largest.
        for i in range(diff_s1s2.shape[0]):
            j = (i + 1) % diff_s1s2.shape[0]
            k = (i + 2) % diff_s1s2.shape[0]
            if similar_s[i] == similar_s[j]:
                if similar_a[k]:
                    print('True by sas')
                    return True
        return False

    # Function for SSS similarity
    @classmethod
    def simi_sss(cls, t1: 'Triangle', t2: 'Triangle') -> bool:
        """3-side similar triangle test.

        Parameters
        ----------
        t1: 'Triangle'
            The first triangle to compare against.
        t2: 'Triangle'
            The second triangle to compare against.

        Returns
        -------
        bool:
            True/False if similar.

        """
        s1 = t1.sides
        s2 = t2.sides
        diff_s1s2 = np.abs(s1 - s2)
        similar_s = np.logical_or(np.logical_or(diff_s1s2 <= t1.radius,
                                                diff_s1s2 <= t2.radius),
                                  diff_s1s2 <= cls.SideError)

        # Check for SSS
        if all(similar_s):
            print('True by sss')
            return True
        return False

    # Function for AAA (Congruency) + Area similarity
    @classmethod
    def simi_aaaa(cls, t1: 'Triangle', t2: 'Triangle') -> bool:
        """3-angle congruency + area similar triangle test.

        Parameters
        ----------
        t1: 'Triangle'
            The first triangle to compare against.
        t2: 'Triangle'
            The second triangle to compare against.

        Returns
        -------
        bool:
            True/False if similar.

        """
        a1 = t1.angles
        a2 = t2.angles
        diff_a = a1 / a2
        diff_area = t1.area / t2.area
        similar = np.logical_and((1. - cls.AngleError) <= diff_a,
                                 diff_a <= (1. + cls.AngleError))

        similar_area = np.logical_and((1. - cls.AreaError) <= diff_area,
                                      diff_area <= (1. + cls.AreaError))
        # Check for AAA _ Area
        # TODO: Why is all(np.ndarray) much faster than np.ndarray.all()
        if all(similar) and similar_area:
            print('True by aaaa')
            return True
        return False


# end of code

# end of file
