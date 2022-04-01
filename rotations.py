"""
Generates n-dimensional rotation matrices
"""

# pylint: disable=I
# pylint: disable=invalid-name

import math
from math import sin, cos
from pprint import pprint as pp
from itertools import combinations
import random
import sys

import numpy
from numpy import linalg

import clifford


class Rotations():
    """
    Generates n-dimensional rotors using Geometric Algebra

    >>> R = Rotations(4)
    >>> rotor = R.bivector_rotation(R.blades['e12'], pi/4) # rotation
    >>> Vr = rotor*V*~rotor
    or
    >>> v = R.to_np_vector(V)
    >>> matrix = R.to_matrix(rotor)
    >>> vr = matrix@v

    """

    def __init__(self, dimensions) -> None:
        """
        Creates Geometric Algebra Rotations class for n-dimensions

        Args:
            dimensions (int): number of dimensions
        """
        self.dimensions = dimensions
        self.layout, self.blades = clifford.Cl(dimensions)
        for name in self.blades:
            setattr(self, name, self.blades[name])

        self.unitvectors = self.layout.blades_of_grade(1)
        # self._heatup()

    def _heatup(self):
        """
        do some calculation to get the jit to kick in
        """
        V1 = self.layout.randomV()
        V2 = self.layout.randomV()
        B = V1^V2
        R = self.bivector_rotation(B, 0.5)
        _ = R*V1*~R

    def __str__(self):
        return f"{self.dimensions}, {self.layout.blades_of_grade(1)}"

    def to_ga_vector(self, v):
        """
        Converts numpy.array vector to GA vector

        Args:
            v (numpy.array): vector

        Returns:
            GA vector: GA vector
        """
        V = v@self.unitvectors
        return V

    def to_np_vector(self, V):
        """
        Converts GA vector to numpy.array vector

        Args:
            V (GA vector): GA vector

        Returns:
            numpy.array: vector
        """
        v = numpy.array([V[e] for e in self.unitvectors])
        return v

    def to_matrix(self, rotor):
        """
        Converts GA rotor to numy.array matrix

        Args:
            rotor (GA rotor): the rotor

        Returns:
            numpy.array: rotation matrix
        """
        B = [rotor*a*~rotor for a in self.unitvectors]

        matrix = numpy.array([
            [float((b | a)(0)) for b in B]
            for a in self.unitvectors])
        return matrix

    def bivector_rotation(self, B, angle):
        """
        Creates rotor by the provided bivector and angle.

        Args:
            B (bivector): plane of rotation
            angle (float): angle of rotation

        Returns:
            GA rotor: the rotor
        """
        B = B/abs(B)
        R = cos(angle/2) - sin(angle/2)*B
        # R = e**(-angle/2*B)  # enacts rotation by
        rotor = R/abs(R)
        return rotor

    def rotation(self, V1, V2, angle):
        """
        Creates rotor by provided vectors and angle

        Args:
            V1 (GA vector): vector 1
            V2 (GA vector): vector 2
            angle (float): angle in radians

        Returns:
            GA rotor: the rotor
        """
        B = V1^V2
        if abs(B) == 0:
            raise ValueError(f"invalid or parallel vectors: {V1} and {V2}")
        return self.bivector_rotation(B, angle)

    def axis_rotation(self, axis1, axis2, angle):
        """
        Creates rotor by provides axis and angle

        Args:
            axis1 (int): axis (1: x, 2: y, 3: z, ...)
            axis2 (int): axis (1: x, 2: y, 3: z, ...)
            angle (float): angle in radians

        Returns:
            GA rotor: the rotor
        """
        E = self.blades[f"e{axis1}{axis2}"]
        return self.bivector_rotation(E, angle)

    def combined_rotations(self, rotations):
        """
        combines single rotations

        Args:
            rotations (list of rotors): set of single rotors

        Returns:
            numpy.array: rotation matrix
        """
        rotor = self.blades['']
        for R in rotations:
            rotor = rotor*R

        return rotor

    def align(self, V1, V2):
        """
        calculates rotor to align two vectors

        Args:
            v1 (numpy.array): first vector
            v2 (numpy.array): second vector

        Returns:
            numpy.array: rotation matrix
        """
        N1 = V1/abs(V1)
        N2 = V2/abs(V2)
        Nm = (N1+N2)/abs(N1+N2)
        rotor = Nm*N1

        return rotor

class RotationMatrix():
    """
    Generates n-dimensional rotation matrices
    """
    def __init__(self, dimensions) -> None:
        self.dimensions = dimensions
        self.identity = numpy.eye(self.dimensions)
        self.axis = [i for i in range(self.dimensions)]
        self.combinations = list(combinations(self.axis, 2))
        self.nrotations = len(self.combinations)

    def single_rotation(self, axis1, axis2, angle):
        """
        Single rotation in plane defined by the provided axis of the coordinate system, leaving the other axis stationary.
        0 = x-axis
        1 = y-axis
        2 = z-axis
        3.. = higher dimensional axis

        Args:
            axis1 (int): first axis
            axis2 (int): second axis
            angle (float): angle of rotation

        Returns:
            numpy.array: rotation matrix
        """
        matrix = self.identity.copy()
        cos_ = math.cos(angle)
        matrix[axis1, axis1] = cos_
        matrix[axis2, axis2] = cos_

        sin_ = math.sin(angle)
        sign = 1
        if (axis2 - axis1) % 2:
            sign = -1
        matrix[axis1, axis2] = -sign * sin_
        matrix[axis2, axis1] = sign * sin_
        return matrix

    def combined_rotations(self, rotations):
        """
        combines single rotations

        Args:
            rotations (list[(axis1: int, axis2: int, angle: float),...]): set of single rotations

        Returns:
            numpy.array: rotation matrix
        """
        M = self.identity.copy()
        for i, j, angle in rotations:
            if i not in self.axis or j not in self.axis:
                continue
            m = self.single_rotation(i, j, angle)
            M = M @ m

        return M

    def align(self, v1, v2):
        """
        calculates rotation matrix to align two vectors

        Args:
            v1 (numpy.array): first vector
            v2 (numpy.array): second vector

        Returns:
            numpy.array: rotation matrix
        """
        v1 = numpy.array(v1)
        v1 = v1 / math.sqrt(v1@v1)

        v2 = numpy.array(v2)
        v2 = v2 / math.sqrt(v2@v2)

        z = numpy.zeros(self.dimensions)

        A = numpy.array([v1, z])
        B = numpy.array([v2, z])

        # R, c, t = self._full_kabsch_umeyama(A, B)
        R = self._simple_kabsch_umeyama(A, B)

        return R

    def _simple_kabsch_umeyama(self, A, B):
        """
        simplified Kabsch Uneyama algorithm

        Args:
            A (numpy.array): first set of vectors
            B (numpy.array): second set of vectors

        Returns:
            numpy.array: rotation matrix
        """
        assert A.shape == B.shape
        n, m = A.shape

        H = ((A).T @ (B)) / n
        U, D, VT = numpy.linalg.svd(H) # pylint: disable=unused-variable
        d = numpy.sign(numpy.linalg.det(U) * numpy.linalg.det(VT))
        S = numpy.diag([1] * (m - 1) + [d])

        R = U @ S @ VT

        return R

    def _kabsch_umeyama(self, A, B):
        """
        full Kabsch Uneyama algorithm.
        using: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

        Args:
            A (numpy.array): first set of vectors
            B (numpy.array): second set of vectors

        Returns:
            numpy.array: rotation matrix
        """
        assert A.shape == B.shape
        n, m = A.shape

        EA = numpy.mean(A, axis=0)
        EB = numpy.mean(B, axis=0)
        VarA = numpy.mean(numpy.linalg.norm(A - EA, axis=1) ** 2)

        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = numpy.linalg.svd(H)
        d = numpy.sign(numpy.linalg.det(U) * numpy.linalg.det(VT))
        S = numpy.diag([1] * (m - 1) + [d])

        R = U @ S @ VT
        c = VarA / numpy.trace(numpy.diag(D) @ S)
        t = EA - c * R @ EB

        return R, c, t


def test_matrix():
    """
    Tests
    """
    numpy.set_printoptions(precision=4, suppress=True, linewidth=95)

    matrix = RotationMatrix(4)
    M = matrix.identity.copy()
    print(matrix.combinations)
    print(len(matrix.combinations))
    for i, j in matrix.combinations:
        m = matrix.single_rotation(i, j, math.pi/4)
        pp(m)
        M = M @ m

    print("")
    pp(M)
    print("transpose")
    pp(M.transpose())
    print("conjugate")
    pp(M.conjugate())
    print("inverse")
    pp(linalg.inv(M))

    M = matrix.single_rotation(
        0, 1, math.pi/2) @ matrix.single_rotation(2, 3, math.pi/7)
    print("\neigen values")
    print(M)
    W, V = linalg.eig(M)
    print(W)
    print(V)
    print(linalg.eigvals(M))



    pp(M.trace())
    pp(linalg.det(M))

    sys.exit()

    print("")
    # math.pi/4
    rotations = [[i,j, random.random() * math.pi] for i,j in matrix.combinations]
    pp(rotations)
    R = matrix.combined_rotations(rotations)
    pp(R)
    pp(linalg.det(R))
    pp(linalg.eig(R))
    pp(linalg.matrix_rank(R))

    v1 = numpy.array([1, 2, 3, 4, 6,7,8,9])
    v1 = v1[:matrix.dimensions]
    v1 = v1 / math.sqrt(v1@v1)

    v2 = numpy.array([5, 6, 7, 8, 9,2,3,4,5])
    v2 = v2[:matrix.dimensions]
    v2 = v2 / math.sqrt(v2@v2)

    print("\nKABSCH")
    pp(v1)
    pp(v2)
    R = matrix.align(v1, v2)
    # R, c, t = kabsch_umeyama(A,B)
    pp(R)
    # pp(c)
    # pp(t)
    print("")
    pp(v1 @ R)
    pp(v2)
    print(numpy.allclose(v1@R, v2))
    print("")
    pp(v2 @ R.T)
    pp(v1)
    print(numpy.allclose(v2@R.T, v1))


    print("")
    pp(linalg.eig(R))

def main():
    """
    main
    """
    test_matrix()

if __name__ == "__main__":
    main()
