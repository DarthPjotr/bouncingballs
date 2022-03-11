"""
Generates n-dimensional rotation matrices
"""

# pylint: disable=I
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

import math
from pprint import pprint as pp
from itertools import combinations
import random
import sys

import numpy
from numpy import linalg

class RotationMatrix():
    """
    Generates n-dimensional rotation matrices
    """
    def __init__(self, dimensions) -> None:
        self.dimensions = dimensions
        self.identity = numpy.eye(self.dimensions)
        self.axis = [i for i in range(self.dimensions)]
        self.combinations = list(combinations(self.axis, 2))

    def single_rotation(self, axis1, axis2, angle):
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
        M = self.identity.copy()
        for i, j, angle in rotations:
            m = self.single_rotation(i, j, angle)
            M = M @ m

        return M

    def align(self, v1, v2):
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
        assert A.shape == B.shape
        n, m = A.shape

        H = ((A).T @ (B)) / n
        U, D, VT = numpy.linalg.svd(H)
        d = numpy.sign(numpy.linalg.det(U) * numpy.linalg.det(VT))
        S = numpy.diag([1] * (m - 1) + [d])

        R = U @ S @ VT

        return R

    def _kabsch_umeyama(self, A, B):
        # https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
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
        0, 1, math.pi/5) @ matrix.single_rotation(2, 3, math.pi/7)
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
    test_matrix()

if __name__ == "__main__":
    main()
