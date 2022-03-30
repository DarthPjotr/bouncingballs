# pylint: disable=C,I
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable

import math
from math import cos, sin, e, pi
import sys
import random
import clifford as cf
import numpy

class RotationMatrix():
    """
    Generates n-dimensional rotation matrices
    """

    def __init__(self, dimensions) -> None:
        self.dimensions = dimensions
        self.layout, self.blades = cf.Cl(dimensions)
        for name in self.blades:
            setattr(self, name, self.blades[name])

        self.unitvectors = self.layout.blades_of_grade(1)

    def __str__(self):
        return f"{self.dimensions}, {self.layout.blades_of_grade(1)}, {self.layout.blades_of_grade(DIMENSIONS)}"

    def to_matrix(self, rotor):
        B = [rotor*a*~rotor for a in self.unitvectors]

        matrix = numpy.array([
            [float((b|a)(0)) for b in B]
            for a in self.unitvectors])
        return matrix

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
        E = self.blades[f"e{axis1}{axis2}"]
        rotor = cos(angle/2) - sin(angle/2)*E
        # rotor = e**(-angle*(E))  # enacts rotation by pi/2
        return rotor

    def single_bivector_rotation(self, B, angle):
        """
        Single rotation in plane defined by the provided bivector.

        Args:
            B (bivector): plane of rotation
            angle (float): angle of rotation

        Returns:
            numpy.array: rotation matrix
        """
        B = B/abs(B)
        R = cos(angle/2) - sin(angle/2)*B
        # R = e**(-angle/2*B)  # enacts rotation by
        rotor = R/abs(R)
        return rotor


    def combined_rotations(self, rotations):
        """
        combines single rotations

        Args:
            rotations (list[(axis1: int, axis2: int, angle: float),...]): set of single rotations

        Returns:
            numpy.array: rotation matrix
        """
        rotor = self.blades['']
        for axis1, axis2, angle in rotations:
            R1 = self.single_rotation(axis1, axis2, angle)
            rotor = rotor*R1

        return rotor

    def align(self, v1, v2):
        """
        calculates rotation matrix to align two vectors

        Args:
            v1 (numpy.array): first vector
            v2 (numpy.array): second vector

        Returns:
            numpy.array: rotation matrix
        """
        V1 = numpy.array([v1])@self.unitvectors
        V2 = numpy.array([v2])@self.unitvectors
        N1 = V1/abs(V1)
        N2 = V2/abs(V2)
        Nm = (N1+N2)/abs(N1+N2)
        rotor = Nm*N1

        return rotor


DIMENSIONS = 10

def main():
    print("\nRotationMatrix")
    numpy.set_printoptions(precision=2, suppress=True)
    M = RotationMatrix(DIMENSIONS)
    locals().update(M.blades)
    print(M)
    print()

    v1 = numpy.array([random.randint(-10, 10) for _ in range(DIMENSIONS)])
    v2 = numpy.array([random.randint(-10, 10) for _ in range(DIMENSIONS)])

    V1 = v1@M.unitvectors
    V2 = v2@M.unitvectors

    #V1 = -(5 ^ M.e1) + (9 ^ M.e2) + (10 ^ M.e3) + (9 ^ M.e4) - (7 ^ M.e5)
    # V2 = -(8 ^ M.e1) - (6 ^ M.e2) + (7 ^ M.e3) - (2 ^ M.e4) - (4 ^ M.e5)

    N1 = V1/abs(V1)
    N2 = V2/abs(V2)
    print(f"V1: {V1}\nV2: {V2}")

    # R1 = M.single_rotation(1, 2, pi/4)
    # R2 = M.single_rotation(3, 4, pi/3)
    # R3 = M.single_rotation(2, 3, pi/4)
    # rotor = R1*R2
    # Mr = M.to_matrix(rotor)
    # print(Mr)
    # # rotor = 0.65328 - (0.65328 ^ e12) - (0.38268 ^ e23)
    # print(rotor)

    # V = 1.0*V1
    # v = v1.copy()
    # print(f"\nSingle rotation: {V}\nrotor:\t{rotor}\n")
    # for i in range(24):
    #     v = Mr@v
    #     print(f"\t{i}: {v}")

    #     V = rotor*V*~rotor
    #     print(f"\t{i}: {V}")

    # print()
    # print(V^V1, V)

    B = V1^V2
    # B = N1^N2
    rotor  = M.single_bivector_rotation(B, pi/4)
    matrix = M.to_matrix(rotor)

    V = 1.0*V1
    print(f"\nBivector rotation:\nbivector:\t{B}\nrotor:\t{rotor}\n\nvector:\t-: {V}")
    for i in range(8):
        V = rotor*V*~rotor
        print(f"\t{i}: {V}")

    print()
    print(V^V1, V)

    v = 1.0*v1
    print(
        f"\nBivector rotation matrix:\nmatrix:\t{matrix}\n\nvector:\t-: {v}")
    for i in range(8):
        v = matrix@v
        print(f"\t{i}: {v}")

    print()
    # print(V ^ V1, V)
    # sys.exit()

    rotor = M.align(v1, v2)
    V3 = rotor*V1*~rotor
    print(f"\nAlign Vectors:\n{V1}\n{V2}\n{V3}\n{V3^V2}")

    Ma = M.to_matrix(rotor)
    # print(Ma, numpy.linalg.det(Ma))
    v3 = Ma@v1
    VV3 = v3@M.unitvectors
    print(f"\n{v1}\n{v2}\n{v3}\n{VV3^V2}")


    # rotations = [(1,2,pi/4), (2,3,pi/3), (4,5, pi/6)]
    # rotor = M.blades['']
    # print(rotor)
    # for a1, a2, angle in rotations:
    #     R1 = M.single_rotation(a1, a2, angle)
    #     rotor = rotor * R1
    #     print(rotor, R1)

    # # rotor = M.combined_rotations(rotations)
    # print(rotor)

if __name__ == "__main__":
    main()