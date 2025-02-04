from kingdon import Algebra, MultiVector
import numpy
import sys
from math import sin, cos

v1 = numpy.array([1,2,3])
v2 = numpy.array([-4,5,6])
v3 = numpy.array([6,-7,8])
# v2 = numpy.array(v1)
angle = numpy.pi/3
print(v1)

alg = Algebra(3, 0, 0)
locals().update(alg.blades)
V1 = alg.purevector(values=v1, grade=1)
V2 = alg.purevector(values=v2, grade=1)
V3 = alg.purevector(values=v3, grade=1)

print(V1)
print(V2)
print(V3)

print(alg.blades.grade(1))
units = alg.blades.grade(1).values()

B = V1^V2
B = B.normalized()
print(B)
R = cos(angle/2) - sin(angle/2)*B
# R = e**(-angle/2*B)  # enacts rotation by
rotor = R.normalized()
print(rotor)

print(rotor*V3*~rotor)
print()


print(rotor)
# print(rotor*e1*~rotor)
B = [rotor*a*~rotor for a in units]
for b in B:
    print(b.grade(1))
    # print(b.grade(1).values())
    # print(b.grade(1).asmatrix())

matrix = numpy.array([
            [(b.grade(1) | a).grade(0).values() for b in B]
            for a in units])
print(matrix)
print()
print(rotor*V3*~rotor)

print(v3 @ matrix)

sys.exit()

b = 2 * e12 + e1 + 4 * e123
v = 3 * e1 + 6 * e23 + e3 + e123 * e023
b.cp(v)
d = b * v

print(alg)
print(alg.blades)
for k in alg.blades:
    # if len(k) < 3:
    print(k, alg.blades[k])
print(alg.signature)
print(alg.p + alg.q + alg.r)
print
print(b, b.normalized())
print(d, ~d) # , d.normalized())



a1 = v1

