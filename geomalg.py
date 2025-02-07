from kingdon import Algebra, MultiVector
import numpy
import sys
from math import sin, cos

v1 = numpy.array([1,2,3,4])
v2 = numpy.array([-4,5,6, 7])
v3 = numpy.array([6,-7,8, 8])
# v2 = numpy.array(v1)
angle = numpy.pi/3
# angle = 0
print(v1)

alg = Algebra(4, 0, 0)
locals().update(alg.blades)
V1 = alg.purevector(values=v1, grade=1)
V2 = alg.purevector(values=v2, grade=1)
V3 = alg.purevector(values=v3, grade=1)

print(V1)
print(V1.grade(1).values())


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
print("\n\n##################")
idx = 1.0*alg.blades.grade(0)["e"]
for a in units:
    idx += a*0

print(idx, type(idx), idx.grades, idx.grade(1))
print(rotor, type(rotor), rotor.grades, rotor.grade(1))
print("##################")
# rotor = alg.blades.grade(0)["e"]
# print(rotor*e1*~rotor)
# rotor = idx
B = [rotor*a*~rotor for a in units]
for b in B:
    print(b.grade(1))
    # print(b.grade(1).values())
    # print(b.grade(1).asmatrix())

print()
for a in units:
    for b in B:
        v = b.grade(1) | a
        print(v, type(v))


M = [
            [(b.grade(1) | a).grade(0).values()[0] for b in B]
            for a in units]

# print([[[1.0], [], [], []], [[], [1.0], [], []], [[], [], [1.0], []], [[], [], [], [1.0]]])
print(M)

matrix = numpy.array(M)


print(matrix)
print()
print(rotor*V3*~rotor)

r = v3 @ matrix
print(r)

# print(alg.blades.grade(0)["e"])
# print(alg.blades.grade(0))

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

