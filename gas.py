# pylint: disable=I
# pylint: disable=invalid-name
# pylint: disable=too-many-lines

"""
Ideal gas in n-dimensional box
"""
import itertools

import random
import math
from math import sin, cos, acos

import locale

import yaml
import numpy
from numpy import linalg
from scipy.spatial import ConvexHull # pylint: disable=no-name-in-module
from scipy.spatial import KDTree
# from numba import jit

from rotations import Rotations

locale.setlocale(locale.LC_ALL, '')

VMAX = 3
RADIUS = 1
MASS = 1
NBALLS = 20

__all__ = ['Box', 'Particle', 'Plane', 'Spring', 'Field', 'load', 'save', 'load_gas', 'FlatD3Shape']

class Field:
    """
    Different field equation.

    A field equation changes the particles speed based on position in the box
    and optionaly other parameters of the particle.
    """
    def __init__(self, box) -> None:
        """
        Creates field

        Args:
            box (Box): the box
        """
        self.box = box
        position = self.box.nullvector.copy()
        speed = self.box.nullvector.copy()
        self.dummy_ball = Particle(self.box, 1, 1, position, speed, 1, False, [0,0,0])
        self.equation = self.nofield

    def _template(self, position=None, ball=None):
        """
        Template field equation. Affects the balls speed based on its position

        The position parameter can be used to calculate the effect at the position.
        This can be used to display the field. The dummy particle can be used in the calculation.

        Args:
            position (numpy.array, optional): position. Defaults to None.
            ball (Particle, optional): A particle in the box. Defaults to None.

        Returns:
            numpy.array: the effect of the field, usually the speed difference, but can also be an other vector or matrix
        """
        if ball is None:
            ball = self.dummy_ball
        if position is None:
            position = ball.position

        dspeed = self.box.nullvector
        if ball is not self.dummy_ball:
            ball.speed += dspeed
        return dspeed

    def nofield(self, position=None, ball=None):
        """
        Dummy field equation, has no effect

        Args:
            position (numpy.array, optional): position. Defaults to None.
            ball (Particle, optional): A particle in the box. Defaults to None.

        Returns:
            numpy.array: change in speed, in this case the zero vector, so no change
        """
        if ball is None:
            ball = self.dummy_ball
        if position is None:
            position = ball.position

        dspeed = self.box.nullvector
        if ball is not self.dummy_ball:
            ball.speed += dspeed
        return dspeed

    def rotate(self, position=None, ball=None):
        """
        Applies matrix rotation to the particles speed.
        Causes all the particles to move in circles.

        Args:
            position (numpy.array, optional): position. Defaults to None.
            ball (Particle, optional): A particle in the box. Defaults to None.

        Raises:
            ValueError: Currently only works for a bo with two dimensions

        Returns:
            numpy.array: change in speed
        """
        if ball is None:
            ball = self.dummy_ball
        if position is None:
            position = ball.position

        if self.box.dimensions > 2:
            raise ValueError("Box dimensions must be 2 to use the rotation field")

        theta = math.radians(ball.mass*ball.speed.dot(ball.speed)/10)
        # theta = math.radians(5)
        c = math.cos(theta)
        s = math.sin(theta)
        M = numpy.array(((c, -s), (s, c)))

        dspeed = numpy.matmul(M, ball.speed)
        if ball is not self.dummy_ball:
            ball.speed = dspeed
        return dspeed

    def sinkR(self, position=None, ball=None):
        """
        Field with a sink 1/R
        """
        if ball is None:
            ball = self.dummy_ball
        if position is None:
            position = ball.position

        dspeed = self.box.nullvector
        center = self.box.box_sizes / 2
        Rmin = 20
        v0 = position - center
        v0dot = v0.dot(v0)
        if abs(v0dot) > Rmin*Rmin:
            u0 = v0/math.sqrt(v0dot)
            dspeed = -50*u0/math.sqrt(v0dot)

        if ball is not self.dummy_ball:
            ball.speed += dspeed
        return dspeed

    def sinkRR(self, position=None, ball=None):
        """
        Field with a sink 1/(R**2)
        """
        if ball is None:
            ball = self.dummy_ball
        if position is None:
            position = ball.position

        dspeed = self.box.nullvector
        center = self.box.box_sizes / 2
        Rmin = 20
        v0 = position - center
        v0dot = v0.dot(v0)
        if abs(v0dot) > Rmin*Rmin:
            u0 = v0/math.sqrt(v0dot)
            dspeed = -2000*u0/v0dot

        if ball is not self.dummy_ball:
            ball.speed += dspeed
        return dspeed

    def rotate_flow(self, position=None, ball=None):
        """
        Rotates alle particles around the center (axis) of the box.

        Args:
            position (numpy.array, optional): position. Defaults to None.
            ball (Particle, optional): the particle to affect. Defaults to None.

        Returns:
            numpy.array: Vector
        """
        if ball is None:
            ball = self.dummy_ball
        if position is None:
            position = ball.position

        center = self.box.box_sizes / 2
        Rc = position - center

        u0 = Rc/math.sqrt(Rc.dot(Rc))
        vector = u0.copy()
        vector[0] = u0[1]
        vector[1] = -u0[0]
        # vector = numpy.array([u0[1], -u0[0]])

        dspeed = math.sqrt(ball.speed.dot(ball.speed)) * vector
        if ball is not self.dummy_ball:
            ball.speed = dspeed
        return vector


class Box:
    """
    n-dimensional box to contain particles
    """
    D1 = X = 0
    D2 = Y = 1
    D3 = Z = 2
    D4 = 3
    D5 = 4

    def __init__(self, box_sizes, torus=False) -> None:
        """
        Create n-dimension box

        Args:
            box_sizes (list of floats): size of the box
        """
        # shape of the box
        self.box_sizes = numpy.array(box_sizes, dtype=float)
        self.torus = torus
        self.color = (200,200,200)
        # calculated properties based on box_sizes
        self.dimensions = len(self.box_sizes)
        self.onevector = numpy.array([1.0]*self.dimensions)
        self.nullvector = self.onevector * 0
        self.vertices = []
        self.edges = []
        self.axis = []
        self.planes = []
        self._get_vertices()
        self._get_edges()
        self._get_axis()
        self._get_planes()
        self.center = sum(self.vertices)/len(self.vertices)
        # physical properties
        self.field = None
        self.gravity = self.nullvector.copy()
        self.friction = 0.0
        self.interaction = 0.0
        self.merge = False
        self.interaction_power = 2
        # content of the box
        self.walls = []
        self.particles = []
        self.merged_particles = []
        self.delete_particles = []
        self.delete_trails = []
        self.springs = []
        self.rods = []
        # dynamic properties
        energies = ["KE", "EE", "PE", "SE"]
        self.energy = {e : 0.0 for e in energies}
        self.momentum = self.nullvector.copy()
        self._normal_momentum = 0
        self.ticks = 0
        # properties for optimization by KDTree
        self._max_radius = 0
        self._min_radius = 0
        self._avg_radius = 0
        self.interaction_radius = max(self.box_sizes)
        self.optimized_collisions = True
        self.optimized_interaction = True
        self._kdtree = None
        self._neighbors = []
        self.interaction_neighbors = 10
        # properties for rotations of content
        self.rotations = []
        self._rotation_matrix = numpy.eye(self.dimensions)
        # self._rotor = RotationMatrix(self.dimensions)
        self._rotor = Rotations(self.dimensions)
        # other properties
        self.calculate_energies = False
        self.trail = 0
        self.skip_trail = 1
        self.object = None
        self.simple_hole_bounce = False

    def get_radi(self, interaction_factor=1, neighbor_count=None):
        """
        calculates parameters for KDtree optimization

        Args:
            interaction_factor (int, optional): factor for interaction radius. Defaults to 1.
            neighbor_count (int, optional): number of neighbors to use. Defaults to None.
        """
        try:
            self._max_radius = max([ball.radius for ball in self.particles])
        except ValueError:
            self._max_radius = 0
        try:
            self._min_radius = min([ball.radius for ball in self.particles])
        except ValueError:
            self._min_radius = 0
        try:
            self._avg_radius = sum([ball.radius for ball in self.particles])/len(self.particles)
        except ZeroDivisionError:
            self._avg_radius = 0
        # if len(self.particles) > 25:
        #     self.interaction_radius = interaction_factor*(self.interaction**(1/self.interaction_power))
        # else:
        #     self.interaction_radius = max(self.box_sizes)
        nballs = 1
        if self.particles:
            nballs = len(self.particles)
        self.interaction_radius = interaction_factor * max(self.box_sizes) / (nballs**(1/self.dimensions))
        if neighbor_count is None:
            neighbor_count = max(10, int(0.1*len(self.particles)))

        self.interaction_neighbors = min(len(self.particles), neighbor_count)

    def _get_vertices(self):
        """
        Calculates the vertices
        """
        # get unit cube coordinates for dimensions of box
        unit_cube = numpy.array(list(itertools.product([0,1], repeat=self.dimensions)))
        # vector multiply with box sizes
        self.vertices = unit_cube*self.box_sizes

    def _get_edges(self):
        """
        Calculates the edges
        """
        for i, _ in enumerate(self.vertices):
            for j in range(i+1, len(self.vertices)):
                v1 = self.vertices[i]
                v2 = self.vertices[j]
                c = 0
                for k, _ in enumerate(v1):
                    if v1[k] == v2[k]:
                        c += 1
                if c == self.dimensions-1:
                    self.edges.append((i,j))

    def _get_axis(self):
        """
        Calculates the axis

        Returns:
            list of numpy.array: the axis
        """
        self.axis = []
        for i, size in enumerate(self.box_sizes):
            _axis = self.nullvector.copy()
            _axis[i] = size
            self.axis.append(_axis)

        return self.axis

    def _get_planes(self):
        coordinates = self.box_sizes
        for i, x in enumerate(coordinates):
            points = [p for p in self.vertices if p[i] == x]
            point = sum(points)/len(points)
            normal = self.nullvector.copy()
            normal[i] = 1
            plane = Plane(self, point=point, normal=normal)
            plane.color = self.color
            #points=points[:self.dimensions])
            # plane.point = point
            self.planes.append(plane)

        coordinates = self.nullvector
        for i, x in enumerate(coordinates):
            points = [p for p in self.vertices if p[i] == x]
            point = sum(points)/len(points)
            normal = self.nullvector.copy()
            normal[i] = 1
            plane = Plane(self, point=point, normal=normal) # points=points[:self.dimensions])
            plane.color = self.color
            # plane.point = point
            self.planes.append(plane)

    def __str__(self) -> str:
        return str(self.box_sizes)

    def out(self):
        """
        dumps all properties

        Returns:
            dict: the dump
        """
        box = {}
        box["sizes"] = [float(f) for f in self.box_sizes] # list(self.box_sizes)
        box['torus'] = self.torus
        box['color'] = [int(i) for i in self.color]
        box['merge'] = self.merge
        box['trail'] = self.trail
        box["gravity"] = [float(f) for f in self.gravity] # list(self.gravity)
        box["friction"] = float(self.friction)
        box["interaction"] = float(self.interaction)
        box["particles"] = [ball.out() for ball in self.particles]
        box["springs"] = [spring.out() for spring in self.springs]
        box["planes"] = [plane.out() for plane in self.planes[2*self.dimensions:]]
        box["interaction_power"] = self.interaction_power
        box["optimized_collisions"] = self.optimized_collisions
        box["optimized_interaction"] = self.optimized_interaction
        box["neighbor_count"] = self.interaction_neighbors
        box['simple_hole_bounce'] = self.simple_hole_bounce
        box['rotations'] = []
        for (v1, v2, angle) in self.rotations:
            rotation = {'vector1': [float(v) for v in v1], 'vector2':  [float(v) for v in v2], 'angle': float(angle)}
            box['rotations'].append(rotation)
        # box['rotations'] = [{'fixed_plane' : [int(x1), int(x2)], 'angle': float(angle)} for (v1, 22, angle) in self.rotations]

        output = {"box": box}

        return output

    def random_position(self, edge=0):
        """
        Gives random position in the box

        Args:
            edge (int, optional): ensures position is not on wall

        Returns:
            numpy.array: position
        """
        V = []
        for max_ in self.box_sizes:
            pos = edge + (max_ - edge*2)*random.random()
            V.append(pos)
        position = numpy.array(V)
        return position

    def random(self, size=1):
        """
        Gives random vector

        Args:
            size (int, optional): Absolute size of vector. Defaults to 1.

        Returns:
            numpy.array: vector
        """
        V = numpy.array([random.random()-0.5 for d_ in self.box_sizes])
        L = math.sqrt(V.dot(V))
        vector = size*V/L
        return vector

    def resize(self, new_sizes):
        """
        Resizes the box

        Args:
            new_sizes (list of floats): new size of the box
        """
        self.box_sizes[0:len(new_sizes)] = new_sizes
        self._get_vertices()
        self._get_edges()
        self._get_axis()
        planes = self.planes[2*self.dimensions:]
        self.planes = []
        self._get_planes()
        for plane in planes:
            # pylint: disable=protected-access
            plane.box_intersections = plane._box_intersections()
            plane.edges = plane._edges()
        self.planes.extend(planes)
        self.center = sum(self.vertices)/len(self.vertices)
        self.ticks = 1
        self._normal_momentum = 0

    def project3d(self, position, axis=3):
        """
        projects extra dimension onto 3D in perspective

        Args:
            position (numpy.array): the position to project
            axis (int, optional): axis to project. Defaults to 3.

        Returns:
            numpy.array: projected position
        """
        if self.dimensions == 3:
            return position
        elif self.dimensions < 3:
            projection = numpy.zeros(3)
            projection[self.Z] = 600
            projection[:len(position)] = position
            return projection

        projection = position.copy()
        min_ = 0.05
        max_ = 0.95
        A = (max_ - min_) / self.box_sizes[axis]
        B = min_

        pos_center_3d = projection[:3] - self.center[:3]
        w = projection[axis]
        f = A*w + B

        pos = pos_center_3d*f + self.center[:3]
        projection[:3] = pos
        return projection

    def set_rotations(self, rotations=None):
        """
        calculates rotation matrix

        Args:
            rotations (list[(vector1: numpy.array, vector2: numpy.array: int, angle: float),...]): set of single rotations
        """
        if rotations is None:
            rotations = self.rotations

        rotor = self._rotor.identity
        for (v1, v2, angle) in rotations:
            V1 = self._rotor.to_ga_vector(v1)
            V2 = self._rotor.to_ga_vector(v2)
            rotor_ = self._rotor.rotation(V1, V2, angle)
            rotor = rotor*rotor_

        R = self._rotor.to_matrix(rotor)

        self._rotation_matrix = R

    def rotate(self):
        """
        rotates everything in the box around the center

        Args:
            rotations (list[(vector1: numpy.array, vector2: numpy.array: int, angle: float),...]): set of single rotations

        Returns:
            numpy.array: rotation matrix
        """
        # R = self._rotor.combined_rotations(rotations)
        # rotor = self._rotor.blades[""]
        # rotor = 1
        # for (v1, v2, angle) in rotations:
        #     V1 = self._rotor.to_ga_vector(v1)
        #     V2 = self._rotor.to_ga_vector(v2)
        #     rotor_ = self._rotor.rotation(V1, V2, angle)
        #     rotor = rotor*rotor_

        # R = self._rotor.to_matrix(rotor)
        R = self._rotation_matrix

        for plane in self.planes[2*self.dimensions:]:
            cpos = plane.point - self.center
            plane.point = self.center + cpos @ R
            normal = plane.unitnormal
            plane.unitnormal = normal @ R

            # pylint: disable=protected-access
            plane._set_params()

            for i, hole in enumerate(plane.holes):
                (point, radius) = hole
                cpos = point - self.center
                point = self.center + cpos @ R
                hole = (point, radius)
                plane.holes[i] = hole

        for ball in self.particles:
            cpos = ball.position - self.center
            ball.position = self.center + cpos @ R
            speed = ball.speed
            ball.speed = speed @ R

    def _rotation_matrix_3d(self, α, β, γ): # pylint: disable=non-ascii-name
        """
        rotation matrix of α, β, γ radians around x, y, z axes (respectively)
        """
        sα, cα = sin(α), cos(α)
        sβ, cβ = sin(β), cos(β)
        sγ, cγ = sin(γ), cos(γ)
        return (
            (cβ*cγ, -cβ*sγ, sβ),
            (cα*sγ + sα*sβ*cγ, cα*cγ - sγ*sα*sβ, -cβ*sα),
            (sγ*sα - cα*sβ*cγ, cα*sγ*sβ + sα*cγ, cα*cβ)
        )

    def rotate_3d(self, α, β, γ): # pylint: disable=non-ascii-name
        """
        Rotates content of box around x, y, z axes
        """
        if self.dimensions < 3:
            return

        for plane in self.planes[2*self.dimensions:]:
            cpos = plane.point[:3] - self.center[:3]
            plane.point[:3] = self.center[:3] + cpos.dot(self._rotation_matrix_3d(α, β, γ))
            normal = plane.unitnormal[:3]
            plane.unitnormal[:3] = normal.dot(self._rotation_matrix_3d(α, β, γ))
            # pylint: disable=protected-access
            plane._set_params()

            for hole in plane.holes:
                (point, _) = hole
                cpos = point[:3] - self.center[:3]
                point[:3] = self.center[:3] + cpos.dot(self._rotation_matrix_3d(α, β, γ))

        for ball in self.particles:
            cpos = ball.position[:3] - self.center[:3]
            ball.position[:3] = self.center[:3] + cpos.dot(self._rotation_matrix_3d(α, β, γ))
            speed = ball.speed[:3]
            ball.speed[:3] = speed.dot(self._rotation_matrix_3d(α, β, γ))

    def rotate_axis_3d(self, axis, rad):
        """
        Rotates context of the box around an axis

        Args:
            axis (int): The axis to rotate around: 0: X, 1: Y, 2: Z
            rad (float): radians to rotate
        """
        if self.dimensions < 3:
            return
        rotation = self.nullvector.copy()[:3]
        rotation[axis] = rad
        self.rotate_3d(*rotation)

    def volume(self) -> float:
        """
        Calculates the volume of the box

        Returns:
            float: volume of the box
        """
        V = numpy.product(self.box_sizes)
        return V

    def area(self) -> float:
        """
        Calculates the area of the box

        Returns:
            float: area of the box
        """
        V = numpy.product(self.box_sizes)
        A = 2*sum([V/s for s in self.box_sizes])
        return A

    def add_particle(self,mass=MASS, radius=RADIUS, position=None, speed=None, charge=0, fixed=False, color=None):
        """
        Adds a particle to the box

        Args:
            mass (float, optional): mass. Defaults to MASS.
            radius (float, optional): radius. Defaults to RADIUS.
            position (list of float, optional): position. Defaults to None for random position
            speed (list of float, optional): speed. Defaults to None for random speed
            charge (int, optional): charge. Defaults to 0.
            color (tuple RGB value, optional): color. Defaults to None for random color

        Returns:
            Particle: particle in the box
        """
        if position is None:
            position = []
        position = numpy.array(position)
        rpos = []
        for size in self.box_sizes:
            l = [int(radius), int(size - radius)]
            l.sort()
            try:
                X = random.randrange(*l)*1.0
            except ValueError:
                X = 0
            rpos.append(X)
        # rpos = [random.randrange(int(radius), int(x - radius))*1.0 for x in self.box_sizes]
        # position.extend(rpos[len(position):])
        position = numpy.append(position, rpos[len(position):])

        if speed is None:
            speed = []
        speed = numpy.array(speed)
        rspeed = [random.randrange(-VMAX,VMAX)*1.0 for dummy in range(self.dimensions)]
        # speed.extend(rspeed[len(speed):])
        speed = numpy.append(speed, rspeed[len(speed):])

        if color is None:
            color = (random.randrange(256), random.randrange(256), random.randrange(256))
        particle = Particle(self, mass, radius, position, speed, charge, fixed, color)
        # print(particle.out())
        self.particles.append(particle)
        return particle

    def _displacement(self, pos1, pos2):
        """
        Calculates displacement vector between two position vectors.
        Accounts for wrapping around if self.torus == True.

        Args:
            pos1 (numpy.array): position 1
            pos2 (numpy.array): position 2

        Returns:
            numpy.array: minimum vector from position 1 to 2
        """
        if not self.torus:
            return pos1 - pos2
        else:
            dpos = pos1 - pos2
            # P1 = [pos1 - s for s in self.axis] + [pos1 + s for s in self.axis]
            P1 = [pos1 + s for s in self.vertices]
            P2 = [pos2 + s for s in self.vertices]
            # print(P1)

            dpos_dot = dpos.dot(dpos)
            for p1 in P1:
                for p2 in P2:
                    dpos2 = p1 - p2
                    dpos_dot2 = dpos2.dot(dpos2)
                    if dpos_dot2 < dpos_dot:
                        dpos = dpos2
                        dpos_dot = dpos_dot2
            return dpos

    def displacement(self, pos1, pos2):
        """
        Calculates displacement vector between two position vectors.
        Accounts for wrapping around if self.torus == True.

        Args:
            pos1 (numpy.array): position 1
            pos2 (numpy.array): position 2

        Returns:
            numpy.array: minimum vector from position 1 to 2
        """
        dpos = pos1 - pos2
        if self.torus:
            dpos = numpy.array([math.remainder(d,s) for (d,s) in zip(dpos, self.box_sizes/2)])

        return dpos

    def set_gravity(self, strength, direction=None):
        """
        Sets the gravity

        Args:
            strength (float): strength of the gravity
            direction (list of float, optional): direction of the gravity. Defaults to None.

        Returns:
            numpy.array: vector of the gravity
        """
        self.gravity = self.nullvector.copy()
        if strength == 0:
            return self.gravity

        if direction is None:
            direction = self.nullvector.copy()
            try:
                direction[1]=-1
            except IndexError:
                pass

        direction = numpy.array(direction)
        D2 = direction.dot(direction)
        if D2 > 0:
            direction = direction/math.sqrt(D2)
        self.gravity = strength * direction

        return self.gravity

    def fall(self, particle):
        """
        Applies gravity to particles speed

        Args:
            particle (Particle): the particle to affect

        Returns:
            Particle: affected particle
        """
        particle.speed += self.gravity
        return particle

    def set_friction(self, friction):
        """
        Sets the friction

        Args:
            friction (float): friction

        Returns:
            float: the friction
        """
        self.friction = friction
        return self.friction

    def slide(self, particle):
        """
        Applies friction to particles speed

        Args:
            particle (Particle ): the particle to affect

        Returns:
            Particle: the affected particle
        """
        particle.speed -= particle.speed*self.friction
        return particle

    def set_interaction(self, interaction, power=2):
        """
        Sets the interaction between particles

        Args:
            interaction (float): the interaction

        Returns:
            float: the interaction
        """
        self.interaction = interaction
        self.interaction_power = power
        return self.interaction

    def avg_momentum(self):
        """
        Calculates the average momentum of all particles in the box,
        including the momentum of the box due to collisions

        Returns:
            numpy.array: momentum vector
        """
        if len(self.particles) == 0:
            return self.nullvector
        tot_momentum = sum([ball.momentum for ball in self.particles]) + self.momentum
        return tot_momentum/len(self.particles)

    def avg_position(self):
        """
        Calculates the average positon of all particles in the box

        Returns:
            numpy.array: position vector
        """

        if len(self.particles) == 0:
            return self.nullvector
        tot_position = sum([ball.position for ball in self.particles])
        return tot_position/len(self.particles)

    def pressure(self):
        """
        Calculates the pressure on the box due to collisions of particles with the box

        Returns:
            float: pressure
        """
        try:
            return self._normal_momentum/(self.ticks*self.area())
        except ZeroDivisionError:
            return self._normal_momentum/self.area()

    def stop_all(self):
        """
        Stop all balls
        """
        for ball in self.particles:
            ball.speed = self.nullvector.copy()
        self.momentum = self.nullvector.copy()

    def center_all(self, fixed=True):
        """
        Move center of mass to center of box and set total momentum to zero
        """
        if len(self.particles) > 0:
            total_mass = sum(ball.mass for ball in self.particles)
            center_of_mass = sum(ball.mass*ball.position for ball in self.particles)/total_mass
            dpos = self.center - center_of_mass

            # avg_speed = sum(ball.speed for ball in self.particles)/len(self.particles)
            avg_momentum = sum(ball.momentum for ball in self.particles)/len(self.particles)
            for ball in self.particles:
                if fixed and ball.fixed:
                    continue
                ball.position += dpos
                # ball.speed -= avg_speed
                ball.speed -= avg_momentum/ball.mass

            self.momentum = self.nullvector.copy()

    def kick_all(self):
        """
        Kick all balls in random direction with 10% of average absolute speed with a minimum of 3
        """
        S = sum(abs(ball.speed) for ball in self.particles)
        ms = 0.1 * math.sqrt(S.dot(S))
        if ms < 0.5:
            ms = 0.5
        for ball in self.particles:
            ball.speed += self.random(ms)

    def go(self, steps=1):
        """
        Calculate speeds and move all particles to next position

        Args:
            steps (int): number of interations

        Returns:
            boolean: true if particles collides into each other
        """
        bounced = False
        for _ in range(steps):
            self.ticks += 1
            self._get_kdtree()
            bounced |= self._speeds()
            if self.calculate_energies:
                self._energies()
            self._move_all()

        return bounced

    def _speeds(self):
        bounced = False

        # apply springs
        for spring in self.springs:
            spring.pull()

        # calculate speeds
        for i, ball in enumerate(self.particles):
            # interaction based on charge
            if self.interaction != 0:
                ball.interact()
            # gravity
            if self.gravity.any() != 0:
                self.fall(ball)
            # apply field
            if self.field is not None:
                self.field.equation(ball=ball)
            # friction
            if self.friction != 0:
                self.slide(ball)
            # bounce against planes
            ball.bounce()

        # collisions between balls
        pairs = []
        if self.optimized_collisions and self._kdtree:
            pairs = self._kdtree.query_pairs(2*self._max_radius)
        else:
            pairs = itertools.combinations(range(len(self.particles)), 2)

        for i, j in pairs:
            try:
                ball1 = self.particles[i]
                ball2 = self.particles[j]
            except IndexError:
                continue
            if self.merge:
                if ball1.merge(ball2):
                    bounced = True
                    if ball1.radius > self._max_radius:
                        self._max_radius = ball1.radius
                    # self._get_kdtree()
            else:
                if ball1.collide(ball2):
                    bounced = True

        # apply rods
        for rod in self.rods:
            rod.pull()

        return bounced

    def _energies(self):
        # calculate energies
        self.energy["KE"] = sum(ball.energy for ball in self.particles)
        self.energy["EE"] = 2 * self.energy["KE"]/self.dimensions # https://en.wikipedia.org/wiki/Kinetic_theory_of_gases#Pressure_and_kinetic_energy
        if self.interaction != 0:
            self.energy["PE"] = sum(ball.potential_energy for ball in self.particles)
        else:
            self.energy["PE"] = 0
        self.energy["SE"] = sum(spring.energy for spring in self.springs)

    def _move_all(self):
        # move all balls
        for ball in self.particles:
            _ = ball.move()

    def _get_kdtree(self):
        if not self.particles:
            return
        positions = [ball.position for ball in self.particles]
        sizes = None
        if self.torus:
            sizes = self.box_sizes
        self._kdtree = KDTree(positions, boxsize=sizes)
        self._neighbors  = self._kdtree.query_ball_tree(self._kdtree, self.interaction_radius)

class Plane:
    """
    Plane
    """
    def __init__(self, box: Box, normal=None, point=None, points=None, color=None, reflect=True) -> None:
        """
        Creates plane

        Args:
            box (Box): the Box
            normal (numpy.array, optional): the normal vector. Defaults to None.
            point (numpy.array, optional): a point on the plane. Defaults to None.
            points (list of numpy.array, optional): point on the plane. Defaults to None.
            reflect (boolean): Does the plane reflect the particles?
                If False the holes will reflect and act as disks
        """
        self.box = box

        if normal is not None and point is not None:
            if len(normal) != self.box.dimensions or len(point) != self.box.dimensions:
                raise ValueError("wrong size")
            else:
                normal = numpy.array(normal)
                self.unitnormal = normal/math.sqrt(normal@normal)
                self.point = numpy.array(point)
        elif points is not None:
            if numpy.shape(points) != (self.box.dimensions, self.box.dimensions):
                raise ValueError("wrong size")
            else:
                if point is None:
                    point = numpy.array(sum(point for point in points)/len(points))
                points = numpy.array(points)
                self.unitnormal = self._get_normal(points)
                self.point = numpy.array(point)
                # self._test_normal(points)
        else:
            raise TypeError("missing required argument")

        if color is None:
            color = [0,0,128]
        self.color = color

        self._set_params()
        self.reflect = reflect
        self.holes = []

        self.object = None

    def _set_params(self):
        self.D = self.unitnormal @ self.point
        self.box_intersections = self._box_intersections()
        self.edges = self._edges()
        self._projected_hull, self._projection_axis = self._get_projected_hull(self.box_intersections)

    def _get_projected_hull(self, points):
        hull = None
        projection_coordinate = 0
        if len(points) < 3:
            return (hull, projection_coordinate)

        # project points on X,Y or Z plane as long as i has a non zero value for the normal
        axis = 0
        for axis, x in enumerate(self.unitnormal):
            if x != 0:
                break

        projected = [numpy.delete(p, axis) for p in points]

        hull = ConvexHull(projected)
        projection_axis = axis
        return (hull, projection_axis)

    def add_hole(self, point, radius):
        """
        adds a hole in the plane, when the point is not on the plane
        it is projected on the plane using the normal

        Args:
            point (numpy.array): point on the plane
            radius (float): radius of the hole

        Returns:
            tuple: (point, radius)
        """
        point = self.project_point(point)
        hole = (point, radius)
        self.holes.append(hole)

        return hole

    def _get_normal(self, points):
        """
        Calculates the normal vector from points on the plane

        Args:
            points (list of numpy.array): the points

        Returns:
            numpy.array: unit normal vector
        """
        shape = numpy.shape(points)
        ones = numpy.ones(shape)
        i = 0
        points = [p for p in points]
        while linalg.det(points) == 0 and i < 100:
            points += ones
            i += 1

        normal = linalg.solve(points, self.box.onevector)

        unitnormal = normal/math.sqrt(normal@normal)
        return unitnormal

    def _test_normal(self, points):
        p = points[-1]
        for q in points:
            d = p - q
            print(d @ self.unitnormal)
            p = q

    def __str__(self) -> str:
        parts = []
        for i, p in enumerate(self.unitnormal):
            part = f"{p}*x{i}"
            parts.append(part)

        equation = " + ".join(parts)
        equation += f" = {self.D}"
        # pstr = "plane:\n normal: {}\n point: {}\n D: {}\nequation: {}".format(self.unitnormal, self.point, self.D, equation)
        pstr = f"n:{self.unitnormal},\tp:{self.point}"
        return pstr

    def out(self):
        """
        dumps all properties

        Returns:
            dict: the dump
        """
        plane = {}
        plane["normal"] = [float(f) for f in self.unitnormal]
        plane["point"] = [float(f) for f in self.point]
        plane["color"] = [int(i) for i in self.color]
        plane["reflect"] = self.reflect
        plane["holes"] = []
        for hole in self.holes:
            (point, radius) = hole
            point = [float(f) for f in point]
            plane["holes"].append({"point": point, "radius": float(radius)})

        return plane

    def intersection(self, planes):
        """
        Calculates intersection point between planes

        Args:
            planes (list of Plane): the planes

        Returns:
            numpy.array: the intersection point
        """
        if self not in planes:
            planes.append(self)

        if len(planes) != self.box.dimensions:
            raise ValueError("not enough planes to calculate intersection")
        normals = [p.unitnormal for p in planes]
        Ds = [p.D for p in planes]

        intersection = linalg.solve(normals, Ds)

        return intersection

    def distance(self, point):
        """
        Calculates distance between point and the plane

        Args:
            point (numpy.array): the point

        Returns:
            float: the distance
        """
        point = numpy.array(point)
        v = point - self.point
        return v @ self.unitnormal

    def _on_plane(self, point):
        hull = self._projected_hull
        axis = self._projection_axis
        point = numpy.delete(point, axis)
        # A is shape (f, d) and b is shape (f, 1).
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]

        eps = numpy.finfo(numpy.float32).eps

        # The hull is defined as all points x for which Ax + b <= 0.
        # We compare to a small positive value to account for floating
        # point issues.
        #
        # Assuming x is shape (m, d), output is boolean shape (m,).
        return numpy.all(numpy.asarray(point) @ A.T + b.T < eps, axis=1)

    def intersect_line(self, point, vector):
        """
        Calculates intersection point between line and the plane

        Args:
            point (numpy.array): a point on the line
            vector (numpy.array): the vector of the line

        Returns:
            numpy.array: the intersection point
        """
        point = numpy.array(point)
        dn = vector @ self.unitnormal
        if dn == 0:
            # line parallel to or on plane, no solution
            return None
        dt = (self.point - point) @ self.unitnormal
        d = dt / dn

        return point + vector * d

    def project_point(self, point):
        """
        projects point on plane, using planes normal vector

        Args:
            point (numpy.array): point to project on to plane

        Returns:
            numpy.array: the projected point
        """
        point = numpy.array(point)
        projected_point = self.intersect_line(point, self.unitnormal)
        return projected_point

    def _box_intersections(self):
        """
        Calculates the intersection points with the box

        Returns:
            list of numpy.array: the intersection points
        """
        points = []
        for (i, j) in self.box.edges:
            v1 = self.box.vertices[i]
            v2 = self.box.vertices[j]
            V = v1 - v2
            intersection = self.intersect_line(v1, V)
            if intersection is not None and numpy.all(intersection <= self.box.box_sizes) and numpy.all(intersection >= self.box.nullvector):
                skip = False
                for ins in points:
                    if numpy.allclose(intersection, ins):
                        skip = True
                        break
                if not skip:
                    points.append(intersection)

        # hull, x = self._get_projected_hull(points)
        # if hull is None:
        #     return points
        # vertices = hull.vertices

        # sorted = []
        # for v in vertices:
        #     sorted.append(points[v])
        # points = sorted

        # return points

        points = self._sort_convexhull(points)

        return points

    def _sort_by_distance(self, points):
        if len(points) > 0:
            sorted_ = []
            imin = 0
            while len(points) > 0:
                p0 = points.pop(imin)
                sorted_.append(p0)
                dmin2 = max(self.box.box_sizes) ** 2
                for i, p1 in enumerate(points):
                    d01 = (p1-p0) @ (p1-p0)
                    if d01 < dmin2:
                        dmin2 = d01
                        imin = i

            points = sorted_

        return points

    def _sort_by_angles(self, points):
        if len(points) > 0:
            p1 = points[0] - self.point
            pt = []

            for p in points:
                p2 = p - self.point
                p1p2 = math.sqrt(p1@p1) * math.sqrt(p2@p2)
                p1dp2 = p1@p2

                cos_ = 0
                if p1p2 != 0:
                    cos_ = p1dp2/p1p2

                if cos_ >= -1 and cos_ <= 1:
                    theta = 360*acos(cos_)/(2*math.pi)
                    pt.append([theta, cos_, p])

            points = [p[1] for p in pt]

        return points

    def _sort_convexhull(self, points):
        if len(points) < 3:
            return points

        # project points on X,Y or Z plane as long as i has a non zero value for the normal
        i = 0
        for i, x in enumerate(self.unitnormal):
            if x != 0:
                break

        projected = [numpy.delete(p, i) for p in points]

        hull = ConvexHull(projected)
        vertices = hull.vertices

        sorted_ = []
        for v in vertices:
            sorted_.append(points[v])

        return sorted_

    def _sort_by_planes(self, points):
        sorted_ = []

        if self.box.dimensions < 4:
            D = 1
        else:
            D = self.box.dimensions - 2

        i = 0
        while len(points) > 0:
            p0 = points.pop(i)
            sorted_.append(p0)
            for i, p1 in enumerate(points):
                # if numpy.any(p0==p1):
                if len([t for t in (p0==p1) if t]) == D:
                    sorted_.append(p1)
                    break

        points = sorted_
        return points

    def _edges__(self):
        """
        Calculates the edges

        Returns:
            list of (int, int): list of edges
        """
        edges = []
        points = self.box_intersections
        for i, p0 in enumerate(points):
            for j, p1 in enumerate(points):
                if numpy.all(p0==p1):
                    continue
                if len([t for t in p0==p1 if t]) == self.box.dimensions - 2:
                    if (i,j) not in edges and (j,i) not in edges:
                        edges.append((i,j))

        return edges

    def _edges(self):
        """
        Calculates the edges

        Returns:
            list of (int, int): list of edges
        """
        edges = []
        size = len(self.box_intersections)

        for i, _ in enumerate(self.box_intersections):
            edge = (i, (i-1) % size)
            edges.append(edge)

        return edges

    def pass_through(self, ball): # pylint: disable=unused-argument
        """
        dummy pass through method
        """
        return False


class _Membrane(Plane):
    """
    conditionals lets particles pass though the plan
    """
    # pylint: disable=missing-function-docstring
    def __init__(self, box: Box, normal=None, point=None, points=None) -> None:
        self.filter = self.no_filter
        self.hole_size = 15
        self.max_speed = 4
        self._bounced = 0
        super().__init__(box, normal=normal, point=point, points=points)

    def no_filter(self, ball):  # pylint: disable=unused-argument
        return False

    def pass_through(self, ball):
        # res = super().pass_through(ball)
        res = self.filter(ball)
        return res

    def mass_filter(self, ball):  # pylint: disable=unused-argument
        return False

    def size_filter(self, ball):
        if ball.radius < self.hole_size:
            return True
        return False

    def speed_filter(self, ball):
        if ball.speed < self.max_speed:
            return True
        return False

    def dynamic_speed_filter(self, ball):
        res = False
        self._bounced += 1
        speed = math.sqrt(ball.speed@ball.speed)
        if speed < self.max_speed:
            res = True

        self.max_speed = speed + (self.max_speed * (self._bounced - 1))/self._bounced
        return res

    def maxwells_demon_filter(self, ball):
        res = False
        self._bounced += 1

        direction = ball.speed @ self.unitnormal
        speed2 = ball.speed @ ball.speed
        speed = math.sqrt(speed2)

        max_speed2 = (self.max_speed * self.max_speed)/9
        if direction == 0:
            res = False
        elif direction > 0:
            if speed2 > max_speed2:
                res = True
        elif direction < 0:
            if speed2 < max_speed2:
                res = True
        else:
            res = False

        if res:
            self.max_speed = (speed + self.max_speed * (self._bounced - 1))/self._bounced

        return res

    def out(self):
        """
        dumps all properties

        Returns:
            dict: the dump
        """
        membrane = super().out()
        membrane['pass_through_function'] = self.filter.__name__
        membrane['hole_size'] = self.hole_size
        membrane['max_speed'] = self.max_speed
        return membrane


class Particle:
    """
    Particle
    """
    def __init__(self, box: Box, mass: float, radius: float, position: list, speed: list, charge: float, fixed: bool, color: list) -> None:
        """
        Creates particle

        Args:
            box (Box): box particle is contained in
            mass (float): mass
            radius (float): radius
            position (list of float): position vector
            speed (list of floate): speed vector
            charge (int): charge
            color (tuple RGB value): color
        """
        self.box = box
        self.mass = mass
        self.radius = radius
        self.position = numpy.array(position, dtype=float)
        self.positions = []
        self.speed = numpy.array(speed, dtype=float)
        self.speeds = []
        self.charge = charge
        self.color = tuple(color)
        self.fixed = fixed
        self.object = None

    def move(self):
        """
        Moves the particle
        """
        if self.fixed:
            self.speed = self.box.nullvector.copy()

        if self.box.trail > 0:
            if self.box.ticks % self.box.skip_trail == 0:
                self.positions.insert(0, self.position.copy())
                self.positions = self.positions[:self.box.trail]

                self.speeds.insert(0, self.speed.copy())
                self.speeds = self.positions[:self.box.trail]

        self.position += self.speed

        # wrap around
        if self.box.torus:
        # if True:
            self.position = numpy.mod(self.position, self.box.box_sizes)
            return self.position

        # Put the particle back in the box. Usefull when the box is resized
        for i, x in enumerate(self.box.nullvector):
            if self.position[i] < x:
                self.position[i] = 1.0*(x + self.radius)

        for i, x in enumerate(self.box.box_sizes):
            if self.position[i] > x:
                self.position[i] = 1.0*(x - self.radius)

        return self.position


    def fast_collision_check(self, ball):
        """
        Fast collision detection

        Args:
            p2 (Particle): particle to check collision with

        Returns:
            boolean: True when possible collision occured
        """
        min_distance = self.radius + ball.radius
        dposition = abs(self.displacement(ball.position))
        #dposition = abs(self.position - p2.position)

        for dpos in dposition:
            if dpos > min_distance:
                return False

        if sum(dposition) > min_distance*min_distance:
            return False
        return True

    def collide(self, ball):
        """
        Handles particle collision

        Based on angle free representation from: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        using vectors generalizes to any number of dimensions

        Args:
            ball (Particle): particle to check collision with

        Returns:
            boolean: True when collision occured
        """
        collided = False

        # if not self.fast_collision_check(ball):
        #     collided = False
        #     return collided

        # dposition = self.position - p2.position
        dposition = self.displacement(ball.position)
        distance2 = dposition.dot(dposition)

        # only collide when particles are moving towards each other:
        # dot product of speed difference and position different < 0
        dspeed = self.speed - ball.speed
        dot_speed_pos = dspeed.dot(dposition)

        dmin = self.radius + ball.radius
        if distance2 > 0 and distance2 < dmin*dmin and dot_speed_pos < 0: # and d2 < distance2:
            dspeed_new = dot_speed_pos*dposition/distance2
            # speed1 = self.speed - (2*p2.mass/(self.mass+p2.mass)) * dspeed.dot(dpos)*(dpos)/distance2
            # speed2 = p2.speed - (2*self.mass/(self.mass+p2.mass)) * -dspeed.dot(-dpos)*(-dpos)/distance2
            speed1 = self.speed - (2*ball.mass/(self.mass + ball.mass)) * dspeed_new
            speed2 = ball.speed - (2*self.mass/(self.mass + ball.mass)) * -dspeed_new
            self.speed = speed1
            ball.speed = speed2
            collided = True

        # self.impuls = self.mass * self.speed
        return collided

    def merge(self, ball):
        """
        Handles particle merge (inelastic collision)

        Args:
            ball (Particle): particle to check collision with

        Returns:
            boolean: True when merge occured
        """
        merged = False

        if not self.fast_collision_check(ball):
            merged = False
            return merged

        # dposition = self.position - p2.position
        dposition = self.displacement(ball.position)
        distance2 = dposition.dot(dposition)

        # only collide when particles are moving towards each other:
        # dot product of speed difference and position different < 0
        dspeed = self.speed - ball.speed
        dot_speed_pos = dspeed.dot(dposition)

        dmin = self.radius + ball.radius
        if distance2 > 0 and distance2 < dmin*dmin and dot_speed_pos < 0: # and d2 < distance2:
            D = self.box.dimensions
            R = (self.radius**D + ball.radius**D)**(1/D)
            C = self.charge + ball.charge
            # color = [c for c in numpy.array(self.color)*0.7]
            color = 0.8 * (numpy.array(self.color) + numpy.array(ball.color))/2
            color = tuple([int(c) for c in color])

            Is = self.mass * self.speed
            Ib = ball.mass * ball.speed
            M = self.mass + ball.mass
            Ir = Is + Ib
            V = Ir / M

            fixed = False
            if ball.fixed:
                P = ball.position
                fixed = True
            elif self.fixed:
                P = self.position
                fixed = True
            else:
                P = (self.position + ball.position)/2

            self.speed = V
            self.position = P
            self.mass = M
            self.radius = R
            self.charge = C
            self.color = color
            self.fixed = fixed

            self.box.merged_particles.append(self)
            index = ball.index()
            self.box.particles.remove(ball)
            self.box.delete_particles.append(ball)
            self.box.delete_trails.append(index)
            merged = True


        # self.impuls = self.mass * self.speed
        return merged


    def bounce_simple(self):
        """
        Check and handles particle hitting the box walls

        Args:
            box (Box): the box

        Returns:
            boolean: True if hit occured
        """
        bounced = False
        old_speed = self.speed.copy()
        index = 0
        for i, x in enumerate(self.box.nullvector):
            if self.position[i] < x + self.radius: # and self.speed[i] < 0:
                self.speed[i] = abs(self.speed[i])
                self.position[i] = x + self.radius
                index = i
                bounced = True
        for i, x in enumerate(self.box.box_sizes):
            if self.position[i] > x - self.radius: # and self.speed[i] > 0:
                self.speed[i] = -abs(self.speed[i])
                self.position[i] = x - self.radius
                index = i
                bounced = True
        # self.impuls = self.mass * self.speed

        momentum = self.mass * (old_speed - self.speed)
        self.box.momentum += momentum
        self.box._normal_momentum += abs(momentum[index])
        return bounced

    def bounce_from_hole(self, plane, hole):
        """
        implements holes in a plane

        Args:
            plane (Plane): the Plane
            hole (tuple(center: numpy.array, radius: float)): the Hole

        Returns:
            boolean: whether to reflect the particle
        """
        (center, radius) = hole
        HR2 = radius * radius
        BR2 = self.radius * self.radius

        ball2center = self.position - center
        distance2plane = ball2center @ plane.unitnormal
        vector2plane = distance2plane * plane.unitnormal
        parallelvector2center = ball2center - vector2plane

        if not self.box.simple_hole_bounce:
            parallelunitnormal =  parallelvector2center / math.sqrt(parallelvector2center @ parallelvector2center)
            edge = center + (parallelunitnormal * radius)
            ball2edge = self.position - edge
            BE2 = ball2edge @ ball2edge
        else:
            BE2 = BR2 + 100

        D2 = parallelvector2center @ parallelvector2center

        reflect = True
        if plane.reflect and (BE2 > BR2) and (D2 < HR2): # hole
            reflect = False

        if not plane.reflect and (BE2 > BR2) and (D2 > HR2): # disk
            reflect = False

        # print(BE2 < BR2, edge, self.position, ball2edge, BE2, BR2)

        return reflect

    def bounce(self):
        """
        Check and handles particle hitting the box walls and the other planes

        Args:
            box (Box): the box

        Returns:
            boolean: True if hit occured
        """
        bounced = False
        old_speed = self.speed.copy()

        start = 0
        if self.box.torus:
            start = self.box.dimensions*2
        for plane in self.box.planes[start:]:
            speed2plane = self.speed @ plane.unitnormal
            distance2plane = abs(plane.distance(self.position))

            # is the particle close enough to bounce of the wall?
            if  distance2plane < self.radius or distance2plane < abs(speed2plane):
                # function from Membrane, allows conditional pass through wall
                if plane.pass_through(self):
                    continue

                reflect = plane.reflect
                for hole in plane.holes:
                    reflected = self.bounce_from_hole(plane, hole)
                    if reflected != reflect:
                        reflect = reflected
                        break

                if not reflect:
                    continue

                hitpoint = plane.intersect_line(self.position, self.speed)
                if hitpoint is None:
                    continue

                normalpoint = plane.intersect_line(self.position, plane.unitnormal)
                vnormalpoint = self.position - normalpoint

                # does particle move towards the wall?
                if (vnormalpoint @ self.speed) < 0:
                    bounced = True
                    vspeed2plane = (self.speed @ plane.unitnormal) * plane.unitnormal
                    #dn = self.speed - vspeed2plane
                    #self.speed = 2*dn - self.speed
                    self.speed = self.speed - 2*vspeed2plane

                    momentum = self.mass * (old_speed - self.speed)
                    self.box.momentum += momentum
                    self.box._normal_momentum += abs(momentum @ plane.unitnormal)
        return bounced

    def displacement(self, position):
        """
        Calculates displacement vector between this particle and a position vectors.
        Accounts for wrapping around if self.torus == True.

        Args:
            position (numpy.array): position

        Returns:
            numpy.array: minimum vector from particle to position
        """
        return self.box.displacement(self.position, position)

    def wrap(self):
        """
        Handles particle wrapping around the box pacman style.

        Args:
            box (Box): the box

        Returns:
            boolean: True when wrapped
        """
        wrapped = False
        for i, x in enumerate(self.box.nullvector):
            if self.position[i] < x:
                self.position[i] = self.box.box_sizes[i] + self.position[i]
                wrapped = True
        for i, x in enumerate(self.box.box_sizes):
            if self.position[i] > x:
                self.position[i] = self.position[i] - self.box.box_sizes[i]
                wrapped = True
        return wrapped

    def interact(self):
        """
        Handles particle, particle interaction, when particles have charge and the box interaction is set
        Uses inverse square

        Args:
            box (Box): the box
            power: float, default=2

        Returns:
            list: new speed vector
        """
        # pylint: disable=protected-access
        if self.box.interaction == 0 or self.charge == 0 or self.fixed:
            return self.speed
        dspeed = self.box.nullvector.copy()
        if self.mass == 0:
            mass = 1
        else:
            mass = self.mass

        distances = []
        if not self.box.optimized_interaction:
            particles = self.box.particles
        else:
            if self.box.interaction_neighbors > 1:
                distances, ids = self.box._kdtree.query(self.position, self.box.interaction_neighbors)
                particles = [self.box.particles[i] for i in ids]
            else:
                # particles = []
                particles = [self.box.particles[i] for i in self.box._neighbors[self.index()]]

        for i, ball in enumerate(particles):
            if ball == self or ball.charge == 0:
                continue

            dpos = self.displacement(ball.position)
            distance2 = dpos.dot(dpos)
            if distance2 > (self.radius+ball.radius)*(self.radius+ball.radius):
                if self.box.optimized_interaction and self.box.interaction_neighbors > 1:
                    distance = distances[i]
                else:
                    distance = math.sqrt(distance2)
                unitnormal = dpos/distance
                if self.box.interaction_power == 2:
                    dspeed += self.charge*ball.charge*self.box.interaction*unitnormal/(mass*distance2)
                else:
                    dspeed += self.charge*ball.charge*self.box.interaction*unitnormal/(mass*(distance**self.box.interaction_power))

        self.speed += dspeed
        return self.speed

    @property
    def potential_energy(self):
        """
        calculates the potential energy of all particles in the box

        Returns:
            float: the potential energy
        """
        PE = 0
        S = 0
        for ball in self.box.particles:
            if ball == self or ball.charge == 0:
                continue

            dpos = self.displacement(ball.position)
            R = math.sqrt(dpos.dot(dpos))
            try:
                S += ball.charge / R
            except ZeroDivisionError:
                pass
        PE = 0.5 * self.box.interaction * self.charge * S
        return PE

    def check_inside(self, coords):
        """
        Checks if coordinates are inside the particle

        Args:
            coords (list of float): coordinates to check

        Returns:
            boolean: True if inside
        """
        inside = False

        for i, x in enumerate(coords):
            if x > self.position[i] - self.radius and x < self.position[i] + self.radius:
                inside = True
        return inside

    @property
    def energy(self):
        """
        return the particle's kinetic energy

        Returns:
            float: energy
        """
        return 0.5 * self.mass * self.speed.dot(self.speed)

    @property
    def momentum(self):
        """
        returns the particles momentum

        Returns:
            list of float: momentum vector
        """
        return self.mass * self.speed

    def __str__(self) -> str:
        pstr = f"particle:\n mass: {self.mass}\n radius: {self.radius}\n position: {self.position}\n speed: {self.speed}"
        return pstr

    def project__(self, axis=3):
        """
        projects extra dimension onto 3D in perspective

        Args:
            axis (int, optional): axis to project. Defaults to 3.

        Returns:
            numpy.array: projected position
        """
        position = self.position.copy()

        if self.box.dimensions < 4:
            return position

        min_ = 0.05
        max_ = 0.95
        A = (min_ - max_) / self.box.box_sizes[axis]
        B = max_
        B = min_

        pos_center_3d = position[:3] - self.box.center[:3]
        w = position[axis]
        f = -A*w + B

        pos = pos_center_3d*f + self.box.center[:3]
        position[:3] = pos
        return position

    def project3d(self, axis=3):
        """
        projects extra dimension onto 3D in perspective

        Args:
            axis (int, optional): axis to project. Defaults to 3.

        Returns:
            numpy.array: projected position
        """
        position = self.box.project3d(self.position, axis)
        return position

    def index(self):
        """
        returns the index of the particle

        Returns:
            int: the index
        """
        i = self.box.particles.index(self)
        return i

    def out(self):
        """
        dumps all properties

        Returns:
            dict: the dump
        """
        particle = {}
        particle["mass"] = float(self.mass)
        particle["radius"] = float(self.radius)
        particle["position"] = [float(f) for f in self.position] # list(self.position)
        particle["speed"] = [float(f) for f in self.speed] # list(self.speed)
        particle["charge"] = float(self.charge)
        particle["fixed"] = self.fixed
        particle["color"] = [int(i) for i in self.color]
        return particle

class Spring:
    """
    Spring between particles
    """
    def __init__(self, length: float, strength: float, damping: float, p1: Particle, p2: Particle) -> None:
        """
        Creates spring between two Particles

        Args:
            length (float): Rest length of spring
            strength (float): Spring constant
            damping (float): Damping constant
            p1 (Particle): First particle
            p2 (Particle): Second particle
        """
        self.length = length
        self.strength = strength
        self.damping = damping
        self.p1 = p1
        self.p2 = p2
        self.fixed = self.p1.fixed and self.p2.fixed
        self.object = None

    def __str__(self) -> str:
        return f'{self.length} {self.strength} {self.damping} {(self.p1.index(), self.p2.index())}'

    def out(self):
        """
        dumps all properties

        Returns:
            dict: the dump
        """
        spring = {}
        spring["length"] = float(self.length)
        spring["strength"] = float(self.strength)
        spring["damping"] = float(self.damping)
        spring["particles"] = [self.p1.index(), self.p2.index()]
        return spring

    # @jit
    def dlength(self):
        """
        Calculates how far spring is stretched or compressed

        Returns:
            float: length
        """
        #dpos = self.p1.position - self.p2.position
        dpos = self.p1.displacement(self.p2.position)
        length2 = dpos.dot(dpos)
        length = math.sqrt(length2)
        dlength = self.length - length
        return dlength

    @property
    def energy(self):
        """
        Potential energy of spring

        Returns:
            float: Potential energy
        """
        dlength = self.dlength()
        return 0.5 * self.strength * dlength * dlength

    # @jit
    def pull(self):
        """
        Uses Hooks law including damping to calculate new particle speeds
        https://en.wikipedia.org/wiki/Harmonic_oscillator
        F = -kx - c(dx/dt)
        """
        if self.fixed:
            return

        dpos = self.p1.displacement(self.p2.position)
        if numpy.all(dpos==0):
            return

        length2 = dpos @ dpos
        length = math.sqrt(length2)
        dlength = self.length - length
        unitnormal = dpos/length

        if dlength:
            sign = dlength/abs(dlength)
        else:
            sign = 1
        dlength = sign * min(abs(dlength), self.length/20)

        dspeed = self.p1.speed - self.p2.speed
        D = self.damping * (dspeed @ dpos)*dpos/length2
        F = -self.strength*dlength*unitnormal

        dv1 = -F/self.p1.mass - D
        dv2 = F/self.p2.mass + D

        self.p1.speed += dv1
        self.p2.speed += dv2

class Source():
    """
    Source or sink of charge
    Exchanges charge with bouncing particle and moves this to other attached sources

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, dcharge, particle1, particle2):
        self.p1 = particle1
        self.p2 = particle2
        self.dcharge = dcharge

class _Rod(Spring):
    """
    A Rod is a Spring with a fixed length
    """
    # pylint: disable=missing-function-docstring
    def __init__(self, length, p1, p2) -> None:
        super().__init__(length=length, strength=0.01, damping=0, p1=p1, p2=p2)

    def __str__(self) -> str:
        return f"{self.length} {(self.p1.index(), self.p2.index())}"

    def pull(self):
        super().pull()
        self._correct_by_length()


    def _correct_by_length(self):
        #correct for fixed length
        # vrod = self.p2.displacement(self.p1.position)
        # vrod = self.p2.position - self.p1.position
        # crod = (self.p2.position + self.p1.position)/2
        # vnormalrod = vrod / (vrod@vrod)

        # speed1_rod = self.p1.speed @ vnormalrod
        # speed2_rod = self.p2.speed @ vnormalrod

        pos1new = self.p1.position + self.p1.speed
        pos2new = self.p2.position + self.p2.speed
        vd = pos2new - pos1new
        cposnew = (pos1new + pos2new)/2
        vnormald = vd / math.sqrt(vd@vd)

        pos1corr = cposnew - (vnormald * self.length)/2
        pos2corr = cposnew + (vnormald * self.length)/2
        # disp = pos2corr - pos1corr
        #l = math.sqrt(disp@disp)

        self.p1.position = pos1corr - self.p1.speed
        self.p2.position = pos2corr - self.p2.speed

    def _correct_by_center_and_rotation(self):
        # pylint: disable=unused-variable
        crod = (self.p2.position + self.p1.position)/2

        pos1new = self.p1.position + self.p1.speed
        pos2new = self.p2.position + self.p2.speed
        cposnew = (pos1new + pos2new)/2

        dcenter = cposnew - crod


    def out(self):
        """
        dumps all properties

        Returns:
            dict: the dump
        """
        rod = {}
        rod["length"] = float(self.length)
        rod["particles"] = [self.p1.index(), self.p2.index()]
        return rod

def save(box, file):
    """
    saves the box and its content to a yaml file

    Args:
        box (Box): the Box
        file (file): the file to save to
    """
    out = box.out()
    yaml.dump(out, file, canonical=False, Dumper=yaml.Dumper, default_flow_style=False, width=120)
    # yaml.dump(out, file, canonical=False, default_flow_style=True, width=120)

def load(file):
    """
    loads a box from a yaml

    Args:
        file (file): the file

    Returns:
        Box: the box with content
    """
    data = yaml.load(file, Loader=yaml.FullLoader)
    box = load_gas(data)
    return box

def load_gas(data):
    """
    initiates a box from a dicts, loaded from a yaml file

    Args:
        data (dict): the box data

    Returns:
        Box: the box with content
    """
    b = data["box"]
    box = Box(b["sizes"])
    shape = box.onevector.shape

    box.gravity = numpy.array(b.get('gravity', box.nullvector.copy()), dtype=float)
    if box.gravity.shape != shape:
        raise ValueError(f'wrong shape for gravity: {box.gravity.shape}, should be {shape}')
    box.set_friction(float(b.get('friction', 0)))
    box.set_interaction(float(b.get('interaction', 0)))
    box.torus = bool(b.get('torus', False))
    box.merge = bool(b.get('merge', False))
    box.trail = int(b.get('trail', 0))
    box.color = b.get('color', (200,200,200))
    box.interaction_power = float(b.get("interaction_power", 2))
    box.optimized_collisions = bool(b.get("optimized_collisions", True))
    box.optimized_interaction = bool(b.get("optimized_interaction", True))
    box.interaction_neighbors = int(b.get("neighbor_count",10))
    box.simple_hole_bounce = bool(b.get("simple_hole_bounce", False))
    if box.dimensions > 1:
        rotations = []
        for rotation in b.get('rotations', []):
            v1 = numpy.array(rotation.get('vector1', box.axis[0]), dtype=float)
            if v1.shape != shape:
                raise ValueError(f'wrong shape for vector1: {v1.shape}, should be {shape}')
            v2 = numpy.array(rotation.get('vector2', box.axis[1]), dtype=float)
            if v2.shape != shape:
                raise ValueError(f'wrong shape for vector2: {v2.shape}, should be {shape}')
            # v2 = numpy.array(rotation['vector2'])
            angle = float(rotation.get('angle', 0))
            rotations.append((v1, v2, angle))
        box.rotations = rotations

    box.set_rotations(box.rotations)

    if "particles" in b:
        for p in b['particles']:
            fixed = bool(p.get('fixed', False))
            charge = float(p.get('charge', 0))
            color = p.get('color', None)
            box.add_particle(p['mass'], p['radius'], p['position'], p['speed'], charge, fixed, color)

    if "springs" in b.keys():
        for s in b['springs']:
            ps = s['particles']
            p1 = box.particles[ps[0]]
            p2 = box.particles[ps[1]]
            damping = float(s.get('damping', 0))

            spring = Spring(s['length'], s['strength'], damping, p1, p2)
            box.springs.append(spring)

    if "planes" in b.keys():
        for p in b["planes"]:
            normal = numpy.array(p["normal"], dtype=float)
            if normal.shape != shape:
                raise ValueError(f'wrong shape for normal: {normal.shape}, should be {shape}')
            point = numpy.array(p["point"], dtype=float)
            if point.shape != shape:
                raise ValueError(f'wrong shape for point: {point.shape}, should be {shape}')
            color = p.get('color', None)
            reflect = bool(p.get('reflect', True))
            if 'pass_through_function' in p:
                plane = _Membrane(box, normal, point)
                plane.filter = getattr(plane, p['pass_through_function']) # pylint: disable=protected-access
                plane.hole_size = p['hole_size']
                plane.max_speed = p['max_speed']
            else:
                plane = Plane(box=box, normal=normal, point=point, color=color, reflect=reflect)

            holes = p.get("holes", [])
            for h in holes:
                point = h.get("point", None)
                radius = h.get("radius", 0)
                if point is not None and radius !=0:
                    plane.add_hole(point, radius)
            box.planes.append(plane)

    box.get_radi()

    return box

class FlatD3Shape():
    """
    Flat 3D shape
    """
    def __init__(self, vertices=None) -> None:
        if vertices is None:
            vertices = []

        self.vertices = vertices
        self.center = [0,0,0]
        self.normal = None
        self._set_props(self.vertices)

    def _get_normal(self):
        """
        Calculates the normal vector from points on the plane

        Args:
            points (list of numpy.array): the points

        Returns:
            numpy.array: unit normal vector
        """
        # points = [v for v in self._vertices]
        if len(self.vertices) < 2:
            raise ValueError("minimal 3 vertices needed")
        c = self.center
        points = [p-c for p in self.vertices]
        shape = numpy.shape(points)
        ones = numpy.ones(shape)
        i = 0
        while linalg.det(points[:3]) == 0 and i < 100:
            points += ones
            i += 1

        normal = linalg.solve(points[:3], numpy.array([1,1,1]))
        unitnormal = normal/math.sqrt(normal@normal)

        c = self.center
        vertices = [v-c for v in self.vertices]
        if not numpy.allclose(vertices@unitnormal, numpy.zeros(len(vertices))):
            raise ValueError("not all points in one plane")

        return unitnormal

    def _set_props(self, vertices):
        self.vertices = vertices
        if self.vertices:
            self.center = sum(vertices)/len(vertices)
            self.normal = self._get_normal()

    def regular_polygon_vertices(self, segments=36):
        """
        creates a regular polygon

        Args:
            segments (int, optional): number of segments. Defaults to 36.

        Returns:
            polygon: the polygon
        """
        points = []
        edges = []

        theta = numpy.deg2rad(360/segments)
        rot = numpy.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


        V = [1,0]
        points.append(V)
        edges.append((0,1))
        for i in range(segments-1):
            edges.append((i+1,(i+2)%segments))
            V = V @ rot
            points.append(V)

        points = [numpy.insert(p,1, 0) for p in points]
        self._set_props(points)

        return (points, edges)

    def skew(self, a):
        """
        creates a skew matrix from a vector

        Args:
            a (numpy.array): the vector

        Returns:
            numpy.array: the skew matrix
        """
        return numpy.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

    def rotate(self, normal):
        """
        aligns the polygon with the given normal

        Args:
            normal (numpy.array): normal vector to align with

        Returns:
            list: the rotated vertices
        """
        normal = numpy.array(normal)
        normal = normal / math.sqrt(normal @ normal)

        V = numpy.cross(self.normal, normal)
        cos_ = self.normal @ normal
        if cos_ == 1:
            return self.vertices

        # I = numpy.array([[1,0,0], [0,1,0], [0,0,1]])
        I = numpy.eye(3)
        Vx = self.skew(V)
        # Vx2 = numpy.square(Vx)
        Vx2 = Vx @ Vx

        mrot = I + Vx + Vx2*(1/(1-cos_))

        c = self.center
        self.vertices = [c+((v-c) @ mrot) for v in self.vertices]
        self._set_props(self.vertices)
        return self.vertices

    def scale(self, size):
        """
        scales the polygon

        Args:
            size (float): scaling factor

        Returns:
            list: the scaled vertices
        """
        c = self.center
        self.vertices = [c+((v-c)*size) for v in self.vertices]
        self._set_props(self.vertices)
        return self.vertices

    def move(self, pos):
        """
        moves the polygon

        Args:
            pos (numpy.array): the shift

        Returns:
            list: the moved vertices
        """
        self.vertices = [v+pos for v in self.vertices]
        self._set_props(self.vertices)
        return self.vertices


def main():
    """
    main
    """
    pass # pylint: disable=unnecessary-pass

if __name__ == "__main__":
    main()
