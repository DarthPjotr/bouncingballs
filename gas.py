"""
Ideal gas in n-dimensional box
"""
import itertools
import numpy
from numpy import linalg
# from pyglet.window.key import E, F
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
import scipy.spatial as sp
import random
import math
from math import sin, cos, acos
import networkx as nx

import yaml
from pprint import pp
import time

import locale
locale.setlocale(locale.LC_ALL, '')

VMAX = 3
RADIUS = 1
MASS = 1
NBALLS = 20

__all__ = ['Box', 'Particle', 'Plane', 'Spring', 'Field', 'ArrangeParticles', 'load', 'save', 'load_gas']

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
        # other properties
        self.trail = 0
        self.skip_trail = 1
        self.object = None
        # properties for optimization by KDTree
        self._max_radius = 0
        self._min_radius = 0
        self._avg_radius = 0
        self._interaction_radius = max(self.box_sizes)
        self._use_kdtree = True
        self._kdtree = None
        self._neighbors = []
        self._neighbor_count = 10
    
    def get_radi(self, interaction_factor=5, neighbor_count=None):
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
        #     self._interaction_radius = interaction_factor*(self.interaction**(1/self.interaction_power))
        # else: 
        #     self._interaction_radius = max(self.box_sizes)
        self._interaction_radius = interaction_factor * max(self.box_sizes) / (len(self.particles)**(1/self.dimensions))
        if not neighbor_count:
            neighbor_count = max(10, int(0.1*len(self.particles)))
        
        self._neighbor_count = min(len(self.particles), neighbor_count)

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
        for i in range(len(self.vertices)):
            for j in range(i+1, len(self.vertices)):
                v1 = self.vertices[i]
                v2 = self.vertices[j]
                c = 0
                for k in range(len(v1)):
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
        box["use_kdtree"] = self._use_kdtree
        box["neighbor_count"] = self._neighbor_count = 10

        output = {"box": box}

        return output
    
    def random_position(self):
        """
        Gives random position in the box

        Returns:
            numpy.array: position
        """        
        V = []
        for max in self.box_sizes:
            V.append(max*random.random())
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
            plane.box_intersections = plane._box_intersections()
            plane.edges = plane._edges()
        self.planes.extend(planes)
        self.center = sum(self.vertices)/len(self.vertices)
        self.ticks = 1
        self._normal_momentum = 0
    
    def _rotation_matrix(self, α, β, γ):
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

    def rotate(self, α, β, γ):
        """
        Rotates content of box around x, y, z axes
        """         
        if self.dimensions < 3:
            return

        for plane in self.planes[2*self.dimensions:]:
            cpos = plane.point[:3] - self.center[:3] 
            plane.point[:3] = self.center[:3] + cpos.dot(self._rotation_matrix(α, β, γ))
            normal = plane.unitnormal[:3]
            plane.unitnormal[:3] = normal.dot(self._rotation_matrix(α, β, γ))
            plane._set_params()

        for ball in self.particles:
            cpos = ball.position[:3] - self.center[:3]
            ball.position[:3] = self.center[:3] + cpos.dot(self._rotation_matrix(α, β, γ))
            speed = ball.speed[:3]
            ball.speed[:3] = speed.dot(self._rotation_matrix(α, β, γ))
    
    def rotate_axis(self, axis, rad):
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
        self.rotate(*rotation)
    
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
        if self.torus == False:
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
        if self.torus == True:
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
            
            avg_speed = sum(ball.speed for ball in self.particles)/len(self.particles)
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
        if ms <0.5: ms = 0.5
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
        if self._use_kdtree:
            pairs = self._kdtree.query_pairs(2*self._max_radius)
        else: 
            pairs = itertools.combinations(range(len(self.particles)), 2)
        
        for i, j in pairs:
            ball1 = self.particles[i]
            ball2 = self.particles[j]
            if self.merge:
                if ball1.merge(ball2): bounced = True
            else:
                if ball1.collide(ball2): bounced = True
        
        # apply rods
        for rod in self.rods:
            rod.pull()

        return bounced
    
    def _energies(self):
        # calculate energies
        self.energy["KE"] = sum(ball.energy for ball in self.particles)
        self.energy["EE"] = 2 * self.energy["KE"]/self.dimensions # https://en.wikipedia.org/wiki/Kinetic_theory_of_gases#Pressure_and_kinetic_energy
        self.energy["PE"] = sum(ball.potential_energy for ball in self.particles)
        self.energy["SE"] = sum(spring.energy for spring in self.springs)

    def _move_all(self):
        # move all balls
        for ball in self.particles:
            position = ball.move()
    
    def _get_kdtree(self):
        positions = [ball.position for ball in self.particles]
        sizes = None
        if self.torus:
            sizes = self.box_sizes
        self._kdtree = KDTree(positions, boxsize=sizes)
        # self._neighbors  = self._kdtree.query_ball_tree(self._kdtree, self._interaction_radius)

class Plane:
    """
    Plane
    """    
    def __init__(self, box: Box, normal=None, point=None, points=None, color=None) -> None:
        """
        Creates plane

        Args:
            box (Box): the Box
            normal (numpy.array, optional): the normal vector. Defaults to None.
            point (numpy.array, optional): a point on the plane. Defaults to None.
            points (list of numpy.array, optional): point on the plane. Defaults to None.
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
        self.object = None
    
    def _set_params(self):
        self.D = self.unitnormal @ self.point
        self.box_intersections = self._box_intersections()
        self.edges = self._edges()
        
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
        while linalg.det(points) == 0 and i < 100:
            points += ones
            i += 1

        normal = linalg.solve(points, self.box.onevector)

        unitnormal = normal/math.sqrt(normal@normal)
        return unitnormal

    def _test_normal(self, points):
        print(self.unitnormal)
        p = points[-1]
        for q in points:
            d = p - q
            print(d @ self.unitnormal)
            p = q
    
    def __str__(self) -> str:
        parts = []
        for i, p in enumerate(self.unitnormal):
            part = "{}*x{}".format(p, i)
            parts.append(part)
        
        equation = " + ".join(parts)
        equation += " = {}".format(self.D)
        # pstr = "plane:\n normal: {}\n point: {}\n D: {}\nequation: {}".format(self.unitnormal, self.point, self.D, equation)
        pstr = "n:{},\tp:{}".format(self.unitnormal, self.point)
        return pstr

    def out(self):
        plane = {}
        plane["normal"] = [float(f) for f in self.unitnormal]
        plane["point"] = [float(f) for f in self.point]
        plane["color"] = [int(i) for i in self.color]
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

    def intersect_line(self, point, vector):
        """
        Calculates intersection point between line and the plane

        Args:
            point (numpy.array): a point on the line
            vector (numpy.array): the vector of the line

        Returns:
            numpy.array: the intersection point
        """ 
        dn = vector @ self.unitnormal
        if dn == 0:
            # line parallel to or on plane, no solution
            return None
        dt = (self.point - point) @ self.unitnormal
        d = dt / dn
        
        return point + vector * d
    
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
        
        # points = self._sort_by_distance(points)
        # points = self._sort_by_planes(points)
        points = self._sort_convexhull(points)

        return points
    
    def _sort_by_distance(self, points):
        if len(points) > 0:
            sorted = []
            imin = 0
            while len(points) > 0:
                p0 = points.pop(imin)
                sorted.append(p0)
                dmin2 = max(self.box.box_sizes) ** 2
                for i, p1 in enumerate(points):
                    d01 = (p1-p0) @ (p1-p0)
                    if d01 < dmin2:
                        dmin2 = d01
                        imin = i
            
            points = sorted

        return points
    
    def _sort_by_angles(self, points):
        if len(points) > 0:
            p1 = points[0] - self.point
            pt = []

            for p in points:
                p2 = p - self.point
                p1p2 = math.sqrt(p1@p1) * math.sqrt(p2@p2)
                p1dp2 = p1@p2

                if p1p2 != 0:
                    cos = p1dp2/p1p2

                if cos >= -1 and cos <= 1:
                    theta = 360*acos(cos)/(2*math.pi)
                    pt.append([theta, cos, p])

            points = [p[1] for p in pt]
        
        return points
    
    def _sort_convexhull(self, points):
        if len(points) < 3:
            return points
        
        # project points on X,Y or Z plane as long as i has a non zero value for the normal
        for i, x in enumerate(self.unitnormal):
            if x != 0:
                break
    
        projected = [numpy.delete(p, i) for p in points]

        hull = ConvexHull(projected)
        vertices = hull.vertices

        sorter = []
        for v in vertices:
            sorter.append(points[v])

        return sorter
    
    def _sort_by_planes(self, points):
        sorted = []

        if self.box.dimensions < 4:
            D = 1
        else:
            D = self.box.dimensions - 2

        i = 0
        while len(points) > 0:
            p0 = points.pop(i)
            sorted.append(p0)
            for i, p1 in enumerate(points):
                # if numpy.any(p0==p1):
                if len([t for t in (p0==p1) if t]) == D:
                    sorted.append(p1)
                    break

        points = sorted
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

        for i, p in enumerate(self.box_intersections):
            edge = (i, (i-1) % size)
            edges.append(edge)
        
        return edges

    def pass_through(self, ball):
        return False

class Membrane(Plane):
    def __init__(self, box: Box, normal=None, point=None, points=None) -> None:
        self._filter = self.no_filter
        self.hole_size = 15
        self.max_speed = 4
        self._bounced = 0
        super().__init__(box, normal=normal, point=point, points=points)
    
    def no_filter(self, ball):
        return False

    def pass_through(self, ball):
        # res = super().pass_through(ball)
        res = self._filter(ball)
        return res
    
    def mass_filter(self, ball):
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
            pass

        return res
    
    def out(self):
        membrane = super().out()
        membrane['pass_through_function'] = self._filter.__name__
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
        
        if (sum(dposition) > min_distance*min_distance):
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
    
    def bounce(self):
        """
        Check and handles particle hitting the box walls

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
                if plane.pass_through(self):
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
        if self.box.interaction == 0 or self.charge == 0 or self.fixed:
            return self.speed
        dspeed = self.box.nullvector.copy()
        if self.mass == 0:
            mass = 1
        else:
            mass = self.mass

        if not self.box._use_kdtree:
            particles = self.box.particles
        else:
            if self.box._neighbor_count > 1:
                distances, ids = self.box._kdtree.query(self.position, self.box._neighbor_count)
                particles = [self.box.particles[i] for i in ids]
            else:
                particles = []

        for i, ball in enumerate(particles):
            if ball == self or ball.charge == 0:
                continue

            dpos = self.displacement(ball.position)
            distance2 = dpos.dot(dpos)
            if distance2 > (self.radius+ball.radius)*(self.radius+ball.radius):
                if self.box._use_kdtree:
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
        pstr = ""
        pstr = "particle:\n mass: {}\n radius: {}\n position: {}\n speed: {}".format(self.mass, self.radius, self.position, self.speed)
        return pstr

    def index(self):
        i = self.box.particles.index(self)
        return i
    
    def out(self):
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
        return "{} {} {} {}".format(self.length, self.strength, self.damping, (self.p1.index(), self.p2.index()))

    def out(self):
        spring = {}
        spring["length"] = float(self.length)
        spring["strength"] = float(self.strength)
        spring["damping"] = float(self.damping)
        spring["particles"] = [self.p1.index(), self.p2.index()]
        return spring
    
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

        length2 = dpos.dot(dpos)
        length = math.sqrt(length2)
        dlength = self.length - length
        N = dpos/length

        dspeed = self.p1.speed - self.p2.speed
        D = self.damping * dspeed.dot(dpos)*dpos/length2
        F = -self.strength*(dlength)*N

        dv1 = -F/self.p1.mass - D
        dv2 = F/self.p2.mass + D

        self.p1.speed += dv1
        self.p2.speed += dv2

class Rod(Spring):
    """
    A Rod is a Spring with a fixed length
    """
    def __init__(self, length, p1, p2) -> None:
        super().__init__(length=length, strength=0.01, damping=0, p1=p1, p2=p2)

    def __str__(self) -> str:
        return "{} {}".format(self.length, (self.p1.index(), self.p2.index()))

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
        crod = (self.p2.position + self.p1.position)/2

        pos1new = self.p1.position + self.p1.speed
        pos2new = self.p2.position + self.p2.speed
        cposnew = (pos1new + pos2new)/2

        dcenter = cposnew - crod


    def out(self):
        rod = {}
        rod["length"] = float(self.length)
        rod["particles"] = [self.p1.index(), self.p2.index()]
        return rod


class ArrangeParticles:
    """
    Standard particle arrangements 
    """    
    def __init__(self, box: Box) -> None:
        """
        Creates particle arrangement

        Args:
            box (Box): The box
        """        
        self.box = box
    
    def set_charge_colors(self, balls):
        for ball in balls:
            if ball.charge >= 1:
                ball.color = [0,255,0]
            elif ball.charge <= -1:
                ball.color = [255,0,0]
            else:
                ball.color = [255,255,0]

        return balls

    def test_bounce(self):
        balls = []
        ball = self.box.add_particle(mass=10, radius=min(self.box.box_sizes)/6,color=[180,180,0])
        balls.append(ball)

        R = 15
        for i in range(R):
            c = (i*255/R) % 255 
            color = [c,c,c]
            ball = self.box.add_particle(mass=1, radius=30, color=color)
            balls.append(ball)

        return balls

    def cuboctahedral (self, radius=10, length=100, strength=0.05, damping=0.01,center=True):       
        G = nx.Graph()
        edges = []
        start = 0
        size = 4
        for i in range(size):
            edge = (start+i, start+(i+1) % size)
            edges.append(edge)
        G.add_edges_from(edges)
        print(edges)

        edges = []
        start = start + size
        size = 8
        for i in range(size):
            edge = (start+i, start+(i+1) % size)
            edges.append(edge)
        G.add_edges_from(edges)

        edges = [(4,0),(4,1),(6,1),(6,2),(8,2),(8,3),(10,3),(10,0)]
        G.add_edges_from(edges)

        edges = [(5,7),(7,9),(9,11),(11,5)]
        G.add_edges_from(edges)

        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)
        return balls

    def football(self, radius=10, length=100, strength=0.05, damping=0.01, center=True):       
        G = nx.Graph()
        edges = []
        start = 0
        size = 5
        for i in range(size):
            edge = (start+i, start+(i+1) % size)
            edges.append(edge)
        
        print(edges)
        G.add_edges_from(edges)

        start = size
        size = 15
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)

        # print(edges)
        G.add_edges_from(edges)

        edges = [(0,5),(1,8),(2,11),(3,14),(4,17)]
        G.add_edges_from(edges)

        start = start + size
        size = 20
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)

        # print(edges)
        G.add_edges_from(edges)

        edges = [(6,20),(7,23),(9,24),(10,27),(12,28),(13,31),(15,32),(16,35),(18,36),(19,39)]
        G.add_edges_from(edges)

        start = start + size
        size = 15
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)
        
        # print(edges)
        G.add_edges_from(edges)

        edges = [(21,40),(22,42),(25,43),(26,45),(29,46), (30,48),(33,49),(34,51),(37,52),(38,54)]
        G.add_edges_from(edges)

        start = start + size
        size = 5
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)
        
        # print(edges)
        G.add_edges_from(edges)

        edges = [(41,55),(44,56),(47,57),(50,58),(53,59)]
        G.add_edges_from(edges)

        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)
        return balls

    def shapes(self, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        G = nx.dodecahedral_graph()
        # G = nx.graph_atlas(134)
        # G = nx.graph_atlas(1167)
        # G = nx.truncated_cube_graph()
        # G = nx.truncated_tetrahedron_graph()
        # G = nx.cycle_graph(5)
        # G = nx.circular_ladder_graph(10)
        # G = nx.circulant_graph(10,[1,4,6])
        # G = nx.frucht_graph()
        # G = nx.moebius_kantor_graph()
        # G = nx.random_tree(10, None)
        # G = nx.sudoku_graph(2)
        # G = nx.pappus_graph()
        # G = nx.octahedral_graph()
        G = nx.hypercube_graph(4)

        # dim = (4,4,4)
        # dim = (2,2,6)
        # dim = (3,3,3)
        # dim = (2,2,2,2)
        # dim = (3,3,6)
        # G = nx.grid_graph(dim=dim, periodic=False)
        # G = nx.wheel_graph(40)
        # G = nx.star_graph(21)

        # G = nx.hexagonal_lattice_graph(4,3)
        # G = nx.triangular_lattice_graph(3,4)
        # G = nx.diamond_graph()

        # G = nx.grid_2d_graph(3,4, periodic=False)
        # G = nx.hypercube_graph(2)
        # G = nx.random_geometric_graph(n=8, radius=8, dim=8, p=10)
        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)

        return balls

    def create_grid(self, dim, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        G = nx.grid_graph(dim=dim, periodic=False)
        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)

        return balls

    def arrange_from_graph(self, G, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        balls = []
        for node in G.nodes:
            try:
                lnode = len(node)
            except TypeError:
                lnode = 1

            if lnode > 2:
                position = numpy.array(node[:self.box.dimensions])
                position = (position * length) # + self.box.center
            else:
                position = self.box.center + self.box.random() * length

            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(mass=1, radius=radius, position=position, speed=speed, charge=1, fixed=False)
            balls.append(ball)
        
        for edge in G.edges:
            if len(edge) == 2:
                node1,node2 = edge
                i = list(G.nodes).index(node1)
                j = list(G.nodes).index(node2)
                p1 = balls[i]
                p2 = balls[j]
                if numpy.all(p1.position == p2.position):
                    p1.position += (self.box.onevector * 10)
                spring = Spring(length=length, strength=strength, damping=damping, p1=p1, p2=p2)
                self.box.springs.append(spring)
            else:
                print(edge)
   
        if center:
            self.box.center_all()

        return balls

    def random_balls(self, nballs: int, mass=None, radius=None, max_speed=VMAX, charge=0):
        """
        Randomly places balls in the box

        Args:
            nballs (int): Number of balls
            mass (float, optional): Mass of the balls. Defaults to None.
            radius (int, optional): Radius of balls. Defaults to None.
            max_speed (numpy.array, optional): Speed of balls. Defaults to VMAX.
            charge (int, optional): Charge of the balls. Defaults to 0.

        Returns:
            list: list of balls
        """        
        rand_m = False
        rand_r = False
        rand_c = False

        if mass is None:
            rand_m = True
        if radius is None:
            rand_r = True
        if charge is None:
            rand_c = True
            
        balls = []
        for i in range(nballs):
            if rand_m:
                mass = random.randrange(1, 30) * 1.0
            if rand_r:
                radius = random.randrange(1, 30) * 1.0
            if rand_c:
                # charge = random.randint(-1, 1)
                charge = random.choice([-1, 1])

            speed = self.box.random(max_speed)
            ball = self.box.add_particle(mass, radius, None, speed, charge, None)
            balls.append(ball)

        return balls
    
    def create_simplex(self, size=200, position=None, charge=0, vertices=None):
        """
        Creates simplex

        Args:
            size (int, optional): Size of the edges. Defaults to 200.
            position (numpy.array, optional): position. Defaults to None.
            charge (int, optional): Charge of the balls. Defaults to 0.
            vertices (int, optional): Number of vertices. Defaults to None.

        Returns:
            list: list of balls
        """        
        if position is None:
            center = self.box.center
        else:
            center = position

        if vertices is None:
            vertices = self.box.dimensions+1
        
        radius = size / 5

        balls = []
        for i in range(vertices):
            pos = center + self.box.random(size)
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, radius, pos, speed, charge)
            balls.append(ball)

        balls[0].speed = 5 * self.box.onevector.copy()
        balls[-1].speed = -5 * self.box.onevector.copy()

        for i, b1 in enumerate(balls):
            for b2 in balls[i:]:
                if b1 != b2:               
                    spring = Spring(size, 0.05, 0.01, b1, b2)
                    self.box.springs.append(spring)
        return balls

    def create_box(self, size, position=None, charge=0):
        """
        Creates a box 

        Args:
            size (float): size of the box
            position (numpy.array, optional): position. Defaults to None.
            charge (int, optional): charge of the balls of the box. Defaults to 0.

        Returns:
            list: list of balls
        """        
        ratio = max(self.box.box_sizes)/size
        sizes = self.box.box_sizes/ratio
        if position is None:
            center = self.box.box_sizes/2
        else:
            center = position
        box = Box(sizes)
        speed = self.box.nullvector.copy()

        balls = []
        for vertex in box.vertices:
            pos = center - (box.box_sizes/2) + vertex
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, 10, pos, speed, charge)
            balls.append(ball)
        
        balls[0].speed = 5 * self.box.onevector.copy()
        balls[-1].speed = -5 * self.box.onevector.copy()

        l = sum(box.box_sizes)/box.dimensions
        for edge in box.edges:
            spring = Spring(l, 0.01, 0.01, balls[edge[0]], balls[edge[1]])
            self.box.springs.append(spring)
        
        return balls
    
    def create_kube_planes(self, size, nballs):
        sizes = numpy.array(self.box.dimensions * [size*1.0])
        kube = Box(sizes)
        dcenter = self.box.center - kube.center
        # for vertix in kube.vertices:
        #     vertix += dcenter
        # for plane in kube.planes:
        #     plane.point += dcenter
        #     plane.box = self.box
        #     
        for plane in kube.planes:
            point = plane.point + dcenter
            normal = plane.unitnormal
            plane_ = Plane(self.box, normal, point)
            self.box.planes.append(plane_)
    
        balls = []

        b = {}
        for (p1, p2) in kube.edges:
            if p1 not in b:
                pos1 = kube.vertices[p1] + dcenter
                speed = self.box.nullvector.copy()
                ball1 = self.box.add_particle(1,3, pos1, speed, 0, True, [255,255,255])
                b[p1] = ball1
                balls.append(ball1)

            if p2 not in b:
                pos2 = kube.vertices[p2] + dcenter
                speed = self.box.nullvector.copy()
                ball2 = self.box.add_particle(1,3, pos2, speed, 0, True, [255,255,255])
                b[p2] = ball2
                balls.append(ball2)
            
            l = math.sqrt((b[p1].position - b[p2].position) @ (b[p1].position - b[p2].position))
            spring = Spring(l, 0, 0, b[p1], b[p2])
            self.box.springs.append(spring)


        for i in range(nballs):
            position = kube.random_position() + dcenter
            ball = self.box.add_particle(10, 50, position)
            balls.append(ball)
        
        return balls

    def create_n_mer(self, nballs, n=2, star=False, circle=False, charge=0):
        """
        Creates shape of n balls in a line, star of circle shape

        Args:
            nballs (int): number of balls
            n (int, optional): number of balls in the shape. Defaults to 2.
            star (bool, optional): Make a star shape. Defaults to False.
            circle (bool, optional): Make a circle. Defaults to False.
            charge (int, optional): Charge of the balls. Defaults to 0.

        Returns:
            list: list of balls
        """        
        radius = 20
        lspring = 150
        balls = []
        alternate = False
        if charge is None:
            alternate = True
            charge = 1
        for i in range(round(nballs/n)):
            pos1 = self.box.random_position()
            speed = self.box.random(3)
            if star and alternate:
                charge = n - 1
            b1 = self.box.add_particle(1, radius, pos1, speed, charge)
            bstart = b1
            balls.append(b1)
            if n > 1:
                for i in range(n-1):
                    if alternate:
                        if star:
                            charge = -1
                        else:
                            charge = -charge
                    pos2 = pos1 + self.box.random() * (lspring)
                    speed2 = speed + self.box.random(0.5)
                    b2 = self.box.add_particle(1, radius, pos2, speed2, charge)
                    balls.append(b2)
                    spring = Spring(lspring, 0.15, 0.00, b1, b2)
                    self.box.springs.append(spring)
                    if not star:
                        b1 = b2
                if circle:
                    spring = Spring(lspring, 0.01, 0, b1, bstart)
                    self.box.springs.append(spring)
        return balls
    
    def create_kube(self, size, position=None, charge=0):
        """
        Creates a kube

        Args:
            size (float): size of the box
            position (numpy.array, optional): position. Defaults to None.
            charge (int, optional): charge of the balls of the box. Defaults to 0.

        Returns:
            list: list of balls
        """        
        if position is None:
            center = self.box.box_sizes/2
        else:
            center = position
        sizes = [size]*self.box.dimensions
        box = Box(sizes)
        speed = self.box.nullvector.copy()

        balls = []
        for vertex in box.vertices:
            pos = center - (box.box_sizes/2) + vertex
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, 1, pos, speed, charge)
            balls.append(ball)
        
        # balls[0].speed = 5 * self.box.onevector.copy()
        # balls[-1].speed = -5 * self.box.onevector.copy()

        for i, b0 in enumerate(balls):
            for b1 in balls[i:]:
                if b1 == b0:
                    continue
                d = math.sqrt((b0.position-b1.position) @ (b0.position-b1.position))
                spring = Spring(d, 0.1, 0.02, b0, b1)
                self.box.springs.append(spring)
    
        return balls 

    def create_pendulum(self, gravity=0.1, direction=None):
        self.box.set_gravity(gravity, direction)
        balls = []

        pos = self.box.center.copy()
        # pos[Box.Y] = self.box.box_sizes[Box.Y]
        speed = self.box.nullvector.copy()
        ancor = self.box.add_particle(1, 1, pos, speed, charge=0, fixed=True, color=[255,255,255])
        balls.append(ancor)
        
        lspring = 150
        pos = pos.copy()
        pos[Box.X] += lspring
        ball1 = self.box.add_particle(10,20, pos, speed, color=[0,255,255])
        balls.append(ball1)

        spring = Spring(lspring, 10, 0.01, ancor, ball1)
        self.box.springs.append(spring)

        lspring = 100
        pos = pos.copy()
        pos += self.box.random(lspring)
        # pos[Box.Z] = self.box.center[Box.Z]
        ball2 = self.box.add_particle(10,20, pos, speed, color=[0,255,255])
        balls.append(ball2)

        spring = Spring(lspring, 10, 0.01, ball1, ball2)
        self.box.springs.append(spring)

        return balls
    
    def test_interaction_simple(self, interaction, power=2):
        self.box.set_interaction(interaction, power)
        balls = []

        dpos = self.box.nullvector.copy()
        dpos[1] = 120
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = 3
        ball = self.box.add_particle(5, 5, position=list(pos), speed=list(speed), charge=-1)
        balls.append(ball)

        dpos = self.box.nullvector.copy()
        dpos[1] = -20
        pos = self.box.center + dpos
        speed[0] = -3/6
        ball = self.box.add_particle(30, 30, position=list(pos), speed=list(speed), charge=1)
        balls.append(ball)

        return balls

    def test_interaction(self, interaction=20000, power=2, M0=40, V0=6, D=140, ratio=0.1):
        self.box.set_interaction(interaction, power)
        balls = []

        dpos = self.box.nullvector.copy()
        dpos[1] = D * ratio
        M = M0 * (1-ratio)
        V = V0 * (ratio)
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = V
        ball = self.box.add_particle(M, int(M), position=list(pos), speed=list(speed), charge=-1, color=[255,0,0])
        balls.append(ball)

        dpos = self.box.nullvector.copy()
        dpos[1] = -D * (1-ratio)
        M = M0 * ratio
        V = -V0 * (1-ratio)
        pos = self.box.center + dpos
        speed[0] = V
        ball = self.box.add_particle(M, int(M), position=list(pos), speed=list(speed), charge=1, color=[0,255,0])
        balls.append(ball)

        return balls

    def test_springs(self, interaction=0):
        self.box.set_interaction(interaction)

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[1] = 120
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = 0
        #speed[1] = 1
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=-1)
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = -20
        pos = self.box.center + dpos
        speed[0] = -0
        #speed[1] = -1/6
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=1)
        balls.append(ball)
        b2 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = 50
        dpos[1] = 50
        pos = self.box.center + dpos
        speed[0] = -0
        #speed[1] = -1/6
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=1)
        balls.append(ball)
        b3 = ball

        spring = Spring(150, 0.03, 0.001, b1, b2)
        self.box.springs.append(spring)
        spring = Spring(150, 0.03, 0.001, b2, b3)
        self.box.springs.append(spring)
        spring = Spring(190, 0.02, 0.001, b1, b3)
        self.box.springs.append(spring)
        
        return balls

    def test_spring(self, length=150, distance=240, strength=0.03, interaction=0 , center=None,speed=None):
        self.box.set_interaction(interaction)
        if center is None:
            center = self.box.center

        if speed is None:
            speed = self.box.nullvector.copy()

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[0] = distance/2
        pos = center + dpos
        # speed = self.box.nullvector.copy()
        # speed[0] = 0

        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=-1, color=[255,0,0])
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = -distance/2
        pos = center + dpos
        # speed[0] = -0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=1, color=[0,255,0])
        balls.append(ball)
        b2 = ball

        spring = Spring(length, strength, 0.000, b1, b2)
        self.box.springs.append(spring)
        
        return balls
    
    def test_fixed_charge(self, interaction=10000):
        self.box.set_interaction(interaction)

        balls = []

        ball = self.box.add_particle(1, 25, self.box.center, speed=self.box.nullvector.copy(), charge=5, fixed=True, color=[255,0,255])
        balls.append(ball)

        balls_ = self.create_n_mer(2, 2)
        for i, ball in enumerate(balls_):
            if i % 2 == 0:
                ball.charge = -1
            else:
                ball.charge = 1
        balls.extend(balls_)

        return balls
    
    def test_walls(self):
        normal = self.box.nullvector.copy()
        normal[0] = 1
        normal[1] = 0.3
        # wall = Plane(self.box, normal, self.box.center)
        wall = Membrane(self.box, normal, self.box.center)
        # wall.hole_size = 15
        wall.max_speed = 4
        wall._filter = wall.maxwells_demon_filter

        self.box.planes.append(wall)

        balls = []
        position = self.box.center.copy()
        position += self.box.center/2
        # position[0] += self.box.center[0]/2
        speed = self.box.onevector.copy()
        ball = self.box.add_particle(1, 50, position, speed=speed, charge=0, fixed=False, color=[0,0,255])
        balls.append(ball)

        position = self.box.center.copy()
        position -= self.box.center/2 
        speed = -3 * self.box.onevector.copy()
        ball = self.box.add_particle(1, 10, position, speed=speed, charge=0, fixed=False, color=[255,255,0])
        balls.append(ball)

        balls += self.random_balls(10, 1, 10, 2)

        return balls

    def test_rod(self, length=150, interaction=0):
        self.box.set_interaction(interaction)

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[Box.X] = length/2
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        # speed = self.box.random(1.0)
        #speed[Box.X] = 1.5
        #speed[Box.Y] = 2.0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=0, color=[255,0,0])
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[Box.X] = -length/2
        # dpos[Box.Y] = 50
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        # speed[Box.Y] = 1.0 # self.box.random(0.5)
        
        speed[Box.X] = 1.5
        speed[Box.Y] = -2.0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=0, color=[0,255,0])
        balls.append(ball)
        b2 = ball

        rod = Rod(length, b1, b2)
        # spring = Spring(length, 1.0, 0, b1, b2)
        # self.box.rods.append(rod)
        self.box.rods.append(rod)
        
        return balls

def save(box, file):
    out = box.out()
    yaml.dump(out, file, canonical=False, Dumper=yaml.Dumper, default_flow_style=False, width=120)
    # yaml.dump(out, file, canonical=False, default_flow_style=True, width=120)

def load(file):
    data = yaml.load(file, Loader=yaml.FullLoader)
    box = load_gas(data)
    return box

def load_gas(data):
    b = data["box"]
    box = Box(b["sizes"])

    box.gravity = numpy.array(b.get('gravity', box.nullvector.copy()))
    box.set_friction(b.get('friction', 0))
    box.set_interaction(b.get('interaction', 0))
    box.torus = b.get('torus', False)
    box.merge = b.get('merge', False)
    box.trail = b.get('trail', 0)
    box.color = b.get('color', (200,200,200))
    box.interaction_power = b.get("interaction_power", 2)
    box._use_kdtree = b.get("use_kdtree", True)
    box._neighbor_count = b("neighbor_count",10)

    if "particles" in b:
        for p in b['particles']:
            fixed = p.get('fixed', False)
            charge = p.get('charge', 0)
            color = p.get('color', None)
            box.add_particle(p['mass'], p['radius'], p['position'], p['speed'], charge, fixed, color)
    
    if "springs" in b.keys():
        for s in b['springs']:
            ps = s['particles']
            p1 = box.particles[ps[0]]
            p2 = box.particles[ps[1]]
            damping = s.get('damping', 0)

            spring = Spring(s['length'], s['strength'], damping, p1, p2)
            box.springs.append(spring)
    
    if "planes" in b.keys():
        for p in b["planes"]:
            normal = p["normal"]
            point = p["point"]
            color = p.get('color', None)
            if 'pass_through_function' in p:
                plane = Membrane(box, normal, point)
                plane._filter = getattr(plane, p['pass_through_function'])
                plane.hole_size = p['hole_size']
                plane.max_speed = p['max_speed']
            else:
                plane = Plane(box, normal, point, color)
            box.planes.append(plane)

    box.get_radi()

    return box


class Test():
    def test_plane(self):
        planes = []
        box = Box([10,20,30])
        plane = Plane(box, [3,2,5], [5,10,30])
        print(plane)
        planes.append(plane)
        

        print("\n####\n")

        # plane2 = Plane(box, points=[[5,10,15],[6,8,2],[7,12,4]])
        plane2 = Plane(box, [2,4,3], [5,10,30])
        # plane2 = Plane(box, [3,2,5], [5,10,30])
        print(plane2)
        planes.append(plane2)

        print("\n####\n")

        plane3 = Plane(box, [5,3,6], [5,15,25])
        print(plane3)
        planes.append(plane3)

        print("\n####\n")
        print(plane.intersection(planes))
        intersection = plane.intersection(planes)

        print("\n####\n")

        print(plane.on([5,10,30]))
        print(plane.on2([5,10,30]))

        print(plane.on([5,13,33]))
        print(plane.on2([5,13,33]))

        print("\n####\n")
        for p in planes:
            print(p.on(intersection))
            print(p.on2(intersection))


    def test_box(self):
        box = Box([800,600,700])
        print(box.vertices)
        print(box.edges)
        print(box.volume())
        print(box.area())
        print(box.axis)

    def test(self):
        BOX_DIMENSIONS = [800, 600, 700, 400, 300]
        NBALLS = 20
        GRAVITY = 0.0
        FRICTION = 0.00
        INTERACTION = 0000.0
        TORUS = False
        DIMENSIONS = 3
        SPEED = VMAX
        box = Box(BOX_DIMENSIONS[:DIMENSIONS])
        # for i in range(NBALLS):
        #     if SPEED is not None:
        #         speed = numpy.array([2*random.random()-1 for r in range(DIMENSIONS)])
        #         speed = SPEED * speed / math.sqrt(speed.dot(speed))
        #         speed = list(speed)
        #     else:
        #         speed = SPEED
        #     box.add_particle(1, 1, None, speed)

        arrangement = ArrangeParticles(box)
        # arrangement.random_balls(25, None, None, VMAX, None)

        # balls = arrangement.test_springs(10000)
        # self.add_balls(balls)

        # balls = arrangement.test_interaction(40000, M0=40, V0=6, D=300, ratio=0.1)
        balls = arrangement.test_spring(length=300, distance=240, strength=0.0001, interaction=000)
        for i in range(1000):
            try:
                box.go()
                # KE = sum(ball.energy for ball in box.particles)
                # EE = KE * (2 / DIMENSIONS) # https://en.wikipedia.org/wiki/Kinetic_theory_of_gases#Pressure_and_kinetic_energy
                # PV = box.pressure()*box.volume()
                # print("{:.2f} {:.2f} {:.2f}".format(PV, EE, PV/EE))
                # PE = sum(ball.potential_energy for ball in box.particles)
                # SE = sum(spring.energy for spring in box.springs)
                KE = box.energy["KE"]
                PE = box.energy["PE"]
                EE = box.energy["EE"]
                SE = box.energy["SE"]

                print("{};{};{};{};{};{}".format(i, KE, PE, SE, EE, KE+PE+SE).replace(".", ","))
                
                # print("{:,2f} {:,2f} {:,2f}".format(PV, E, PV/E))

            except KeyboardInterrupt:
                print("user interupted")
                break
        
        from pprint import pp
        pp(box.out())            

    def test_displacement(self):
        box = Box([10, 20])
        p1 = numpy.array([1,2])
        p2 = numpy.array([2,19])  
        p3 = numpy.array([9,19])
        box.torus = True

        d = box.displacement(p1, p2)
        print(d, p1 - p2)

        d = box.displacement(p1, p3)
        print(d, p1 - p3)

    def test_save_yaml(self):
        FILE = "D:\\temp\\box.yml"
        BOX_DIMENSIONS = [800, 600, 700, 400, 300]
        NBALLS = 20
        GRAVITY = 0.0
        FRICTION = 0.00
        INTERACTION = 0000.0
        TORUS = False
        DIMENSIONS = 3
        box = Box(BOX_DIMENSIONS[:DIMENSIONS])
        arrangement = ArrangeParticles(box)
        balls = arrangement.test_spring(length=300, distance=240, strength=0.0001, interaction=000)

        with open(FILE) as file:
            save(box, file)

    def test_load_yaml(self):
        FILE = "D:\\temp\\box.yml"
        with open(FILE) as file:
            box = load(file)
        print(box)

        for p in box.particles:
            print(p)
        
        for s in box.springs:
            print(s)
        
        for i in range(100):
            box.go()
            print(box.ticks)
    
    def plane(self):
        # sizes = [1000, 900, 1080]
        sizes = [100, 200, 300]
        box = Box(sizes)
        print(box.vertices)
        print(box.edges)
        print(box.axis)
        for p in box.planes:
            print(p)

        # plane = Plane(box, points=box.axis)
        point = box.center
        normal = numpy.array([1,2,0,1])

        normal = [1,1,1,1]

        normal = normal[:box.dimensions]
        point = box.center
        plane = Plane(box, normal=normal, point=point)
        # print(plane)
        box.planes.append(plane)
        plane = Plane(box, normal=normal, point=point)
        # print(plane)
        # box.planes.append(plane)

        print("\n####\n")
        for plane in box.planes[2*box.dimensions:]:
            points = plane.box_intersections
            print(plane)
            print("\n")
            points = numpy.array(points)
            # points.sort(axis=2)
            for point in points:
                print(point)
            print("\n")
        
        for plane in box.planes[2*box.dimensions:]:
            points = plane.box_intersections
            print(plane)
            for (i,j) in plane.edges:
                print(i, j, points[i], points[j])
            
            # print("\n")
            # results = []
            # points = []
            # for (i, j) in box.edges:
            #     v1 = box.vertices[i]
            #     v2 = box.vertices[j]
            #     V = v1 - v2
            #     intersection = plane.intersect_line(v1, V)
            #     if intersection is not None and numpy.all(intersection <= box.box_sizes) and numpy.all(intersection >= box.nullvector):
            #         skip = False
            #         for ins in points:
            #             if numpy.allclose(intersection, ins):
            #                 skip = True
            #                 break
            #         if not skip:
            #             points.append(intersection)
            #             results.append(((i,j),intersection))
                
            # for p in results:
            #     print(p)

    def shapes(self):
        sizes = [100, 200, 300]
        box = Box(sizes)
        arr = ArrangeParticles(box)
        balls = arr.shapes()

        for i in range(100):
            box.go()
    
    def kdtree(self):
        sizes = [100, 200, 300, 200]
        box = Box(sizes)
        box.interaction = 000
        box._use_kdtree = True
        box._neighbor_count=20

        arr = ArrangeParticles(box)
        nballs = 50
        balls = arr.random_balls(nballs=nballs, mass=1, radius=5, charge=0)
        balls = arr.random_balls(nballs=nballs, mass=1, radius=5, charge=0)
        box.get_radi()
        # print([ball.position for ball in box.particles])

        for i in range(100):
            box.go()
        
        ticks = box.ticks
        return ticks

def main():
    print("START")

    t = Test()
    
    # t.test_wall()
    # t.test_box()
    # t.test()
    # t.test_save_yaml()
    # t.test_load_yaml()
    # t.test_displacement()
    # t.normal()
    # t.shapes()
    # t.test_shapes()
    start = time.perf_counter()
    ticks = t.kdtree()
    end = time.perf_counter()
    dtime = end - start
    print("time = {:.2f}, tick/sec = {:.2f}".format(dtime, ticks/dtime))

    print("END")

if __name__ == "__main__": 
    main()
