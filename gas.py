"""
Ideal gas in n-dimensional box
"""
import itertools
import numpy
from numpy import linalg
import random
import math
from math import sin, cos

import yaml
from pprint import pp

import locale
locale.setlocale(locale.LC_ALL, '')

# SCREEN_WIDTH = 700
# SCREEN_HEIGHT = 500
# SCREEN_DEPTH = 400

# DIMENSIONS=2

# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# BLUE = (0, 0, 255)
# RED = (255, 0 ,0)
# GREEN = (0, 255,0)

VMAX = 3
RADIUS = 1
MASS = 1
NBALLS = 20


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
        self.dummy_ball = Particle(self.box, 1, 1, position, speed, 1, [0,0,0])
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
        # content of the box
        self.walls = []
        self.particles = []
        self.springs = []
        self.rods = []
        # dynamic properties
        energies = ["KE", "EE", "PE", "SE"]
        self.energy = {e : 0.0 for e in energies}
        self.momentum = self.nullvector.copy()
        self._normal_momentum = 0
        self.ticks = 0

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
            plane = Plane(self, point=point, points=points[:self.dimensions])
            # plane.point = point
            self.planes.append(plane)

        coordinates = self.nullvector
        for i, x in enumerate(coordinates):
            points = [p for p in self.vertices if p[i] == x]
            point = sum(points)/len(points)
            plane = Plane(self, point=point, points=points[:self.dimensions])
            # plane.point = point
            self.planes.append(plane)

    def __str__(self) -> str:
        return str(self.box_sizes)
    
    def out(self):
        box = {}
        box["sizes"] = [float(f) for f in self.box_sizes] # list(self.box_sizes)
        box['torus'] = self.torus
        box["gravity"] = [float(f) for f in self.gravity] # list(self.gravity)
        box["friction"] = float(self.friction)
        box["interaction"] = float(self.interaction)
        box["particles"] = [ball.out() for ball in self.particles]
        box["springs"] = [spring.out() for spring in self.springs]

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
        rpos = [random.randrange(radius, x - radius)*1.0 for x in self.box_sizes]
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
        
        D2 = direction.dot(direction)
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
    
    def set_interaction(self, interaction):
        """
        Sets the interaction between particles

        Args:
            interaction (float): the interaction

        Returns:
            float: the interaction
        """
        self.interaction = interaction
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
    
    def center_all(self):
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

    def go(self):
        """
        Calculate speeds and move all particles to next position

        Returns:
            boolean: true if particles collides into each other
        """
        bounced = False
        self.ticks += 1

        bounced = self._speeds()
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
            # Bounce or wrap the ball if needed
            if self.torus:
                ball.wrap()
            else:
                ball.bounce()      
            # hit the walls:
            for wall in self.walls:
                ball.hit(wall)
            # collide the balls
            for ball2 in self.particles[i:]:
                if ball.collide(ball2): bounced = True
        
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
            ball.move()

class Plane:
    def __init__(self, box: Box, normal=None, point=None, points=None) -> None:
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
                pass
                # raise ValueError("wrong size")
            else:
                if point is None:
                    point = numpy.array(sum(point for point in points)/len(points))
                points = numpy.array(points)
                self.unitnormal = self._get_normal(points)
                self.point = numpy.array(point)
                # self._test_normal(points)
        else:
            raise TypeError("missing required argument")
        
        self.D = self.unitnormal @ self.point
        
    def _get_normal(self, points):

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
        pstr = "plane:\n normal: {}\n point: {}\n D: {}\nequation: {}".format(self.unitnormal, self.point, self.D, equation)
        return pstr
    
    def intersection(self, planes):
        if self not in planes:
            planes.append(self)
        
        if len(planes) != self.box.dimensions:
            raise ValueError("not enough planes to calculate intersection")
        normals = [p.unitnormal for p in planes]
        Ds = [p.D for p in planes]

        intersection = linalg.solve(normals, Ds)

        return intersection
    
    # def on(self, point):
    #     v = numpy.array(point) - self.point
    #     if v @ self.normal == 0:
    #         return True
    #     return False
    
    # def on2(self, point):
    #     point = numpy.array(point)
    #     if sum(point * self.normal) == self.D:
    #         return True
    #     else:
    #         return False
    
    def distance(self, point):
        point = numpy.array(point)
        v = point - self.point
        return v @ self.unitnormal
    
    def intersect_line(self, point, vector):
        dt = (self.point - point) @ self.unitnormal
        dn = vector * self.unitnormal
        if dt == 0:
            return None
        elif dn == 0:
            return None
        else:
            d = dt / dn
        
        return point + vector * d


class Wall:
    """
    Extra wall inside the box
    """
    def __init__(self, box: Box, rpos: float, dimension: int) -> None:
        """
        Creates wall

        Args:
            box (Box): the box
            rpos (float): relative position in the box, must be between 0 and 1
            dimension (int): which box dimension to make wall, must be between 0 and box.dimensions-1

        Raises:
            ValueError: rpos must be between 0 and 1
        """
        self.box = box
        if rpos < 0 or rpos > 1:
            raise ValueError("rpos must be between 0 and 1")
        self.dimension = dimension
        self.rpos = rpos
        self._set_props()
    
    def _set_props(self):
        """
        Sets wall properties
        """
        self._vector = self.box.onevector.copy()
        self._vector[self.dimension] *= self.rpos
        self.position = self.box.box_sizes * self._vector
        self.vertices = numpy.array([vertex*self._vector for vertex in self.box.vertices if vertex[self.dimension] != 0]) 
        self.center = sum(self.vertices)/len(self.vertices)
    
    def __str__(self) -> str:
        return str(self.vertices)

    def move(self, new_rpos):
        """
        Moves the wall

        Args:
            new_rpos (float): new relative position
        """
        self.rpos = new_rpos
        self._set_props()


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
        self.speed = numpy.array(speed, dtype=float)
        self.charge = charge
        self.color = tuple(color)
        self.fixed = fixed
        self.object = None
        # self.impuls = self.mass * self.speed
    
    def move(self):
        """
        Moves the particle
        """
        if self.fixed:
            self.speed = self.box.nullvector.copy()

        self.position += self.speed

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
            p2 (Particle): particle to check collision with

        Returns:
            boolean: True when collision occured
        """
        collided = False

        if not self.fast_collision_check(ball):
            collided = False
            return collided

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
        
    def _bounce_org(self):
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
        index = 0

        for plane in self.box.planes:
            if abs(plane.distance(self.position)) < self.radius:
                bounced = True
                dp = (self.speed @ plane.unitnormal) * plane.unitnormal
                dn = self.speed - dp
                self.speed = 2*dn - self.speed
                
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

    def hit(self, wall):
        """
        Handles hitting extra walls in the box

        Args:
            wall (Wall): the wall

        Returns:
            boolean: True when the wall was hit
        """
        bounced = False
        dwall = self.position[wall.dimension] - wall.position[wall.dimension]
        new_pos = dwall + self.speed[wall.dimension]
        #print(ball, dwall, new_pos, (dwall < 0 and new_pos > 0) or (dwall > 0 and new_pos < 0))
        if (dwall <= 0 and new_pos > 0) or (dwall >= 0 and new_pos < 0):
            old_speed = self.speed.copy()
            self.speed[wall.dimension] = -self.speed[wall.dimension]
            self.position += self.speed
            wall.box.impuls += self.mass * (old_speed - self.speed)
            bounced = True
        
        return bounced
    
    def interact(self):
        """
        Handles particle, particle interaction, when particles have charge and the box interaction is set
        Uses inverse square

        Args:
            box (Box): the box

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

        for ball in self.box.particles:
            if ball == self or ball.charge == 0:
                continue

            dpos = self.displacement(ball.position)
            distance2 = dpos.dot(dpos)
            if distance2 > (self.radius+ball.radius)*(self.radius+ball.radius):
                N = dpos/math.sqrt(distance2)
                dspeed += self.charge*ball.charge*self.box.interaction*N/(mass*distance2)
        
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
    
    def __str__(self) -> str:
        return "{} {} {}".format(self.strength, self.damping, (self.p1.index(), self.p2.index()))

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
        #dpos = self.p1.position - self.p2.position
        dpos = self.p1.displacement(self.p2.position)
        length2 = dpos.dot(dpos)
        length = math.sqrt(length2)
        dlength = self.length - length
        N = dpos/length

        dspeed = self.p1.speed - self.p2.speed
        #dot_speed_pos = dspeed.dot(dpos)
        #D = self.damping * dot_speed_pos*dpos/length2
        D = self.damping * dspeed.dot(dpos)*dpos/length2
        F = -self.strength*(dlength)*N

        dv1 = -F/self.p1.mass - D
        dv2 = F/self.p2.mass + D

        self.p1.speed += dv1
        self.p2.speed += dv2

class Rod:
    def __init__(self, box, lenght, p1, p2, vcenter=None, vrot=None) -> None:
        self.box = box
        self.length = lenght
        self.p1 = p1
        self.p2 = p2

        if vcenter is None:
            vcenter = numpy.zeros(self.box.dimensions)
        self.vcenter = vcenter
        if vrot is None:
            vrot = numpy.zeros(self.box.dimensions)
        self.vrot = vrot
        self.results = ()
        # self.hit(self.p1, self.p2)
        # self.hit(self.p2, self.p1)

    def __str__(self) -> str:
        return "{} {}".format(self.length, (self.p1.index(), self.p2.index()))

    def pull_old(self):
        # dpos = self.p1.displacement(self.p2.position)
        dpos = self.p1.position - self.p2.position
        R2 = dpos @ dpos
        V1r = self.p1.speed.dot(dpos)*dpos/R2
        V2r = self.p2.speed.dot(dpos)*dpos/R2
        V1p = self.p1.speed - V1r
        V2p = self.p2.speed - V2r
        dVr = V1r - V2r


        self.p1.speed = V1p + dVr
        self.p2.speed = V2p + dVr
    

    def hit_(self):
        hit = False
        C1 = (self.p1.position + self.p2.position)/2
        vc1 = vc2 = vr1 = vr2 = self.box.nullvector.copy()

        dv1 = self.p1.speed - self.vcenter - self.vrot
        dv2 = self.p2.speed - self.vcenter + self.vrot
        print(dv1, dv2)

        if not dv2 @ dv2 < 0.001:
            p2n = self.p2.position + dv2
            dp = (p2n - self.p1.position)
            p1n = p2n - (self.length * dp/math.sqrt(dp @ dp))
            C2 = (p1n + p2n)/2
            vc2 = C2 - C1
            vr2 = (p2n - vc2) - self.p2.position
            hit = True

        if not dv1 @ dv1 < 0.001:
            p1n = self.p1.position + dv1
            dp = (p1n - self.p2.position)
            p2n = p1n - (self.length * dp/math.sqrt(dp @ dp))
            C2 = (p2n + p1n)/2
            vc1 = C2 - C1
            vr1 = (p1n - vc1) - self.p1.position
            hit = True


        self.vcenter += vc1 + vc2
        self.vrot += vr1 + vr2

        return hit

    def hit__(self, p1, p2):
        hit = False
        C1 = (p1.position + p2.position)/2
        vc1 = self.box.nullvector.copy()
        vc2 = self.box.nullvector.copy()
        vr1 = self.box.nullvector.copy()
        vr2 = self.box.nullvector.copy()

        dv = p1.speed - self.vcenter - self.vrot
        if not dv @ dv < 0.0001:
            p1n = p1.position + dv
            dp = (p1n - p2.position)
            p2n = p1n - (self.length * dp/math.sqrt(dp @ dp))
            C2 = (p2n + p1n)/2
            vc1 = C2 - C1
            vr1 = dv - vc1
            # vr2_ = (p2n - p2.position) - vc1
            # print(vr2_ - vr1)
            hit = True


        self.vcenter += vc1 + vc2
        self.vrot += vr1 + vr2

        return hit

    def hit(self, p1, p2):
        hit = False
        C = (p1.position + p2.position) / 2

        vcenter1 = self.box.nullvector.copy()
        vcenter2 = self.box.nullvector.copy()
        vrot1 = self.box.nullvector.copy()
        vrot2 = self.box.nullvector.copy()

        dv = p1.speed - self.vcenter - self.vrot
        if (dv @ dv) > 0.0001:
            vcd1 = (p1.position + p1.speed/2) - C         
            vcenter1 = (p1.speed @ vcd1) * vcd1 / (vcd1 @ vcd1)
            vrot1 = p1.speed - vcenter1
            hit = True

        dv = p2.speed - self.vcenter + self.vrot
        if (dv @ dv) > 0.0001:
            vcd2 = (p2.position + p2.speed/2) - C
            vcenter2 = (p2.speed @ vcd2) * vcd2 / (vcd2 @ vcd2)
            vrot2 = p2.speed - vcenter2
            hit = True

        if hit:
            self.vcenter = vcenter1 + vcenter2
            self.vrot = vrot1 + vrot2
            # vrot = self.vrot.copy()

            # p1n = p1.position + self.vrot
            # dpn = p1n - C
            # p1n = C + (self.length/2) * dpn / (math.sqrt(dpn @ dpn))
            # self.vrot = p1n - p1.position

            # p1n = p1.position + self.vrot
            # p2n = p2.position - self.vrot
            # dpn = p1n - p2n

            # print(math.sqrt(dpn @ dpn), self.length, self.vrot - vrot)
        # print(vcenter1, vrot1, vcenter2, vrot2, self.vcenter, self.vrot)

        return hit

    def rotate(self):
        # self.vcenter = self.box.nullvector.copy()

        # dp2 = self.p1.position - self.p2.position
        dp2 = (self.p1.position + self.vrot) - (self.p2.position - self.vrot)
        vrotp = (self.vrot @ dp2) * dp2 / (dp2 @ dp2)

        self.vrot -= 2*vrotp
    
    def pull(self):
        h1 = h2 = False
        h1 = self.hit(self.p1, self.p2)
        # h2 = self.hit(self.p2, self.p1)
        if not h1:
        # if True:
            self.rotate()
            pass

        self.p1.speed = self.vcenter + self.vrot
        self.p2.speed = self.vcenter - self.vrot
        # print(self.p1.speed, self.p2.speed, self.vcenter, self.vrot)
        

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
                radius = round(mass)
            if rand_c:
                charge = random.randint(-1, 1)

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

        balls = []
        for i in range(vertices):
            pos = center + self.box.random(size)
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, 10, pos, speed, charge)
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
        radius = 5
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
                    spring = Spring(lspring, 0.15, 0.03, b1, b2)
                    self.box.springs.append(spring)
                    if not star:
                        b1 = b2
                if circle:
                    spring = Spring(lspring, 0.01, 0, b1, bstart)
                    self.box.springs.append(spring)
        return balls
    

    def create_pendulum(self, gravity=0.1):
        self.box.set_gravity(gravity)
        balls = []

        pos = self.box.center.copy()
        pos[Box.Y] = self.box.box_sizes[Box.Y] - 150
        speed = self.box.nullvector.copy()
        ancor = self.box.add_particle(1, 1, pos, speed, charge=0, fixed=True, color=[255,255,255])
        balls.append(ancor)
        
        lspring = 150
        pos = pos.copy()
        pos[Box.X] += lspring
        ball1 = self.box.add_particle(10,10, pos, speed, color=[0,255,255])
        balls.append(ball1)

        spring = Spring(lspring, 10, 0.01, ancor, ball1)
        self.box.springs.append(spring)

        lspring = 100
        pos = pos.copy()
        pos += self.box.random(lspring)
        # pos[Box.Z] = self.box.center[Box.Z]
        ball2 = self.box.add_particle(10,10, pos, speed, color=[0,255,255])
        balls.append(ball2)

        spring = Spring(lspring, 10, 0.01, ball1, ball2)
        self.box.springs.append(spring)

        return balls
    
    def test_interaction_simple(self, interaction):
        self.box.set_interaction(interaction)
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

    def test_interaction(self, interaction=20000, M0=40, V0=6, D=140, ratio=0.1):
        self.box.set_interaction(interaction)
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

    def test_spring(self, length=150, distance=240, strength=0.03, interaction=0):
        self.box.set_interaction(interaction)

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[0] = distance/2
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = 0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=-1, color=[255,0,0])
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = -distance/2
        pos = self.box.center + dpos
        speed[0] = -0
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

        rod = Rod(self.box, length, b1, b2, vcenter=None, vrot=None)
        self.box.rods.append(rod)
        
        return balls

def load(path):
    if isinstance(path, str):
        file = open(path)
    else:
        file = path
    
    data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    box = load_gas(data)
    return box

def load_gas(data):
    b = data["box"]
    box = Box(b["sizes"])
    try:
        box.gravity = numpy.array(b['gravity'])
        box.set_friction(b['friction'])
        box.set_interaction(b['interaction'])
        box.torus = b['torus']
    except:
        pass

    for p in b['particles']:
        try:
            fixed = p["fixed"]
        except KeyError:
            fixed = False
        
        try: 
            charge = p['charge']
        except KeyError:
            charge = 0

        try:
            color = p['color']
        except KeyError:
            color = None
        box.add_particle(p['mass'], p['radius'], p['position'], p['speed'], charge, fixed, color)
    
    for s in b['springs']:
        ps = s['particles']
        p1 = box.particles[ps[0]]
        p2 = box.particles[ps[1]]
        try: 
            damping = s["damping"] 
        except KeyError: 
            damping = 0
        spring = Spring(s['length'], s['strength'], damping, p1, p2)
        box.springs.append(spring)

    return box

def save(box, path):
    if isinstance(path, str):
        file = open(path)
    else:
        file = path
    out = box.out()

    yaml.dump(out, file, canonical=False, Dumper=yaml.Dumper, default_flow_style=False)
    file.close()

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



    def test_wall(self):
        D = 1
        box = Box([10,20,30])
        print(box.vertices)
        print(box.edges)
        print()
        wall = Wall(box, 0.5, D)
        print(box.box_sizes)
        print(wall._vector, wall.position)
        print("wall: ", wall)
        center = sum(wall.vertices)/len(wall.vertices)
        coords = [[vertex[0],vertex[1]] for vertex in wall.vertices]
        print("coords: ", coords)
        size = numpy.array(max(coords))

        print("center: ", center)

        try:
            size[D] = -1
        except IndexError:
            pass
        print("size: ", size)

        print()
        print(wall.vertices.mean(axis=D))
        pass

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

        save(box, FILE)

    def test_load_yaml(self):
        FILE = "D:\\temp\\box.yml"
        box = load(FILE)
        print(box)

        for p in box.particles:
            print(p)
        
        for s in box.springs:
            print(s)
        
        for i in range(100):
            box.go()
            print(box.ticks)
    
    def normal(self):
        sizes = [100,100]
        box = Box(sizes)
        print(box.vertices)
        print(box.edges)
        print(box.axis)
        for p in box.planes:
            print(p)

        return 

        n = linalg.solve(box.axis, box.onevector)
        N = n / math.sqrt(n@n)
        print(n, N, N@N)

        points = []
        for i in range(box.dimensions):
            p = box.random_position()
            points.append(p)


        print("\n######\n")
        print(points)

        n = linalg.solve(points, box.onevector)
        N = n/math.sqrt(n@n)
        print(N)
        p = points[-1]
        for q in points:
            d = p - q
            print(d@N)
            p = q

        # print(box.edges)
        

if __name__ == "__main__": 
    print("START")

    t = Test()
    
    # t.test_wall()
    # t.test_box()
    # t.test()
    # t.test_save_yaml()
    # t.test_load_yaml()
    # t.test_displacement()
    # t.normal()
    t.normal()

    print("END")