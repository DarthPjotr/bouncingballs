"""
Ideal gas in n-dimensional box
"""
import itertools
import numpy
import random
import math

# SCREEN_WIDTH = 700
# SCREEN_HEIGHT = 500
# SCREEN_DEPTH = 400

# DIMENSIONS=2

# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# BLUE = (0, 0, 255)
# RED = (255, 0 ,0)
# GREEN = (0, 255,0)

VMAX = 5
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
        self.unitvector = numpy.array([1.0]*self.dimensions)
        self.nullvector = self.unitvector * 0
        self.vertices = []
        self.egdes = []
        self.axis = []
        self._get_vertices()
        self._get_edges()
        self._get_axis()
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
        # dynamic properties
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
                    self.egdes.append((i,j))
    
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

    def __str__(self) -> str:
        return str(self.box_sizes)
    
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
        self.ticks = 1
        self._normal_momentum = 0
    
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
    
    def add_particle(self,mass=MASS, radius=RADIUS, position=None, speed=None, charge=0, color=None):
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
        particle = Particle(self, mass, radius, position, speed, charge, color)
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
        if strength != 0 and direction is None:
            direction = self.nullvector.copy()
            try:
                direction[1]=-1
            except IndexError:
                pass
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

    def go(self):
        """
        Moves all particles to next position

        Returns:
            boolean: true if particles collides into each other
        """
        bounced = False
        self.ticks += 1
    
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
            # friction
            if self.friction != 0:
                self.slide(ball)
            # apply field
            if self.field is not None:
                self.field.equation(ball=ball)
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
        
        # move all balls
        for ball in self.particles:
            ball.move()
        
        return bounced


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
        self._vector = self.box.unitvector.copy()
        self._vector[self.dimension] *= self.rpos
        self.position = self.box.box_sizes * self._vector
        self.vertices = numpy.array([vertix*self._vector for vertix in self.box.vertices if vertix[self.dimension] != 0]) 
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
    def __init__(self, box: Box, mass: float, radius: float, position: list, speed: list, charge: float, color: list) -> None:
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
        self.color = color
        self.object = None
        # self.impuls = self.mass * self.speed
    
    def move(self):
        """
        Moves the particle
        """
        self.position += self.speed

    def fast_collision_check(self, p2):
        """
        Fast collision detection

        Args:
            p2 (Particle): particle to check collision with

        Returns:
            boolean: True when possible collision occured
        """
        min_distance = self.radius + p2.radius
        dposition = abs(self.displacement(p2.position))
        #dposition = abs(self.position - p2.position)

        for dpos in dposition:
            if dpos > min_distance:
                return False
        
        if (sum(dposition) > min_distance*min_distance):
            return False
        return True
    
    def collide(self, p2):
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

        if not self.fast_collision_check(p2):
            collided = False
            return collided

        # dposition = self.position - p2.position
        dposition = self.displacement(p2.position)
        distance2 = dposition.dot(dposition)

        # only collide when particles are moving towards each other: 
        # dot product of speed difference and position different < 0
        dspeed = self.speed - p2.speed
        dot_speed_pos = dspeed.dot(dposition)

        dmin = self.radius + p2.radius
        if distance2 > 0 and distance2 < dmin*dmin and dot_speed_pos < 0: # and d2 < distance2:
            dspeed_new = dot_speed_pos*dposition/distance2
            # speed1 = self.speed - (2*p2.mass/(self.mass+p2.mass)) * dspeed.dot(dpos)*(dpos)/distance2
            # speed2 = p2.speed - (2*self.mass/(self.mass+p2.mass)) * -dspeed.dot(-dpos)*(-dpos)/distance2
            speed1 = self.speed - (2*p2.mass/(self.mass + p2.mass)) * dspeed_new
            speed2 = p2.speed - (2*self.mass/(self.mass + p2.mass)) * -dspeed_new
            self.speed = speed1
            p2.speed = speed2
            collided = True
        
        # self.impuls = self.mass * self.speed
        return collided
        
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
        if self.box.interaction == 0 or self.charge == 0:
            return self.speed
        dspeed = self.box.nullvector.copy()
        for p in self.box.particles:
            if p == self or p.charge == 0:
                continue
            dpos = self.position - p.position
            distance2 = dpos.dot(dpos)
            charge = self.charge*p.charge
            if distance2 < (self.radius+p.radius)*(self.radius+p.radius):
                charge = abs(charge)
            if distance2 > 0:
                N = dpos/math.sqrt(distance2)
                dspeed += charge*self.box.interaction*N/(self.mass*distance2)
        
        self.speed += dspeed
        return self.speed
    
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

class Spring:
    """
    Spring between particles
    """
    def __init__(self, length: float, strength: float, damping: float, p1: Particle, p2: Particle) -> None:
        """
        Creates spring betrween two Particles

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
        Kinetic energy of spring

        Returns:
            float: kinetic energy
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

class ArrangeParticles:
    def __init__(self, box: Box) -> None:
        self.box = box

    def random_balls(self, nballs: int, mass=None, radius=None, max_speed=VMAX):
        if mass is None:
            mass = random.randrange(0,10) * 1.0
        if radius is None:
            radius =  mass
        balls = []
        for i in range(nballs):
            speed = self.box.random(max_speed)
            ball = self.box.add_particle(mass, radius, None, speed, 0, None)
            balls.append(ball)

        return balls
    
    def create_simplex(self, size=200, position=None, charge=0, nedges=None):
        if position is None:
            center = self.box.center
        else:
            center = position

        if nedges is None:
            nedges = self.box.dimensions+1

        balls = []
        for i in range(nedges):
            pos = center + self.box.random(size)
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, 10, pos, speed, charge)
            balls.append(ball)

        balls[0].speed = 5 * self.box.unitvector.copy()
        balls[-1].speed = -5 * self.box.unitvector.copy()

        for i, b1 in enumerate(balls):
            for b2 in balls[i:]:
                if b1 != b2:               
                    spring = Spring(size, 0.01, 0.01, b1, b2)
                    self.box.springs.append(spring)
        return balls

    def create_box(self, size, position=None, charge=0):
        ratio = max(self.box.box_sizes)/size
        sizes = self.box.box_sizes/ratio
        if position is None:
            center = self.box.box_sizes/2
        else:
            center = position
        box = Box(sizes)
        speed = self.box.nullvector.copy()

        balls = []
        for vertix in box.vertices:
            pos = center - (box.box_sizes/2) + vertix
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, 10, pos, speed, charge)
            balls.append(ball)
        
        balls[0].speed = 5 * self.box.unitvector.copy()
        balls[-1].speed = -5 * self.box.unitvector.copy()

        l = sum(box.box_sizes)/box.dimensions
        for edge in box.egdes:
            spring = Spring(l, 0.01, 0.01, balls[edge[0]], balls[edge[1]])
            self.box.springs.append(spring)
        
        return balls

    def create_n_mer(self, nballs, n=2, star=False, circle=False, charge=0):
        radius = 5
        lspring = 20
        balls = []
        for i in range(round(nballs/n)):
            pos1 = self.box.random_position()
            speed = self.box.random() * VMAX * random.random()
            b1 = self.box.add_particle(1, radius, pos1, speed, charge)
            bstart = b1
            balls.append(b1)
            if n > 1:
                for i in range(n-1):
                    pos2 = pos1 + self.box.random() * (lspring + 10)
                    speed2 = speed + self.box.random()
                    b2 = self.box.add_particle(1, radius, pos2, speed2, charge)
                    balls.append(b2)
                    spring = Spring(lspring, 0.01, 0, b1, b2)
                    self.box.springs.append(spring)
                    if not star:
                        b1 = b2
                if circle:
                    spring = Spring(lspring, 0.01, 0, b1, bstart)
                    self.box.springs.append(spring)
        return balls
    
    def test_interaction(self, interaction):
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

def test_wall():
    D = 1
    box = Box([10,20,30])
    print(box.vertices)
    print(box.egdes)
    print()
    wall = Wall(box, 0.5, D)
    print(box.box_sizes)
    print(wall._vector, wall.position)
    print("wall: ", wall)
    center = sum(wall.vertices)/len(wall.vertices)
    coords = [[vertix[0],vertix[1]] for vertix in wall.vertices]
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

def test_box():
    box = Box([800,600,700])
    print(box.vertices)
    print(box.egdes)
    print(box.volume())
    print(box.area())
    print(box.axis)


def test():
    BOX_DIMENSIONS = [500, 600, 700, 400, 300]
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

    filling = ArrangeParticles(box)
    filling.random_balls(10, 10)

    for i in range(1000):
        try:
            box.go()
            K = sum([ball.energy for ball in box.particles])
            E = K * (2 / DIMENSIONS) # https://en.wikipedia.org/wiki/Kinetic_theory_of_gases#Pressure_and_kinetic_energy
            PV = box.pressure()*box.volume()
            print("{:.2f} {:.2f} {:.2f}".format(PV, E, PV/E))
        except KeyboardInterrupt:
            print("user interupted")
            break

def test_displacement():
    box = Box([10, 20])
    p1 = numpy.array([1,2])
    p2 = numpy.array([2,19])  
    p3 = numpy.array([9,19])
    box.torus = True

    d = box.displacement(p1, p2)
    print(d, p1 - p2)

    d = box.displacement(p1, p3)
    print(d, p1 - p3)


if __name__ == "__main__": 
    print("START")
    
    # test_wall()
    # test_box()
    test()
    # test_displacement()

    print("END")