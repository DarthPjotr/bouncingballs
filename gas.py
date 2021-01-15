"""
Test
"""
import itertools
import numpy
import random
import math

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 500
SCREEN_DEPTH = 400

DIMENSIONS=2

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0 ,0)
GREEN = (0, 255,0)

VMAX = 5
RADIUS = 10
MASS = 10
NBALLS = 30


class VectorField:
    def __init__(self, box) -> None:
        self.box = box
        self.field = self.nofield
        super().__init__()
    
    def apply(self, particle):
        (vector, effect) = self.field(particle.mass, particle.position, particle.speed)
        if effect == "add":
            particle.speed += vector
        elif effect == "flow":
            particle.speed = math.sqrt(particle.speed.dot(particle.speed)) * vector
        elif effect == "mul":
            particle.speed = particle.speed * vector
        elif effect == "rot":
            particle.speed = numpy.matmul(vector, particle.speed)
        else:
            pass
        return particle.speed
    
    def getvalue(self, position):
        (vector, effect) = self.field(0, position, self.box.nullvector)
        if effect == "rot":
            vector = numpy.matmul(vector, self.box.nullvector)
        return vector

    def nofield(self, mass, position, speed):
        effect = "add"
        position = self.box.nullvector
        speed = self.box.nullvector
        return (self.box.nullvector, effect)

    def gravity(self, mass, position, speed):
        effect = "add"
        dspeed = self.box.nullvector
        dspeed[1] = -0.1
        return (dspeed, effect)

    def rotate_flow(self, mass, position, speed):
        effect = "flow"
        center = self.box.box_sizes / 2
        v0 = position - center
        
        u0 = v0/math.sqrt(v0.dot(v0))
        dspeed = numpy.array([u0[1], -u0[0]])
        return (dspeed, effect)
    
    def rotate(self, mass, position, speed):
        effect = "rot"
        theta = math.radians(mass*speed.dot(speed)/100)
        c = math.cos(theta)
        s = math.sin(theta)
        M = numpy.array(((c, -s), (s, c)))
 
        return (M, effect)

    def sinkR(self, mass, position, speed):
        effect = "add"
        dspeed = self.box.nullvector
        center = self.box.box_sizes / 2
        r = 20
        v0 = position - center
        v0dot = v0.dot(v0)
        if abs(v0dot) > r*r:
            u0 = v0/math.sqrt(v0dot)
            dspeed = -50*u0/math.sqrt(v0dot)
        return (dspeed, effect)

    def sinkRR(self, mass, position, speed):
        effect = "add"
        dspeed = self.box.nullvector
        center = self.box.box_sizes / 2
        r = 20
        v0 = position - center
        v0dot = v0.dot(v0)
        if abs(v0dot) > r*r:
            u0 = v0/math.sqrt(v0dot)
            dspeed = -2000*u0/v0dot
        return (dspeed, effect)


class Box:
    def __init__(self, box_sizes) -> None:
        self.box_sizes = numpy.array(box_sizes, dtype=float)
        self.dimensions = len(self.box_sizes)
        self.vertices = []
        self.egdes = []
        self._get_vertices()
        self._get_edges()
        self.particles = []
        self.unitvector = numpy.array([1.0]*self.dimensions)
        self.nullvector = self.unitvector * 0
        self.impuls = self.nullvector.copy()
        self.vectorfield = None

    def _get_vertices(self):
        # get unit cube coordinates for dimensions of box
        unit_cube = numpy.array(list(itertools.product([0,1], repeat=self.dimensions)))
        # vector multiply with box sizes
        self.vertices = unit_cube*self.box_sizes

    def _get_edges(self):
        for i in range(len(self.vertices)):
            for j in range(i+1, len(self.vertices)):
                v1 = self.vertices[i]
                v2 = self.vertices[j]
                c = 0
                for k in range(len(v1)):
                    if v1[k] == v2[k]:
                        c +=1
                if c==self.dimensions-1:
                    self.egdes.append((i,j))

    def __str__(self) -> str:
        return str(self.box_sizes)

    def resize(self, new_sizes):
        self.box_sizes[0:len(new_sizes)] = new_sizes
        self._get_vertices()
        self._get_edges()
    
    def add_particle(self,mass=MASS, radius=RADIUS, position=None, speed=None, color=None):
        if position == None:
            position = [random.randrange(radius, x - radius)*1.0 for x in self.box_sizes]
        if speed == None:
            speed = [random.randrange(-VMAX,VMAX)*1.0 for dummy in range(self.dimensions)]
        if color == None:
            color = (random.randrange(256), random.randrange(256), random.randrange(256))
        particle = Particle(mass, radius, position, speed, color)
        self.particles.append(particle)
        return particle

    def go(self):
        bounced = False
        for i, ball in enumerate(self.particles):
            #dspeed = self.applyfield(ball)
            #ball.speed += dspeed
            self.vectorfield.apply(ball)
            # Move the ball's center
            ball.move()
            # Bounce the ball if needed
            ball.bounce(self)
        for i, ball in enumerate(self.particles):
            for ball2 in self.particles[i:]:
                if ball.collide(ball2): bounced = True
        return bounced


class Particle:
    def __init__(self, mass, radius, position, speed, color) -> None:
        self.mass = mass
        self.radius = radius
        self.position = 1.0*numpy.array(position)
        self.speed = 1.0*numpy.array(speed)
        self.color = color
        self.object = None
        self.impuls = self.mass * self.speed
    
    def move(self):
        self.position += self.speed
    
    def collide(self, p2):
        """
        Based on angle free representation from: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        """
        collided = False
        dpos = self.position - p2.position
        distance2 = dpos.dot(dpos)

        # only collide if particles are moving towards each other: dot product of speed difference and position different < 0
        dspeed = self.speed - p2.speed
        dot = dspeed.dot(dpos)

        dmin = (self.radius + p2.radius)
        if distance2 > 0 and distance2 < dmin*dmin and dot < 0: # and d2 < distance2:
            ds = dot*dpos/distance2
            # s1 = self.speed - (2*p2.mass/(self.mass+p2.mass)) * dspeed.dot(dpos)*(dpos)/distance2
            # s2 = p2.speed - (2*self.mass/(self.mass+p2.mass)) * -dspeed.dot(-dpos)*(-dpos)/distance2
            s1 = self.speed - (2*p2.mass/(self.mass+p2.mass)) * ds
            s2 = p2.speed - (2*self.mass/(self.mass+p2.mass)) * -ds
            self.speed = s1
            p2.speed = s2
            collided = True
        
        self.impuls = self.mass * self.speed
        return collided
        
    def bounce(self, box):
        bounced = False
        old_speed = self.speed.copy()
        for i, x in enumerate(box.nullvector):
            if self.position[i] < x + self.radius: # and self.speed[i] < 0:
                self.speed[i] = abs(self.speed[i])
                self.position[i] = x + self.radius
                bounced = True
        for i, x in enumerate(box.box_sizes):
            if self.position[i] > x - self.radius: # and self.speed[i] > 0:
                self.speed[i] = -abs(self.speed[i])
                self.position[i] = x - self.radius
                bounced = True
        self.impuls = self.mass * self.speed
        box.impuls += self.mass * (old_speed - self.speed)
        return bounced
    
    def check_inside(self, coords):
        inside = False

        for i, x in enumerate(coords):
            if x > self.position[i] - self.radius and x < self.position[i] + self.radius:
                inside = True
        return inside

    def energy(self):
        return self.mass * self.speed.dot(self.speed) 

    def __str__(self) -> str:
        pstr = ""
        pstr = "particle:\n radius: {}\nposition: {}\nspeed: {}".format(self.radius, self.position, self.speed)
        return pstr 
