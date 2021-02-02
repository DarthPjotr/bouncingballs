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


class Field:
    def __init__(self, box) -> None:
        self.box = box
        self.field = self.nofield
        super().__init__()
    
    def apply(self, particle):
        (value, effect) = self.field(particle.mass, particle.position, particle.speed)
        if effect == "add":
            particle.speed += value
        elif effect == "flow":
            particle.speed = math.sqrt(particle.speed.dot(particle.speed)) * value
        elif effect == "mul":
            particle.speed = particle.speed * value
        elif effect == "rot":
            particle.speed = numpy.matmul(value, particle.speed)
        elif effect == "replace":
            particle.speed = value
        else:
            pass
        return particle.speed
    
    def getvalue(self, position):
        (value, effect) = self.field(0, position, self.box.nullvector)
        if effect == "rot":
            value = numpy.matmul(value, self.box.nullvector)
        return value

    def nofield(self, mass, position, speed):
        effect = "add"
        position = self.box.nullvector
        speed = self.box.nullvector
        return (speed, effect)

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
    
    def friction(self, mass, position, speed):
        effect = "replace"
        speed -= speed * 0.01
        return (speed, effect)


class Box:
    def __init__(self, box_sizes) -> None:
        self.box_sizes = numpy.array(box_sizes, dtype=float)
        self.dimensions = len(self.box_sizes)
        self.vertices = []
        self.egdes = []
        self._get_vertices()
        self._get_edges()
        self.center = sum(self.vertices)/len(self.vertices)
        self.wall = None
        self.torus = False
        self.particles = []
        self.unitvector = numpy.array([1.0]*self.dimensions)
        self.nullvector = self.unitvector * 0
        self.impuls = self.nullvector.copy()
        self.field = None
        self.gravity = self.nullvector.copy()
        self.friction = 0.0
        self.interaction = 0.0
        self.ticks = 0

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
                        c += 1
                if c == self.dimensions-1:
                    self.egdes.append((i,j))

    def __str__(self) -> str:
        return str(self.box_sizes)

    def resize(self, new_sizes):
        self.box_sizes[0:len(new_sizes)] = new_sizes
        self._get_vertices()
        self._get_edges()
    
    def volume(self):
        V = numpy.product(self.box_sizes)
        return V
    
    def area(self):
        V = numpy.product(self.box_sizes)
        A = 2*sum([V/s for s in self.box_sizes])
        return A
    
    def add_particle(self,mass=MASS, radius=RADIUS, position=None, speed=None, charge=0, color=None):
        if position is None:
            position = []
        rpos = [random.randrange(radius, x - radius)*1.0 for x in self.box_sizes]
        position.extend(rpos[len(position):])

        if speed is None:
            speed = []
        rspeed = [random.randrange(-VMAX,VMAX)*1.0 for dummy in range(self.dimensions)]
        speed.extend(rspeed[len(speed):])

        if color is None:
            color = (random.randrange(256), random.randrange(256), random.randrange(256))
        particle = Particle(mass, radius, position, speed, charge, color)
        self.particles.append(particle)
        return particle
    
    def set_gravity(self, strength, direction=None):
        if direction is None:
            direction = self.nullvector.copy()
            direction[1]=-1
        self.gravity = strength * direction
        return self.gravity
    
    def fall(self, particle):
        particle.speed += self.gravity
        return particle

    def set_friction(self, friction):
        self.friction = friction
        return self.friction
    
    def slide(self, particle):
        particle.speed -= particle.speed*self.friction
        return particle
    
    def set_interaction(self, interaction):
        self.interaction = interaction
        return self.interaction
    
    def avg_impuls(self):
        if len(self.particles) == 0:
            return self.nullvector
        tot_impuls = sum([ball.impuls for ball in self.particles]) + self.impuls
        return tot_impuls/len(self.particles)

    def avg_position(self):
        if len(self.particles) == 0:
            return self.nullvector
        tot_position = sum([ball.position for ball in self.particles])
        return tot_position/len(self.particles)

    def go(self):
        bounced = False
        self.ticks += 1

        # first calculate speed
        for i, ball in enumerate(self.particles):
            # interaction
            ball.interact(self)
            # gravity
            self.fall(ball)
            # friction
            self.slide(ball)
            # apply field
            if self.field is not None:
                self.field.apply(ball)
            # Bounce or wrap the ball if needed
            if self.torus:
                ball.wrap(self)
            else:
                ball.bounce(self)      
            #hit the wall:
            if self.wall is not None:
                ball.hit(self.wall)
            # collide the balls
            for ball2 in self.particles[i:]:
                if ball.collide(ball2): bounced = True
        
        # move all balls
        for ball in self.particles:
            ball.move()
        
        return bounced


class Wall:
    def __init__(self, box: Box, rpos: int, dimension: int) -> None:
        self.box = box
        if rpos < 0 or rpos > 1:
            raise ValueError("rpos must be between 0 and 1")
        self.dimension = dimension
        self.rpos = rpos
        self._properties()
    
    def _properties(self):
        self._vector = self.box.unitvector.copy()
        self._vector[self.dimension] *= self.rpos
        self.position = self.box.box_sizes * self._vector
        self.vertices = numpy.array([vertix*self._vector for vertix in self.box.vertices if vertix[self.dimension] != 0]) 
        self.center = sum(self.vertices)/len(self.vertices)
    
    def __str__(self) -> str:
        return str(self.vertices)

    def move(self, new_rpos):
        self.rpos = new_rpos
        self._properties()


    def hit(self, ball):
        bounced = False
        dwall = ball.position[self.dimension] - self.position[self.dimension]
        new_pos = dwall + ball.speed[self.dimension]
        #print(ball, dwall, new_pos, (dwall < 0 and new_pos > 0) or (dwall > 0 and new_pos < 0))
        if (dwall <= 0 and new_pos > 0) or (dwall >= 0 and new_pos < 0):
            old_speed = ball.speed.copy()
            ball.speed[self.dimension] = -ball.speed[self.dimension]
            ball.position += ball.speed
            self.box.impuls += ball.mass * (old_speed - ball.speed)
            bounced = True
        
        return bounced


class Particle:
    def __init__(self, mass, radius, position, speed, charge, color) -> None:
        self.mass = mass
        self.radius = radius
        self.position = numpy.array(position, dtype=float)
        self.speed = numpy.array(speed, dtype=float)
        self.charge = charge
        self.color = color
        self.object = None
        # self.impuls = self.mass * self.speed
    
    def move(self):
        self.position += self.speed

    def fast_collision_check(self, p2):
        min_distance = self.radius + p2.radius
        dposition = abs(self.position - p2.position)

        for dpos in dposition:
            if dpos > min_distance:
                return False
        
        if (sum(dposition) > min_distance*min_distance):
            return False
        return True
    
    def collide(self, p2):
        """
        Based on angle free representation from: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        using vectors generalizes to any number of dimensions
        """
        collided = False

        if not self.fast_collision_check(p2):
            collided = False
            return collided

        dposition = self.position - p2.position
        distance2 = dposition.dot(dposition)

        # only collide if particles are moving towards each other: 
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
        # self.impuls = self.mass * self.speed
        box.impuls += self.mass * (old_speed - self.speed)
        return bounced

    def wrap(self, box):
        wrapped = False
        for i, x in enumerate(box.nullvector):
            if self.position[i] < x:
                self.position[i] = box.box_sizes[i] + self.position[i]
                wrapped = True
        for i, x in enumerate(box.box_sizes):
            if self.position[i] > x:
                self.position[i] = self.position[i] - box.box_sizes[i]
                wrapped = True
        return wrapped

    def hit(self, wall):
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
    
    def interact(self, box):
        if box.interaction == 0:
            return self.speed
        dspeed = box.nullvector.copy()
        for p in box.particles:
            if p == self:
                continue
            dpos = self.position - p.position
            distance2 = dpos.dot(dpos)
            charge = self.charge*p.charge
            if distance2 < (self.radius+p.radius)*(self.radius+p.radius):
                charge = abs(charge)
            if distance2 > 0:
                N = dpos/math.sqrt(distance2)
                dspeed += charge*box.interaction*N/(self.mass*distance2)
        
        self.speed += dspeed
        return self.speed
    
    def check_inside(self, coords):
        inside = False

        for i, x in enumerate(coords):
            if x > self.position[i] - self.radius and x < self.position[i] + self.radius:
                inside = True
        return inside

    @property
    def energy(self):
        return self.mass * self.speed.dot(self.speed) 
    
    @property
    def impuls(self):
        return self.mass * self.speed

    def __str__(self) -> str:
        pstr = ""
        pstr = "particle:\n mass: {}\n radius: {}\n position: {}\n speed: {}".format(self.mass, self.radius, self.position, self.speed)
        return pstr 


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
    box = Box([10,20,30])
    print(box.vertices)
    print(box.egdes)
    print(box.volume())
    print(box.area())



if __name__ == "__main__":
    # test_wall()
    test_box()