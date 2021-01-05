"""
Test
"""
import itertools
import numpy
import random
import time
import pygame

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
RADIUS = 50
MASS = 10
NBALLS = 20

class Box:
    def __init__(self, box_sizes) -> None:
        self.box_sizes = box_sizes
        self.dimensions = len(self.box_sizes)
        self.vertices = []
        self.egdes = []
        self._get_vertices()
        self._get_edges()
        self.particles = []

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
                    #print(v1, v2, "edge")

    def __str__(self) -> str:
        return str(self.box_sizes)

    def add_particle(self,mass=MASS, radius=RADIUS, position=None, speed=None):
        if position == None:
            position = [random.randrange(radius, x - radius)*1.0 for x in self.box_sizes]

        if speed == None:
            speed = [random.randrange(-VMAX,VMAX)*1.0 for dummy in range(self.dimensions)]
        particle = Particle(mass, radius, position, speed)
        self.particles.append(particle)

    def resize(self, new_sizes):
        self.box_sizes[0:len(new_sizes)] = new_sizes
        self._get_vertices()
        self._get_edges()

    
    def go(self):
        for i, p in enumerate(self.particles):
            p.move()
            bounced = p.bounce(self)
            print("{}: , {}, {}".format(i, p.position, bounced))


class Particle:
    def __init__(self, mass, radius, position, speed) -> None:
        self.mass = mass
        self.radius = radius
        self.position = numpy.array(position)
        self.speed = numpy.array(speed)
    
    def move(self):
        self.position += self.speed
    
    def collide(self, p2):
        collided = False
        dpos = self.position - p2.position
        distance2 = dpos.dot(dpos)

        # only collide if particles are moving towards each other, so check next position
        #dpos2 = (self.position + self.speed) - (p2.position + p2.speed)
        #d2 = dpos2.dot(dpos2)

        dmin = (self.radius + p2.radius)
        if distance2 > 0 and distance2 < dmin*dmin: # and d2 < distance2:
            dspeed = self.speed - p2.speed
            dot = dspeed.dot(dpos)
            if dot > 0:
                return False
            s = dot*dpos/distance2
            #s1 = self.speed - (2*p2.mass/(self.mass+p2.mass)) * dspeed.dot(dpos)*(dpos)/distance2
            s1 = self.speed - (2*p2.mass/(self.mass+p2.mass)) * s
            #s2 = p2.speed - (2*self.mass/(self.mass+p2.mass)) * -dspeed.dot(-dpos)*(-dpos)/distance2
            s2 = p2.speed - (2*self.mass/(self.mass+p2.mass)) * -s
            self.speed = s1
            p2.speed = s2
            collided = True
        return collided
        
    def bounce(self, box):
        bounced = False
        for i, x in enumerate(len(box.box_sizes)*[0]):
            if self.position[i] < x + self.radius: # and self.speed[i] < 0:
                self.speed[i] = abs(self.speed[i])
                bounced = True
        for i, x in enumerate(box.box_sizes):
            if self.position[i] > x - self.radius: # and self.speed[i] > 0:
                self.speed[i] = -abs(self.speed[i])
                bounced = True
        return bounced

    def energy(self):
        return self.mass * self.speed.dot(self.speed) 

    def __str__(self) -> str:
        pstr = ""
        pstr = "particle:\n radius: {}\nposition: {}\nspeed: {}".format(self.radius, self.position, self.speed)
        return pstr 

def getcolors(steps):
    colors = []
    for i in range(0,255, round(255/steps)):
        color = (255-i,0,i)
        colors.append(color)
    return colors
       

def paint():
    """
    This is our main program.
    """
    pygame.init()

    # Set the height and width of the screen
    size = [SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DEPTH, 100]
    screen = pygame.display.set_mode(size[:2], pygame.RESIZABLE)

    pygame.display.set_caption("Bouncing Balls")
    #colors = getcolors(NBALLS)

    # Loop until the user clicks the close button.
    done = False
    pause = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    box_sizes = size[:DIMENSIONS]
    #box_sizes.append(100)
    #box_sizes.append(SCREEN_DEPTH)
    #print(box_sizes)
    box = Box(box_sizes)
    # ball in border
    #box.add_particle(MASS, RADIUS, [-5,size[1]/2], [-1,0])
    # ball overlap
    #box.add_particle(MASS, RADIUS, [size[0]/2,size[1]/2], [1,0])
    #box.add_particle(MASS, RADIUS, [size[0]/2+RADIUS/2,size[1]/2], [-1,1])
    for i in range(NBALLS):
        #mass = MASS*random.random()+1
        #radius = mass*5
        box.add_particle(mass=MASS, radius=RADIUS)

    # -------- Main Program Loop -----------
    while not done:
        # --- Event Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                # Space bar! Spawn a new ball.
                if event.key == pygame.K_SPACE:
                    #mass = MASS*random.random()+1
                    #radius = mass*5
                    box.add_particle(mass=MASS, radius=RADIUS)
                elif event.key == pygame.K_p:
                    pause = not pause
                elif event.key == pygame.K_q:
                    done = True
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode([event.w, event.h],
                                              pygame.RESIZABLE)
                box.resize([event.w, event.h])

        if not pause:
            # --- Logic
            for i, ball in enumerate(box.particles):
                # Move the ball's center
                ball.move()
                # Bounce the ball if needed
                ball.bounce(box)
            for i, ball in enumerate(box.particles):
                for ball2 in box.particles[i:]:
                    ball.collide(ball2)

            #print(sum([ball.energy() for ball in box.particles]))
        
            # --- Drawing
            # Set the screen background
            screen.fill(BLACK)

            # Draw the balls
            for i, ball in enumerate(box.particles):
                if box.dimensions > 2:
                    radius = ball.radius*(ball.position[-1]/SCREEN_DEPTH)
                else:
                    radius = ball. radius
                color = WHITE
                if i == 0:
                    color = RED
                elif i == box.dimensions-1:
                    color = GREEN
                elif False:
                    idx = round(ball.speed.dot(ball.speed)) % len(colors)
                    #print(idx, ball.speed.dot(ball.speed))
                    color = colors[idx]

                pygame.draw.circle(screen, color, ball.position[:2], radius)
                pygame.draw.line(screen, BLUE, ball.position[:2], (ball.position + 5*ball.speed)[:2], width=2)

        # --- Wrap-up
        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        if not pause:
            pygame.display.flip()

    # Close everything down
    pygame.quit()

def main():
    
    box_sizes = [50,100]
    #box_sizes = [1,2]
    box = Box(box_sizes)
    # print(box.vertices)
    print(box)
    print(box.vertices)
    for i, v in enumerate(box.vertices):
    	print(i, v)
    print(box.egdes)

    box.add_particle()
    box.add_particle()
    for p in box.particles:
        print(p)

    try:
        while True:
            box.go()
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    random.seed()
    paint()
    # main()

