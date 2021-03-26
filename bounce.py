#import itertools
#import numpy
import random
import time
import pygame
from gas import *

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
            box.go()

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
    print(box.edges)

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

