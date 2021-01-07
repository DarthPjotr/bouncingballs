#import itertools
#import numpy
import random
import time
import pygame
from gas import *
     

def paint():
    """
    This is our main program.
    """
    pygame.init()

    # Set the height and width of the screen
    size = [SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DEPTH, 100]
    screen = pygame.display.set_mode(size[:2], pygame.RESIZABLE)

    pygame.display.set_caption("Bouncing Balls")

    # Loop until the user clicks the close button.
    done = False
    pause = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    box_sizes = size[:DIMENSIONS]

    box = Box(box_sizes)

    # -------- Main Program Loop -----------
    while not done:
        # --- Event Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                # Space bar! Spawn a new ball.
                if event.key == pygame.K_SPACE:
                    mass = random.randrange(5, 50)
                    box.add_particle(mass=mass, radius=mass)
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
       
            # --- Drawing
            # Set the screen background
            screen.fill(BLACK)

            # Draw the balls
            for i, ball in enumerate(box.particles):
                pygame.draw.circle(screen, ball.color, ball.position[:2], ball.radius)
                pygame.draw.line(screen, BLUE, ball.position[:2], (ball.position + 5*ball.speed)[:2], width=2)

        # --- Wrap-up
        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        if not pause:
            pygame.display.flip()

    # Close everything down
    pygame.quit()

if __name__ == "__main__":
    random.seed()
    paint()
    # main()

