"""
Bounce balls on the screen.
Spawn a new ball for each mouse-click.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.bouncing_balls
"""

import arcade
import random
import numpy
import math

from gas import *

# --- Set up the constants

# Size of the screen
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Bouncing Balls Example"

   
def gravity(box, mass, position, speed):
    dspeed = numpy.array([0.0]*box.dimensions)
    dspeed[1] = -0.1
    return dspeed

def rotate(box, mass, position, speed):
    center = numpy.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])
    v0 = position - center
    
    u0 = v0/math.sqrt(v0.dot(v0))
    dspeed = numpy.array([u0[1], -u0[0]])
    return dspeed

def sink(box, mass, position, speed):
    center = numpy.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])
    v0 = position - center
    u0 = v0/math.sqrt(v0.dot(v0))
    dspeed = -2000*u0/v0.dot(v0)
    return dspeed

class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)
        self.box = Box([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.box.vectorfield = None
        self.pause = False
        self.sound = arcade.load_sound(".\\sounds\\c_bang1.wav")
        self.bounced = False
        #self.impulses = VectorSum(numpy.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2]), [self.box.nullvector])
    
    def draw_field(self):
        if self.box.vectorfield == None:
            return
        gx = numpy.arange(0, self.box.box_sizes[0], 80)
        gy = numpy.arange(0, self.box.box_sizes[1], 80)
        for x in gx:
            for y in gy:
                speed = 100*self.box.vectorfield(self.box, 0, [x,y], self.box.nullvector)
                arcade.draw_line(x, y, x+speed[0], y+speed[1], [255,255,255], 1)
                arcade.draw_circle_filled(x+speed[0], y+speed[1], 2, [255,255,255])

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        arcade.start_render()

        self.draw_field()

        for ball in self.box.particles:
            arcade.draw_circle_filled(ball.position[0], ball.position[1], ball.radius, ball.color)
            end = ball.position + 5*ball.speed
            arcade.draw_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 1)
        
        #self.impulses.draw()

        # Put the text on the screen.
        output = "Balls: {}".format(len(self.box.particles))
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

        output = "Energy: {:.2f}".format(sum([ball.energy() for ball in self.box.particles]))
        arcade.draw_text(output, 10, 50, arcade.color.WHITE, 14)

        
        # impuls = sum([ball.impuls for ball in self.box.particles]) + self.box.impuls
        # output = "Total impuls: {}".format(impuls)
        # arcade.draw_text(output, 10, 80, arcade.color.WHITE, 14)
        # if len(self.box.particles) > 0:
        #     balls = len(self.box.particles)
        # else:
        #     balls = 1
        # arcade.draw_line(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, impuls[0]/balls + SCREEN_WIDTH/2, impuls[1]/balls + SCREEN_HEIGHT/2, arcade.color.WHITE_SMOKE, 2)

        # output = "Box Impuls: {}".format(self.box.impuls)
        # arcade.draw_text(output, 10, 110, arcade.color.WHITE, 14)

        # bimpuls = sum([ball.impuls for ball in self.box.particles]) + self.box.nullvector
        # output = "Ball impuls: {}".format(bimpuls)
        # arcade.draw_text(output, 10, 140, arcade.color.WHITE, 14)

        #play sound
        if self.bounced:
            arcade.play_sound(self.sound, 0.1)

    def on_update(self, delta_time):
        """ Movement and game logic """
        if not(self.pause):
            self.bounced = self.box.go()
            #self.impulses.vectors = [ball.impuls for ball in self.box.particles]
            #self.impulses.vectors.append(self.box.impuls)


    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called whenever the mouse button is clicked.
        """
        if button == arcade.MOUSE_BUTTON_LEFT:
            mass = random.randrange(5, 50)
            self.box.add_particle(mass=mass, radius=mass, position=[x,y])
        elif button == arcade.MOUSE_BUTTON_RIGHT:
            for i, ball in enumerate(self.box.particles):
                if ball.check_inside([x,y]):
                    self.box.particles.pop(i)
                    return

            if len(self.box.particles) > 0:
                self.box.particles.pop(random.randrange(0, len(self.box.particles)))
            
        
        # self.impulses.vectors = [ball.impuls for ball in self.box.particles]
        # self.impulses.vectors.append(self.box.impuls)
    
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.P:
            self.pause = not(self.pause)
        return super().on_key_press(symbol, modifiers)

    def on_resize(self, width: float, height: float):
        self.box.resize([width,height])
        return super().on_resize(width, height)


def main():
    MyGame()
    arcade.run()


if __name__ == "__main__":
    main()