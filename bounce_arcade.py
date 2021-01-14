"""
Bounce balls on the screen.
Spawn a new ball for each mouse-click.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.bouncing_balls
"""

import arcade
import random
import numpy

from palettable.scientific.diverging import Roma_18_r as cmap

from gas import *

# --- Set up the constants

# Size of the screen
(w,h) = arcade.get_display_size()
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Bouncing Balls Example"


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)
        self.box = Box([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.box.vectorfield = VectorField(self.box)
        self.box.vectorfield.field = self.box.vectorfield.rotate
        self.pause = False
        self.sound = arcade.load_sound(".\\sounds\\c_bang1.wav")
        self.bounced = False
        self.background = None
    
    def draw_field(self):
        if self.background is not None:
            arcade.draw_lrwh_rectangle_textured(0, 0,
                                            self.box.box_sizes[0], self.box.box_sizes[1],
                                            self.background)
        else:
            arcade.cleanup_texture_cache()
            if self.box.vectorfield == None:
                return
            gx = numpy.arange(0, self.box.box_sizes[0], 80)
            gy = numpy.arange(0, self.box.box_sizes[1], 80)
            for x in gx:
                for y in gy:
                    speed = 100*self.box.vectorfield.getvalue([x,y])
                    try:
                        arcade.draw_line(x, y, x+speed[0], y+speed[1], [255,255,255], 1)
                        arcade.draw_circle_filled(x+speed[0], y+speed[1], 2, [255,255,255])
                    except:
                        pass

            self.background = arcade.Texture("background", arcade.get_image(0,0))

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

        #play sound
        if self.bounced:
            arcade.play_sound(self.sound, 0.1)

    def on_update(self, delta_time):
        """ Movement and game logic """
        if not(self.pause):
            self.bounced = self.box.go()

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
        self.background = None
        return super().on_resize(width, height)


def main():
    MyGame()
    arcade.run()


if __name__ == "__main__":
    main()