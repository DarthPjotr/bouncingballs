"""
Bounce balls on the screen.
Spawn a new ball for each mouse-click.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.bouncing_balls
"""

import arcade
import random
from gas import *

# --- Set up the constants

# Size of the screen
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Bouncing Balls Example"


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.box = Box([SCREEN_WIDTH, SCREEN_HEIGHT])

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        arcade.start_render()

        for ball in self.box.particles:
            arcade.draw_circle_filled(ball.position[0], ball.position[1], ball.radius, ball.color)
            end = ball.position + 5*ball.speed
            arcade.draw_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 1)
            

        # Put the text on the screen.
        output = "Balls: {}".format(len(self.box.particles))
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

    def on_update(self, delta_time):
        """ Movement and game logic """
        self.box.go()

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called whenever the mouse button is clicked.
        """
        mass = random.randrange(5, 50)
        self.box.add_particle(mass=mass, radius=mass, position=[x,y])

def main():
    MyGame()
    arcade.run()


if __name__ == "__main__":
    main()