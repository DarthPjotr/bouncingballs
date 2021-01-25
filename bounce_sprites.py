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
(DISPLAY_WIDTH, DISPLAY_HEIGHT) = arcade.get_display_size()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
DEPTH = 200
SCREEN_TITLE = "Bouncing Balls Example"

# Physical contants
GRAVITY = 0.0
FRICTION = 0.00
INTERACTION = 60000.0
TORUS = False


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True, visible=False)  
        self.center_window()
        self.background = None
        self.sound = arcade.load_sound(".\\sounds\\c_bang1.wav")
        self.sprite_list = arcade.SpriteList()
        self.set_visible()

        self.box = None
        self.pause = False
        self.bounced = False
        self.quiet = False
        self.left_mouse_down = False
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.center = numpy.array(self.get_size(),dtype=float)/2
              
    def setup(self):
        self.box = Box([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.box.set_gravity(GRAVITY)
        self.box.set_friction(FRICTION)
        self.box.set_interaction(INTERACTION)
        self.box.torus = TORUS
        self.box.field = Field(self.box)
        self.box.field.field = self.box.field.nofield

        self.firstballs()
    
    def firstballs(self):
        center = self.box.box_sizes/2

        dpos = self.box.nullvector.copy()
        dpos[1] = 120
        pos = center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = 6
        #speed[1] = 1
        ball = self.box.add_particle(10, 10, position=list(pos), speed=list(speed), charge=-1)
        ball.object = arcade.SpriteCircle(ball.radius+15, ball.color, True)
        self.sprite_list.append(ball.object)

        dpos[1] = -20
        pos = center + dpos
        speed[0] = -1
        #speed[1] = -1/6
        ball = self.box.add_particle(60, 60, position=list(pos), speed=list(speed), charge=1)
        ball.object = arcade.SpriteCircle(ball.radius+15, ball.color, True)
        self.sprite_list.append(ball.object)
        self.pause = True

    
    def draw_field(self):
        if self.background is not None:
            arcade.draw_lrwh_rectangle_textured(0, 0,
                                            self.box.box_sizes[0], self.box.box_sizes[1],
                                            self.background)
        else:
            arcade.cleanup_texture_cache()
            if self.box.field == None:
                return
            gx = numpy.arange(0, self.box.box_sizes[0], 80)
            gy = numpy.arange(0, self.box.box_sizes[1], 80)
            for x in gx:
                for y in gy:
                    speed = 100*self.box.field.getvalue([x,y])
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

        # self.draw_field()
        if self.left_mouse_down:
            arcade.draw_line(self.mouse_x, self.mouse_y, self.mouse_dx, self.mouse_dy, arcade.color.WHITE, 1)
        
        # draw avg impuls vector
        start = [self.width/2,self.height/2]
        avg_impuls = self.box.avg_impuls()
        end = [self.width/2 + avg_impuls[0],self.height/2 + avg_impuls[1]]
        arcade.draw_line(*start[:2], *end[:2], color=arcade.color.WHITE, line_width=1)

        for ball in self.box.particles:
            #arcade.draw_circle_filled(ball.position[0], ball.position[1], ball.radius, ball.color)
            end = ball.position + 5*ball.speed
            arcade.draw_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 2)

            if ball.charge < 0:
                output = "+"
            elif ball.charge > 0:
                output = "-"
            if ball.charge != 0:
                arcade.draw_text(output, ball.position[0]+5, ball.position[1]-10, arcade.color.WHITE, 20, font_name="Calibri Bold")

            ball.object.center_x = ball.position[0]
            ball.object.center_y = ball.position[1]
            if self.box.dimensions > 2:
                ball.object.scale = ball.position[2]/self.box.box_sizes[2]
            if self.box.dimensions > 3:
                ball.object.alpha = 255*(ball.position[3]/self.box.box_sizes[3]) % 255
        
        self.sprite_list.draw()
        #self.sprite_list.draw_hit_boxes((255,255,255), 2)

        # Put the text on the screen.
        output = "Balls: {}".format(len(self.box.particles))
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

        output = "Energy: {:.2f}".format(sum([ball.energy for ball in self.box.particles]))
        arcade.draw_text(output, 10, 50, arcade.color.WHITE, 14)

        output = "Avg Impuls: {}".format(self.box.avg_impuls())
        arcade.draw_text(output, 10, 80, arcade.color.WHITE, 14)

        # output = "Avg position: {}".format(self.box.avg_position())
        # arcade.draw_text(output, 10, 110, arcade.color.WHITE, 14)

        #play sound
        if self.bounced and not(self.quiet):
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
            self.mouse_x = x
            self.mouse_y = y
            self.mouse_dx = x
            self.mouse_dy = y
            self.left_mouse_down = True
            # mass = random.randrange(5, 50)
            # charge = random.choice([-1,1])
            # #charge = 1
            # ball = self.box.add_particle(mass=mass, radius=mass, position=[x,y], charge=charge)
            # ball.object = arcade.SpriteCircle(ball.radius+15, ball.color, True)
            # self.sprite_list.append(ball.object)
        elif button == arcade.MOUSE_BUTTON_RIGHT:
            for i, ball in enumerate(self.box.particles):
                if ball.object.collides_with_point([x,y]):
                # if ball.check_inside([x,y]):
                    self.box.particles.pop(i)
                    ball.object.kill()
                    return

            if len(self.box.particles) > 0:
                ball = self.box.particles.pop(random.randrange(0, len(self.box.particles)))
                ball.object.kill()
    
    def on_mouse_drag(self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int):
        if buttons == arcade.MOUSE_BUTTON_LEFT:
            self.mouse_dx = x
            self.mouse_dy = y
        return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        if button == arcade.MOUSE_BUTTON_LEFT:
            mass = random.randrange(5, 50)
            charge = random.choice([-1,1])
            #charge = 1
            position = [self.mouse_x,self.mouse_y]
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            if abs(dx) < 1 and abs(dy) < 1:
                speed = None
            else:
                speed = [dx/5,dy/5]
            ball = self.box.add_particle(mass=mass, radius=mass, position=position, speed=speed, charge=charge)
            ball.object = arcade.SpriteCircle(ball.radius+15, ball.color, True)
            self.sprite_list.append(ball.object)
        
        self.left_mouse_down = False
        return super().on_mouse_release(x, y, button, modifiers)
        
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.P:
            self.pause = not(self.pause)
        elif symbol == arcade.key.S:
            self.quiet = not(self.quiet)
        elif symbol == arcade.key.Q:
            self.close()
        return super().on_key_press(symbol, modifiers)

    def on_resize(self, width: float, height: float):
        self.box.resize([width,height])
        # self.background = None
        if self.background is not None:
            arcade.draw_lrwh_rectangle_textured(0, 0,
                                            self.width, self.height,
                                            self.background)
        numpy.array(self.get_size(),dtype=float)/2
        return super().on_resize(width, height)


def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()