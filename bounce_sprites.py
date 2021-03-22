"""
Bounce balls on the screen.
Spawns a new ball for each mouse-click.
"""

import arcade
import random
import numpy

from palettable.scientific.diverging import Roma_20_r as colormap
# from palettable.mycarta import CubeYF_20 as colormap

import tkinter
from tkinter import filedialog as fd

from gas import *

# Set up the constants

ASK_LOAD = False

UPDATE_RATE = 1/60
TEXT_REFRESH = 10
ROTATION = 2*math.pi*15/360
TICK_RATE = 1

# Size of the screen
(DISPLAY_WIDTH, DISPLAY_HEIGHT) = arcade.get_display_size()
SCREEN_WIDTH = 1000 # DISPLAY_WIDTH
SCREEN_HEIGHT = 900 # DISPLAY_HEIGHT - 100
DEPTH = DISPLAY_HEIGHT # 500
D4 = 500
D5 = 400
SCREEN_TITLE = "Bouncing Balls Example"

# Use change in Sprites alpha value for 3rd dimension
DALPHA = Box.Z
# Use change in Sprites size for 4th dimension
DSIZE = Box.D4

# Physical contants
BOX_DIMENSIONS = [SCREEN_WIDTH, SCREEN_HEIGHT, DEPTH, D4, D5]
BALLS = 6
GRAVITY = 0.0
FRICTION = 0.00
INTERACTION = 10000.0
TORUS = False
DIMENSIONS = 2
# HEIGHT = 30
SPEED = 3

# because it looks better with soft SpriteCircle add bit to sprite radius
D_SPRITE_RADIUS = 15

MAXCOLOR = 255
INVMAXCOLOR = 1/MAXCOLOR



def loaddialog():
    root = tkinter.Tk()
    root.withdraw()
    file = fd.askopenfile(parent=root, title="Load", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")])
    root.destroy()
    return file

def savedialog():
    root = tkinter.Tk()
    root.withdraw()
    file = fd.asksaveasfile(mode="w", parent=root, title="Save", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")], defaultextension=".yml")
    root.destroy()
    return file

class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True, visible=False, update_rate=UPDATE_RATE)  
        self.center_window()
        self.background = None
        self.fps = 1/UPDATE_RATE
        self.sound = arcade.load_sound(".\\sounds\\c_bang1.wav")
        self.ball_list = arcade.SpriteList(use_spatial_hash=False)
        self.arrow_list = arcade.ShapeElementList()
        self.text_list = arcade.ShapeElementList()
        self.set_visible()
        self._frames = 0

        self.box = None
        self.pause = False
        self.bounced = False
        self.quiet = False
        self.text = True
        self.left_mouse_down = False
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.center = numpy.array(self.get_size(),dtype=float)/2

        self._MAX = 0
        self._output = {}
              
    def setup(self):
        if ASK_LOAD:
            file = loaddialog()
            self.box = load(file)
            self.add_balls(self.box.particles)
            self.set_size(int(self.box.box_sizes[Box.X]), int(self.box.box_sizes[Box.Y]))
            # self.center_window()
            # (width, height) = self.get_size()
            # if DIMENSIONS == 1:
            #     self.box.resize([width])
            # else:
            #     self.box.resize([width ,height])
        else:
            self.box = Box(BOX_DIMENSIONS[:DIMENSIONS], TORUS)
            direction = self.box.nullvector.copy()
            direction[0] = 1
            self.box.set_gravity(GRAVITY)
            self.box.set_friction(FRICTION)
            self.box.set_interaction(INTERACTION)
            self.box.torus = TORUS

            # self.box.field = Field(self.box)
            # self.box.field.equation = None # self.box.field.nofield

            # wall = Wall(self.box, 0.4, 0)
            # self.box.walls.append(wall)
            # wall = Wall(self.box, 0.6, 1)
            # self.box.walls.append(wall)

            self.place_balls()
        
        # print(self.box)
        
        # for p in self.box.particles:
        #     print(p)
    
        # for s in self.box.springs:
        #     print(s)

    def getcolor(self, value, vmin, vmax):
        i = (value-vmin)/(vmax-vmin)
        rgb = colormap.mpl_colormap(i)
        V = numpy.array(rgb[:3])
        V = V*255
        color = [round(c) for c in V]
        return color

    def place_balls(self):

        arrangement = ArrangeParticles(self.box)

        # balls = arrangement.test_spring(length=300, distance=240, strength=0.0001, interaction=000)
        # self.add_balls(balls)
        # balls = arrangement.test_springs(10000)
        # self.add_balls(balls)

        # balls = arrangement.test_interaction(40000, M0=40, V0=6, D=300, ratio=0.1)
        # balls = arrangement.test_interaction(30000/100, M0=40, V0=6/10, D=300, ratio=0.1)
        # balls = arrangement.test_spring(length=300, distance=240, strength=0.0001, interaction=000)
        # self.add_balls(balls)
        
        # self.box.set_friction(0.005)
        # direction = self.box.nullvector.copy()
        # direction[self.box.Z] = 1
        # self.box.set_gravity(0.0, direction)
        # self.box.set_interaction(2500)
        # balls = arrangement.random_balls(20, 30, 30, charge=1)
        # # balls = arrangement.random_balls(10, charge=1)
        # # balls = arrangement.create_n_mer(5*5, 5, star=True, charge=None)
        # for i, ball in enumerate(balls):
        #     if ball.charge == -1:
        #         ball.color = arcade.color.RED
        #     else:
        #         ball.color = arcade.color.GREEN
        # self.add_balls(balls)

        # balls = arrangement.random_balls(20, 30, 30, charge=-1)
        # # balls = arrangement.random_balls(10, charge=-1)
        # # balls = arrangement.create_n_mer(20, 2,charge=-None)
        # for ball in balls:
        #     ball.color = arcade.color.RED
        # self.add_balls(balls)

        # for i, ball in enumerate(self.box.particles):
        #     print(i, ball.position, ball.speed)

        # balls = arrangement.create_n_mer(4, 4, False, True, 1)
        # self.add_balls(balls)

        # balls = arrangement.create_n_mer(4, 4, False, True, 0)
        # self.add_balls(balls)

        # balls = arrangement.create_n_mer(4, 4, False, True, -1)
        # self.add_balls(balls)

        # balls = arrangement.create_simplex(200, self.box.center, 0, self.box.dimensions)
        # self.add_balls(balls)

        # balls = arrangement.create_pendulum()
        balls = arrangement.test_rod(150)
        # balls = arrangement.test_fixed_charge(10000)

        self.add_balls(balls)
        #self.box.kick_all()

        # balls = arrangement.create_box(200, charge=1)
        # self.add_balls(balls)

        # self.box.torus = False
        # S = 3
        # balls = arrangement.create_simplex(50, self.box.center, 1)
        # for ball in balls:
        #     ball.speed[0] = -S/len(balls)
        
        # self.add_balls(balls)

        # pos = self.box.center.copy()
        # pos[1] += 250
        # speed = self.box.nullvector.copy()
        # speed[0] = S
        # self.add_ball(1, 5, pos, speed, -1)

        # balls = arrangement.create_box(100, self.box.random_position(), 1)
        # self.add_balls(balls)

        self.quiet = True
        self.pause = True
        self.text = False

    def add_balls(self, balls=list):
        for ball in balls:
            if ball.fixed:
                ball.object = arcade.SpriteCircle(int(ball.radius)+5, ball.color, False)
            else:
                ball.object = arcade.SpriteCircle(int(ball.radius)+D_SPRITE_RADIUS, ball.color, True)
            self.ball_list.append(ball.object)

    def add_ball(self, mass, radius, position=None, speed=None, charge=0, fixed=False, color=None):
        ball = self.box.add_particle(mass, radius, position, speed, charge, fixed, color)
        ball.object = arcade.SpriteCircle(int(ball.radius)+D_SPRITE_RADIUS, ball.color, True)
        self.ball_list.append(ball.object) 

        return ball
    
    def draw_field(self):
        if self.background is not None:
            arcade.draw_lrwh_rectangle_textured(0, 0,
                                            self.box.box_sizes[0], self.box.box_sizes[1],
                                            self.background)
        else:
            arcade.cleanup_texture_cache()
            if self.box.field == None:
                return
            gx = numpy.arange(0, self.box.box_sizes[0], 50)
            gy = numpy.arange(0, self.box.box_sizes[1], 50)
            for x in gx:
                for y in gy:
                    position = self.box.center.copy()
                    position[0] = x
                    position[1] = y
                    value = 10*self.box.field.equation(position=position)
                    try:
                        arcade.draw_line(x, y, x+value[0], y+value[1], [255,255,255], 1)
                        arcade.draw_circle_filled(x+value[0], y+value[1], 2, [255,255,255])
                    except:
                        pass

            self.background = arcade.Texture("background", arcade.get_image(0,0))

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        arcade.start_render()
        self._frames += 1

        if self.box.field is not None and self.box.field.equation != self.box.field.nofield:
            self.draw_field()

        if self.left_mouse_down:
            arcade.draw_line(self.mouse_x, self.mouse_y, self.mouse_dx, self.mouse_dy, arcade.color.WHITE, 1)

        # draw walls
        for wall in self.box.walls:
            coords = [[vertix[0],vertix[1]] for vertix in wall.vertices]
            size = numpy.array(max(coords))
            center = wall.center[:2]

            try:
                size[wall.dimension] = 5
            except IndexError:
                pass

            # arcade.draw_rectangle_filled(*center, *size,  (150,150,150))
            wall_ = arcade.create_rectangle_filled(*center, *size,  (150,150,150))
            wall_.draw()
        
        # draw avg impuls vector
        start = self.box.box_sizes/2
        avg_impuls = self.box.avg_momentum()
        end = start + avg_impuls
        arcade.draw_line(*start[:2], *end[:2], color=arcade.color.WHITE, line_width=1)

        # for shape in self.arrow_list:
        #     self.arrow_list.remove(shape)
        for i, ball in enumerate(self.box.particles):
            #arcade.draw_circle_filled(ball.position[0], ball.position[1], ball.radius, ball.color)
            end = ball.position + 10*ball.speed
            arcade.draw_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 2)
            # arrow = arcade.create_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 2)
            # self.arrow_list.append(arrow)

            output = ""
            if ball.charge < 0:
                output = "-"
            elif ball.charge > 0:
                output = "+"
            if ball.charge != 0 and len(output) > 0 and self.box.interaction != 0:
                arcade.draw_text(output, ball.position[0]+5, ball.position[1]-10, arcade.color.WHITE, 20, font_name="Calibri Bold")
            
            # output = str(i)
            # arcade.draw_text(output, ball.position[0]-0, ball.position[1]+0, arcade.color.WHITE, 8, font_name="Calibri Bold")

            ball.object.center_x = ball.position[0]
            ball.object.center_y = ball.position[1]
            if self.box.dimensions > DSIZE:
                ball.object.scale = ball.position[DSIZE]/self.box.box_sizes[DSIZE]
            if self.box.dimensions > DALPHA:
                ball.object.alpha = 255*(ball.position[DALPHA]/self.box.box_sizes[DALPHA]) % 255

        # draw springs
        for i, spring in enumerate(self.box.springs):
            v = MAXCOLOR + 1/((spring.dlength()/10000) - INVMAXCOLOR)
            color = self.getcolor(v, 0, 255)
            if self.box.dimensions > DALPHA:
                pos = (spring.p1.position[DALPHA] + spring.p2.position[DALPHA])/2
                alpha = 255*(pos/self.box.box_sizes[DALPHA]) % 255
                color.append(alpha)
            arcade.draw_line(*spring.p1.position[:2], *spring.p2.position[:2], color=color, line_width=1)
        
        # draw rods
        for i, rod in enumerate(self.box.rods):
            arcade.draw_line(*rod.p1.position[:2], *rod.p2.position[:2], color=arcade.color.GRAY, line_width=2)

        # if len(self.arrow_list) > 0:
        #     self.arrow_list.draw()
        self.ball_list.draw()
        #self.sprite_list.draw_hit_boxes((255,255,255), 2)

    
        if self.text:
            # Put the text on the screen.

            charge = sum(p.charge for p in self.box.particles)
            output = "Ticks: {}, Dimensions: {}, Balls: {}, Charge: {}".format(self.box.ticks, self.box.dimensions, len(self.box.particles), charge)
            arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

            if self._frames < TEXT_REFRESH or self._frames % TEXT_REFRESH == 0:
                KE = self.box.energy["KE"]
                self._output["KE"] = "Kinetic energy: {:.2f}".format(KE)

                PE = self.box.energy["PE"]
                self._output["PE"] = "Potential Energy: {:.2f}".format(PE)

                SE = self.box.energy["SE"]
                self._output["SE"] = "Spring Energy: {:.2f}".format(SE)

                TE = KE + PE + SE
                self._output["TE"] = "Total Energy: {:.2f}".format(TE)
            
            try:
                arcade.draw_text(self._output["KE"], 10, 50, arcade.color.WHITE, 14)
                arcade.draw_text(self._output["PE"], 10, 80, arcade.color.WHITE, 14)
                arcade.draw_text(self._output["SE"], 10, 120, arcade.color.WHITE, 14)
                arcade.draw_text(self._output["TE"], 10, 150, arcade.color.WHITE, 14)
                # arcade.draw_text(self._output["R"], 10, 180, arcade.color.WHITE, 14)
            except KeyError: 
                pass

            # output = "Avg Impuls: {}".format(self.box.avg_momentum())
            # arcade.draw_text(output, 10, 80, arcade.color.WHITE, 14)
            
            # P = self.box.pressure()
            # output = "pressure: {:}".format(P)
            # arcade.draw_text(output, 10, 110, arcade.color.WHITE, 14)

            # PV = self.box.pressure() * self.box.volume()
            # output = "PV: {:.2f}".format(PV)
            # arcade.draw_text(output, 10, 150, arcade.color.WHITE, 14)

            # try:
            #     output = "PV/nE: {}".format(PV/(E*len(self.box.particles)))
            #     arcade.draw_text(output, 10, 180, arcade.color.WHITE, 14)
            # except:
            #     raise

            # output = "Avg position: {}".format(self.box.avg_position())
            # arcade.draw_text(output, 10, 110, arcade.color.WHITE, 14)

        #play sound
        if self.bounced and not(self.quiet):
            arcade.play_sound(self.sound, 0.1)

    def on_update(self, delta_time):
        """ Movement and game logic """
        #arcade.check_for_collision_with_list
        if not(self.pause):
            for i in range(TICK_RATE):
                self.bounced = self.box.go()
        
        # for i, ball in enumerate(self.box.particles):
        #     if numpy.isnan(ball.position.sum()) or numpy.isnan(ball.speed.sum()):
        #         print(i, self.box.ticks, ball.position, ball.speed)

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
        elif button == arcade.MOUSE_BUTTON_RIGHT:
            for i, ball in enumerate(self.box.particles):
                if ball.object.collides_with_point([x,y]):
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
            # ball = self.box.add_particle(mass=mass, radius=mass, position=position, speed=speed, charge=charge)
            # ball.object = arcade.SpriteCircle(ball.radius+15, ball.color, True)
            # self.ball_list.append(ball.object)
            self.add_ball(mass, mass, position, speed, charge, None)
        
        self.left_mouse_down = False
        return super().on_mouse_release(x, y, button, modifiers)
    
    def _do_rotation(self, symbol):
        action = {
                    arcade.key.A: (Box.Y,  ROTATION),
                    arcade.key.D: (Box.Y, -ROTATION),
                    arcade.key.W: (Box.X,  ROTATION),
                    arcade.key.S: (Box.X, -ROTATION),
                    arcade.key.X: (Box.Z,  ROTATION),
                    arcade.key.V: (Box.Z, -ROTATION),
                }
        if symbol in action.keys():
            self.box.rotate_axis(*action[symbol])
        
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.P:
            # pause everything
            self.pause = not(self.pause)
        elif symbol == arcade.key.O:
            # no sound
            self.quiet = not(self.quiet)
        elif symbol == arcade.key.K:
            # kick the balls
            self.box.kick_all()
        elif symbol == arcade.key.Z:
            # stop the balls
            self.box.stop_all()
        elif symbol == arcade.key.C:
            # center the balls
            self.box.center_all()  
        elif symbol == arcade.key.R:
            # reset framerate
            self.fps = 1/UPDATE_RATE
            self.set_update_rate(UPDATE_RATE)
        elif symbol == arcade.key.EQUAL:
            # increase framerate
            if self.fps < 10:
                self.fps +=1
            else:
                self.fps += 5

            if self.fps < 1:
                self.fps = 1
            self.set_update_rate(1/self.fps)
        elif symbol == arcade.key.MINUS:
            # decrease framerate     
            if self.fps < 10:
                self.fps -=1
            else:
                self.fps -= 5

            if self.fps < 1:
                self.fps = 1
            self.set_update_rate(1/self.fps)
        elif symbol == arcade.key.T:
            # display text
            self.text = not(self.text)
        elif symbol == arcade.key.Q:
            # quit
            self.close()
        elif symbol == arcade.key.S and modifiers & arcade.key.MOD_CTRL:
            file = savedialog()
            save(self.box, file)
            file.close()
        else:
            self._do_rotation(symbol)
        return super().on_key_press(symbol, modifiers)
    

    def on_resize(self, width: float, height: float):
        if DIMENSIONS == 1:
            self.box.resize([width])
        else:
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