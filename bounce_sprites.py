# pylint: disable=I
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

"""
Bounce balls on the screen.
Spawns a new ball for each mouse-click.
"""

import random
import math

import tkinter
from tkinter import filedialog as fd

import yaml
import numpy
import arcade

from palettable.scientific.diverging import Roma_20_r as colormap
# from palettable.mycarta import CubeYF_20 as colormap

from gas import * # pylint: disable=wildcard-import, unused-wildcard-import
from setupbox import Setup, ArrangeParticles

# Set up the constants
ASK_LOAD = False

UPDATE_RATE = 1/60
TEXT_REFRESH = 10
TICK_RATE = 1
ROTATION = 2*math.pi*15/360

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
DIMENSIONS = 3
# HEIGHT = 30
SPEED = 3

# because it looks better with soft SpriteCircle add bit to sprite radius
FUZZIE = False
if FUZZIE:
    D_SPRITE_RADIUS = 5
else:
    D_SPRITE_RADIUS = 0

MAXCOLOR = 255
INVMAXCOLOR = 1/MAXCOLOR


def loaddialog():
    root = tkinter.Tk()
    root.withdraw()
    # file = fd.askopenfile(parent=root, title="Load", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")])
    path = fd.Open(parent=root, title="Load", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")]).show()
    root.destroy()
    return path

def savedialog():
    root = tkinter.Tk()
    root.withdraw()
    # file = fd.asksaveasfile(mode="w", parent=root, title="Save", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")], defaultextension=".yml")
    path = fd.SaveAs(parent=root, title="Save", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")], defaultextension=".yml").show()
    root.destroy()
    return path

class World(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True, visible=False, update_rate=UPDATE_RATE, antialiasing=True, samples=16)
        self.center_window()
        self.background = None
        self.fps = 1/UPDATE_RATE
        self.sound = arcade.load_sound(".\\sounds\\c_bang1.wav")
        self.ball_list = arcade.SpriteList(use_spatial_hash=False)
        self.plane_list = arcade.ShapeElementList()
        self.hole_list = arcade.ShapeElementList()
        self.arrow_list = arcade.ShapeElementList()
        self.trail_list = arcade.ShapeElementList()

        self.set_visible()
        self._frames = 0

        self.box = None
        self.pause = False
        self.bounced = False
        self.quiet = False
        self.text = True
        self._draw_planes = True
        self.left_mouse_down = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.center = numpy.array(self.get_size(),dtype=float)/2

        self._MAX = 0
        self._output = {}

    def load(self, file):
        self.ball_list = None
        self.ball_list = arcade.SpriteList(use_spatial_hash=False)

        data = yaml.load(file, Loader=yaml.FullLoader)

        # ugly ...
        global TEXT_REFRESH, TICK_RATE, D_SPRITE_RADIUS, FUZZIE
        try:
            config = data['config']
        except KeyError:
            config = {}

        try:
            self.fps = config['fps']
            self.quiet = config['quiet']
            self.pause = config['pause']
            self.text = config['text']
        except KeyError:
            pass

        TEXT_REFRESH = config.get('textrefresh', TEXT_REFRESH)
        TICK_RATE = config.get('tickrate', TICK_RATE)
        D_SPRITE_RADIUS = config.get('spriteradius', D_SPRITE_RADIUS)
        FUZZIE = config.get('fuzzie', FUZZIE)

        self.set_update_rate(1/self.fps)

        self.box = load_gas(data)
        self.add_balls(self.box.particles)
        self.set_size(int(self.box.box_sizes[Box.X]), int(self.box.box_sizes[Box.Y]))

    def out(self):
        config = {}
        config['fps'] = self.fps
        config['quiet'] = self.quiet
        config['pause'] = self.pause
        config['text'] = self.text
        config['textrefresh'] = TEXT_REFRESH
        config['tickrate'] = TICK_RATE
        config['spriteradius'] = D_SPRITE_RADIUS
        config['fuzzie'] = FUZZIE
        out = {"config": config}
        box = self.box.out()
        return {**out, **box}

    def save(self, file):
        out = self.out()
        yaml.dump(out, file, canonical=False, Dumper=yaml.Dumper, default_flow_style=False)

    def setup(self):
        self.set_visible(False)
        self.set_location(50,50)
        if ASK_LOAD:
            file = loaddialog()
            self.load(file)

            # self.center_window()
            # (width, height) = self.get_size()
            # if DIMENSIONS == 1:
            #     self.box.resize([width])
            # else:
            #     self.box.resize([width ,height])
        else:

            # self.box.field = Field(self.box)
            # self.box.field.equation = None # self.box.field.nofield

            self.setup_box()
            #self.set_size(int(self.box.box_sizes[Box.X]), int(self.box.box_sizes[Box.Y]))
        self.center_window()
        self.set_visible(True)

    @staticmethod
    def getcolor(value, vmin, vmax):
        i = (value-vmin)/(vmax-vmin)
        rgb = colormap.mpl_colormap(i)
        V = numpy.array(rgb[:3])
        V = V*255
        color = [round(c) for c in V]
        return color

    def setup_box(self):
        setup = Setup(self, dimensions=2)
        (box, _) = setup.make()
        self.box = box
        self.add_balls(self.box.particles)

    def setup_box_(self):
        self.quiet = True
        sizes = numpy.array([1500, 1500, 1200, 1000, 1000, 1000, 1000, 1000])
        dimensions = 2
        #sizes = sizes/25

        self.box = Box(sizes[:dimensions])
        self.box.torus = False
        self.box.merge = False
        self.box.trail = 100
        self.box.skip_trail = 1
        self.box.optimized_collisions = True
        self.box.optimized_interaction = True

        interaction = 5000.0
        power = 2.0
        friction = 0.0 #0.035
        gravity_strength = 0.5
        gravity_direction = self.box.nullvector.copy()
        if dimensions > 2:
            gravity_direction[self.box.Z] = -0
        else:
            gravity_direction[self.box.Y] = -0

        charge_colors = True
        hole_in_walls = False
        interaction_factor = 1
        neigbor_count = 20
        _dummy = False

        arrangement = ArrangeParticles(self.box)
        balls = []

        self.box.set_interaction(interaction, power)
        self.box.set_friction(friction)
        self.box.set_gravity(gravity_strength, gravity_direction)

        if hole_in_walls:
            for plane in self.box.planes[:self.box.dimensions*2]:
                plane.reflect = True
                plane.add_hole(plane.point, 500)

        normal = [0,1,0.5,0,0,0,0,0]
        plane = Plane(self.box, normal[:self.box.dimensions], self.box.center+numpy.array([0,-400]), reflect=True)
        #plane.add_hole(self.box.center+[-350,0], 300)
        plane.add_hole(self.box.center+[350,0], 300)
        # plane.color = [0,255,0]
        self.box.planes.append(plane)

        ball = self.box.add_particle(mass=1, radius=150, position=self.box.center-[0,300], speed=numpy.array([0.5])*5, charge=1)
        balls.append(ball)
        ball = self.box.add_particle(mass=1, radius=150, position=self.box.center-[-300,-300], speed=numpy.array([0.3,1])*5, charge=-1)
        balls.append(ball)

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
        # balls = arrangement.random_balls(15, 3, 30, charge=None)
        # # balls = arrangement.create_kube(400)
        # normal = [0,1,0,1,1,1]
        # point = self.box.center
        # plane = Plane(self.box, normal=normal[:DIMENSIONS], point=point)
        #self.box.planes.append(plane)

        # self.box.set_gravity(-0.3, normal[:DIMENSIONS])
        # self.box.set_friction(0.001)

        # normal = [0,1,0,1,1,1]
        # point = [500, 700, 540]# self.box.center
        # # point = self.box.nullvector.copy()
        # plane = Plane(self.box, normal=normal[:DIMENSIONS], point=point)
        # self.box.planes.append(plane)

        # self.box.set_interaction(1500)
        # balls = arrangement.random_balls(10, 1, 10, 2, charge=-1)
        # balls += arrangement.random_balls(10, 1, 10, 2, charge=1)
        # balls = arrangement.set_charge_colors(balls)
        # # balls = arrangement.create_n_mer(20, 2,charge=-None)
        # for ball in balls:
        #     ball.color = arcade.color.RED
        # balls = arrangement.create_kube_planes(500, 50)


        # balls = arrangement.create_pendulum()
        # balls = arrangement.test_walls()
        # balls = arrangement.test_rod()

        # self.add_balls(balls)
        # self.add_planes(self.box.planes[2*self.box.dimensions:])

        # for i, ball in enumerate(self.box.particles):
        #     print(i, ball.position, ball.speed)

        # balls = arrangement.create_n_mer(2, 2, False, False, 0)
        # speed = [1,1,1]
        # center = self.box.center
        # center[0] = self.box.box_sizes[0] - 50
        # balls = arrangement.test_spring(length=150, distance=160, center=center[:self.box.dimensions], speed=speed[:self.box.dimensions])
        # self.add_balls(balls)

        # balls = arrangement.create_n_mer(4, 4, False, True, 0)
        # self.add_balls(balls)

        # balls = arrangement.create_n_mer(4, 4, False, True, -1)
        # self.add_balls(balls)

        # balls = arrangement.create_simplex(200, self.box.center, 0, self.box.dimensions+1)
        # self.add_balls(balls)

        # balls = arrangement.create_pendulum()
        # balls = arrangement.test_rod(150)
        # balls = arrangement.test_fixed_charge(10000)
        if charge_colors:
            arrangement.set_charge_colors(balls)
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
        self.box.get_radi(interaction_factor=interaction_factor, neighbor_count=neigbor_count)

        self.quiet = True
        # self.pause = True
        self.text = False

    def add_balls(self, balls=list):
        for ball in balls:
            color = tuple([int(c) for c in ball.color])
            if ball.fixed:
                ball.object = arcade.SpriteCircle(int(ball.radius)+5, color, False)
            else:
                ball.object = arcade.SpriteCircle(int(ball.radius)+D_SPRITE_RADIUS, color, FUZZIE)
                # ball.object = arcade.SpriteCircle(int(ball.radius), ball.color, False)
            self.ball_list.append(ball.object)

    def add_ball(self, mass, radius, position=None, speed=None, charge=0, fixed=False, color=None):
        ball = self.box.add_particle(mass, radius, position, speed, charge, fixed, color)
        ball.object = arcade.SpriteCircle(int(ball.radius)+D_SPRITE_RADIUS, ball.color, FUZZIE)
        self.ball_list.append(ball.object)

        return ball

    def draw_plane_holes(self, plane):
        self.hole_list = None
        self.hole_list = arcade.ShapeElementList()
        if self.box.dimensions < 3:
            for hole in plane.holes:
                (point, radius) = hole
                pnormal = numpy.array([plane.unitnormal[1], -plane.unitnormal[0]])
                start = point - (radius * pnormal)
                end = point + (radius * pnormal)
                line = arcade.create_line(*start, *end, arcade.color.GO_GREEN, 4)
                self.hole_list.append(line)
        else:
            for hole in plane.holes:
                (point, radius) = hole
                shape = FlatD3Shape()
                shape.regular_polygon_vertices(360)
                shape.rotate(plane.unitnormal)
                shape.scale(radius)
                shape.move(point)
                points = [v[:2] for v in shape.vertices]
                polygon = arcade.create_polygon(points, arcade.color.GO_GREEN)
                if plane.reflect:
                    pass

                self.hole_list.append(polygon)


    def draw_planes(self, planes=list):
        self.plane_list = None
        self.plane_list = arcade.ShapeElementList()
        for plane in planes:
            if self._draw_planes:
                for (i,j) in plane.edges:
                    p0 = plane.box_intersections[i]
                    p1 = plane.box_intersections[j]

                    dot = arcade.create_ellipse(*p0[:2], 5, 5, arcade.color.LIGHT_GRAY)
                    self.plane_list.append(dot)
                    dot = arcade.create_ellipse(*p1[:2], 5, 5, arcade.color.LIGHT_GRAY)
                    self.plane_list.append(dot)

                    line = arcade.create_line(*p0[:2], *p1[:2], arcade.color.LIGHT_GRAY, 1)
                    self.plane_list.append(line)

            point = arcade.create_ellipse(*plane.point[:2], 5, 5, arcade.color.LIGHT_GREEN)
            start = plane.point
            end1 = plane.point + 15*plane.unitnormal
            end2 = plane.point - 15*plane.unitnormal
            normal1 = arcade.create_line(*start[:2], *end1[:2], arcade.color.LIGHT_GREEN, 1)
            normal2 = arcade.create_line(*start[:2], *end2[:2], arcade.color.LIGHT_RED_OCHRE, 1)
            self.plane_list.append(point)
            self.plane_list.append(normal1)
            self.plane_list.append(normal2)

            self.draw_plane_holes(plane)

    def draw_field(self):
        if self.background is not None:
            arcade.draw_lrwh_rectangle_textured(0, 0,
                                            self.box.box_sizes[0], self.box.box_sizes[1],
                                            self.background)
        else:
            arcade.cleanup_texture_cache()
            if self.box.field is None:
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
                    except Exception:
                        pass

            self.background = arcade.Texture("background", arcade.get_image(0,0))

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        arcade.start_render()
        self._frames += 1

#         arcade.draw_text("Test", 50, 50)

        if self.box.field is not None and self.box.field.equation != self.box.field.nofield:
            self.draw_field()

        if self.left_mouse_down:
            arcade.draw_line(self.mouse_x, self.mouse_y, self.mouse_dx, self.mouse_dy, arcade.color.WHITE, 1)

        # draw avg impuls vector
        start = self.box.box_sizes/2
        avg_impuls = self.box.avg_momentum()
        end = start + avg_impuls
        arcade.draw_line(*start[:2], *end[:2], color=arcade.color.WHITE, line_width=1)

        self.arrow_list = None
        self.arrow_list = arcade.ShapeElementList()
        # if self.box.trail > 0:
        self.trail_list = None
        self.trail_list = arcade.ShapeElementList()

        while self.box.delete_particles:
            ball = self.box.delete_particles.pop()
            ball.object.kill()

        while self.box.merged_particles:
            ball = self.box.merged_particles.pop()
            ball.object.kill()
            # +D_SPRITE_RADIUS
            ball.object = arcade.SpriteCircle(int(ball.radius)+D_SPRITE_RADIUS, ball.color, True)
            self.ball_list.append(ball.object)

        for _, ball in enumerate(self.box.particles):
            #arcade.draw_circle_filled(ball.position[0], ball.position[1], ball.radius, ball.color)
            end = ball.position + ball.speed
            # arcade.draw_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 2)
            arrow = arcade.create_line(ball.position[0], ball.position[1], end[0], end[1], arcade.color.GRAY_ASPARAGUS, 2)
            self.arrow_list.append(arrow)
            if ball.positions:
                positions = [(p[0], p[1]) for p in ball.positions]
                trail = arcade.create_line_strip(positions, arcade.color.GRAY, 1)
                self.trail_list.append(trail)

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
        for _, spring in enumerate(self.box.springs):
            v = MAXCOLOR + 1/((spring.dlength()/10000) - INVMAXCOLOR)
            color = self.getcolor(v, 0, 255)
            if self.box.dimensions > DALPHA:
                pos = (spring.p1.position[DALPHA] + spring.p2.position[DALPHA])/2
                alpha = 255*(pos/self.box.box_sizes[DALPHA]) % 255
                color.append(alpha)
            arcade.draw_line(*spring.p1.position[:2], *spring.p2.position[:2], color=color, line_width=1)

            # dpos = self.box.displacement(spring.p1.position, spring.p2.position)
            # # start = spring.p1.position
            # # end = spring.p1.position + dpos
            # start = self.box.center
            # end = self.box.center + dpos
            # arcade.draw_line(*start[:2], *end[:2], color=(255,0,0), line_width=2)

            # dpos = self.box._displacement(spring.p1.position, spring.p2.position)
            # # start = spring.p1.position
            # # end = spring.p1.position + dpos
            # start = self.box.center
            # end = self.box.center + dpos
            # arcade.draw_line(*start[:2], *end[:2], color=(0,255,0), line_width=2)

        # draw rods
        for _, rod in enumerate(self.box.rods):
            arcade.draw_line(*rod.p1.position[:2], *rod.p2.position[:2], color=arcade.color.GRAY, line_width=2)

        self.arrow_list.draw()
        self.ball_list.draw()
        self.trail_list.draw()
        if self._draw_planes:
            self.plane_list.draw()
            self.hole_list.draw()
        #self.sprite_list.draw_hit_boxes((255,255,255), 2)

        if self.text:
            # Put the text on the screen.

            charge = sum(p.charge for p in self.box.particles)
            output = f"Ticks: {self.box.ticks}, Dimensions: {self.box.dimensions}, Balls: {len(self.box.particles)}, Charge: {charge}"
            arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

            if self._frames < TEXT_REFRESH or self._frames % TEXT_REFRESH == 0:
                KE = self.box.energy["KE"]
                self._output["KE"] = f"Kinetic energy: {KE:.2f}"

                PE = self.box.energy["PE"]
                self._output["PE"] = f"Potential Energy: {PE:.2f}"

                SE = self.box.energy["SE"]
                self._output["SE"] = f"Spring Energy: {SE:.2f}"

                TE = KE + PE + SE
                self._output["TE"] = f"Total Energy: {TE:.2f}"

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
        if self.bounced and not self.quiet:
            arcade.play_sound(self.sound, 0.1)

    def on_update(self, delta_time):
        """ Movement and game logic """
        if not self.pause:
            for _ in range(TICK_RATE):
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
        if symbol in action:
            self.box.rotate_axis_3d(*action[symbol])
            self.draw_planes(self.box.planes[2*self.box.dimensions:])

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.P:
            # pause everything
            self.pause = not self.pause
        elif symbol == arcade.key.O:
            # no sound
            self.quiet = not self.quiet
        elif symbol == arcade.key.K:
            # kick the balls
            self.box.kick_all()
        elif symbol == arcade.key.Z:
            # stop the balls
            self.box.stop_all()
        elif symbol == arcade.key.C:
            # center the balls
            self.box.center_all(fixed=True)
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
            self.text = not self.text
        elif symbol == arcade.key.Q or symbol == arcade.key.ESCAPE:
            # quit
            self.close()
        elif symbol == arcade.key.L and modifiers & arcade.key.MOD_CTRL:
            path = loaddialog()
            if path is not None and len(path) > 0:
                with open(path, encoding="utf8") as file:
                    self.load(file)
        elif symbol == arcade.key.S and modifiers & arcade.key.MOD_CTRL:
            path = savedialog()
            if path is not None and len(path) > 0:
                with open(path, "w", encoding="utf8") as file:
                    self.save(file)
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
        # numpy.array(self.get_size(),dtype=float)/2

        self.draw_planes(self.box.planes[2*self.box.dimensions:])

        return super().on_resize(width, height)


def main():  # pylint: disable=function-redefined
    world = World()
    world.setup()
    arcade.run()


if __name__ == "__main__":
    main()
