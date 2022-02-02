from math import pi, sin, cos
import sys
import numpy as np

import tkinter
from tkinter import filedialog as fd
from tkinter.messagebox import showwarning
import yaml

from palettable.scientific.diverging import Roma_20_r as colormap


from direct.showbase.ShowBase import ShowBase
# from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import OnscreenText
# from direct.gui.DirectGui import OkDialog
from direct.task import Task
# from direct.actor.Actor import Actor
# from direct.interval.IntervalGlobal import Sequence
from panda3d.core import NodePath, Material, Fog, AntialiasAttrib, PandaNode
# from panda3d.core import Point3
from panda3d.core import AmbientLight, Texture
from panda3d.core import Vec3, Vec4
from panda3d.core import LineSegs
from panda3d.core import DirectionalLight
from panda3d.core import WindowProperties
from panda3d.core import TextNode
from panda3d.core import loadPrcFile
from panda3d.core import TransparencyAttrib
from panda3d.core import CardMaker, Spotlight
# from panda3d.core import TextFont

from panda3d.core import Triangulator
from panda3d.core import GeomVertexFormat
from panda3d.core import GeomVertexData
from panda3d.core import Geom
from panda3d.core import GeomVertexWriter
from panda3d.core import GeomTriangles
from panda3d.core import GeomNode

from gas import *

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

def warning(title, message):
    root = tkinter.Tk()
    root.withdraw()
    showwarning(title, message)
    root.destroy()


class Polygon():
    def __init__(self, vertices=[]):
        # self.vertices=list(tuple(v) for v in vertices)
        self.vertices=numpy.array(vertices)
        self.normal = self._get_normal()

    def _get_normal(self):
        """
        Calculates the normal vector from points on the plane

        Args:
            points (list of numpy.array): the points

        Returns:
            numpy.array: unit normal vector
        """   
        points = [v for v in self.vertices]     
        shape = numpy.shape(points)
        ones = numpy.ones(shape)
        i = 0
        while linalg.det(points[:3]) == 0 and i < 100:
            points += ones
            i += 1

        size = len(points)
        normal = linalg.solve(points[:3], numpy.array([1,1,1]))
        unitnormal = normal/math.sqrt(normal@normal)

        if not (numpy.allclose(points[size-3:]@normal, numpy.array([1,1,1]))):
            raise ValueError("not all points in one plane")
        
        return unitnormal

    def create(self):
        xyzero = False
        for i, x in enumerate(self.normal):
            if x == 1:
                xyzero = True
                break

        # vt=tuple(self.vertices)

        t=Triangulator()
        fmt=GeomVertexFormat.getV3cp()
        vdata = GeomVertexData('name', fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        for point in self.vertices:
            (x,y,z) = point
            v = (x,y)
            if not xyzero:
                v = (x,y)
            elif i == 0:
                v = (y,z)
            elif i == 1:
                v = (x,z)
            elif i == 2:
                v = (x,y)
                    
            t.addPolygonVertex(t.addVertex(*v))
            vertex.addData3f(x,y,z)
        t.triangulate()
        prim = GeomTriangles(Geom.UHStatic)

        for n in range(t.getNumTriangles()):
            prim.addVertices(t.getTriangleV0(n),t.getTriangleV1(n),t.getTriangleV2(n))

        prim.closePrimitive()
        geom = Geom(vdata)
        geom.addPrimitive(prim) 
        node = GeomNode('gnode')
        node.addGeom(geom)

        return node

class World(ShowBase):
    def __init__(self):
        super().__init__()

        # defaults
        self.pause = False
        self._draw_planes = True
        self.dynamic_string_coloring = False
        self.trails = []
        self.tick_rate = 1
        self.quiet = True
        self.bounced = False

        self.boxnode = None
        self.sound = self.loader.load_sfx('sounds/c_bang1.wav')
        # self.sound = self.loader.load_sfx('sounds/Knife Hit Short - QuickSounds.com.mp3')
        # self.sound = self.loader.load_sfx('sounds/Sword Hit Wood 1 - QuickSounds.com.mp3')
        # self.sound = self.loader.load_sfx('sounds/Sword Stab Thin Wood - QuickSounds.com.mp3')
        # self.sound = self.loader.load_sfx('sounds/Weapon Axe Hit Wood And Metal - QuickSounds.com.mp3')
        # self.sound = self.loader.load_sfx('sounds/mixkit-wood-hard-hit-2182.wav')

        # setup window
        properties = WindowProperties()
        properties.setSize(1800, 900)
        self.win.requestProperties(properties)
        self.render.setAntialias(8|64)
    
        # setup the box
        self.setup_box()
        self.draw_box()

        # setup scene, camera lighting and text
        
        self.load_scene()
        self.set_camera()
        self.disableMouse()
        self.set_main_lighting()
        self.set_spotligth()
        # self.set_background()
        self.set_background()
        self.font = self.loader.load_font('fonts/CascadiaCode.ttf')
        self.textnode = self.draw_text("The Box:", 0.1, -0.1)

        # properties for camera control
        self.mouse1_down = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_x_old = 0
        self.mouse_y_old = 0
    
        # register events and tasks
        self.register_key_and_mouse_events()
        self.taskMgr.add(self.task_box_go, 'move')
        self.taskMgr.add(self.task_mouse_move, 'mouse')

    def set_main_lighting(self):
        mainLight = DirectionalLight("main light")
        mainLight.setColor(Vec4(0.3, 0.3, 0.3, 1))
        self.mainLightNodePath = self.render.attachNewNode(mainLight)
        mainLight.setShadowCaster(True)
        self.mainLightNodePath.setHpr(45, -80, 0)
        self.render.setLight(self.mainLightNodePath)
        
        # ambientLight = AmbientLight("ambient light")
        # ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        # self.ambientLightNodePath = self.render.attachNewNode(ambientLight)
        # self.render.setLight(self.ambientLightNodePath)
        # self.render.setShaderAuto()

    def set_spotligth(self):
        self.light = self.render.attachNewNode(Spotlight("Spot"))
        self.light.node().setScene(self.render)
        self.light.node().setShadowCaster(True)
        self.light.node().setColor(Vec4(0.1, 0.1, 0.1, 1))
        # self.light.node().showFrustum()
        self.light.node().getLens().setFov(40)
        self.light.node().getLens().setNearFar(10, 10000)
        pos = self.box.center.copy()
        pos[1] = -500
        pos[2] = 3000
        self.light.setPos(*pos[:3])
        # self.light.setP(-60)
        self.light.setHpr(45, -80, 0)
        self.render.setLight(self.light)

        self.alight = self.render.attachNewNode(AmbientLight("Ambient"))
        self.alight.node().setColor(Vec4(0.2, 0.2, 0.2, 1))
        self.render.setLight(self.alight)

        # Important! Enable the shader generator.
        self.render.setShaderAuto()

        # default values
        self.cameraSelection = 0
        self.lightSelection = 0

        # self.incrementCameraPosition(0)
        # self.incrementLightPosition(0)

    def set_background(self):
        # color = np.array([0.1, 0.1, 0.1])
        # expfog = Fog("Scene-wide exponential Fog object")
        # expfog.setMode(1)
        # expfog.setColor(*color)
        # expfog.setExpDensity(0.0015)
        # self.render.setFog(expfog)

        color = np.array([0, 0, 0])
        self.setBackgroundColor(*color/3)
    
    def load_scene(self):
        # Load the scene.
        floorTex = self.loader.loadTexture('maps/grid.jpg')

        cm = CardMaker('')
        # cm.setFrame(-2, 2, -2, 2)
        cm.setFrame(0, self.box.box_sizes[0], 0, self.box.box_sizes[1])
        # cm.setFrame(0, 0, self.box.box_sizes[0], self.box.box_sizes[1])
        floor = self.render.attachNewNode(PandaNode("floor"))
        for y in range(1):
            for x in range(1):
                nn = floor.attachNewNode(cm.generate())
                nn.setP(-90)
                # nn.setPos((x - 6) * 4, (y - 6) * 4, 0)
                nn.setPos((x - 0) * 4, (y - 0) * 4, 0)
        floor.setTexture(floorTex)
        floor.flattenStrong()
    

    def set_background_old(self):
        # Load the environment model.
        worldLight = AmbientLight("world light")
        worldLight.setColor(Vec4(1, 1, 1, 1))
        texture = self.loader.loadTexture("maps/star.bmp")
        # texture.setWrapU(Texture.WM_repeat)
        # texture.setWrapV(Texture.WM_repeat)
        self.world = self.loader.loadModel("models/test")
        # Reparent the model to render.
        self.world.reparentTo(self.render)

        self.worldLightNodePath = self.render.attachNewNode(worldLight)
        self.world.setLight(self.worldLightNodePath)
        # self.scene.setColorScale(100, 100, 0, 10)
        # Apply scale and position transforms on the model.
        self.world.setScale(1000, 1000, 1000)
        self.world.setPos(0, 0, 0)
        self.world.setTwoSided(True)
        self.world.setTexture(texture, 1)
        #self.world.setShaderAuto()

    def set_background_older(self):
        # Load the environment model.
        worldLight = AmbientLight("world light")
        worldLight.setColor(Vec4(1, 1, 1, 1))
        texture = self.loader.loadModel("models/Sphere_HighPoly")
        # texture.setWrapU(Texture.WM_repeat)
        # texture.setWrapV(Texture.WM_repeat)
        self.world = self.loader.loadModel("models/test")
        # Reparent the model to render.
        self.world.reparentTo(self.render)

        self.worldLightNodePath = self.render.attachNewNode(worldLight)
        self.world.setLight(self.worldLightNodePath)
        # self.scene.setColorScale(100, 100, 0, 10)
        # Apply scale and position transforms on the model.
        self.world.setScale(1000, 1000, 1000)
        self.world.setPos(0, 0, 0)
        self.world.setTwoSided(True)
        # self.world.setTexture(texture, 1)
        # self.world.setShaderAuto()

    def draw_text(self, text, x, y):
        # self.font = self.loader.load_font('fonts/CascadiaCode.ttf')
        textnode = OnscreenText(text=text,
                     style=1, 
                     fg=(1, 1, 1, 1), 
                     # bg=(0, 0, 1, 1),
                     # shadow=(1, 0, 0, 1),
                     # frame=(0.5, 0.5, 0.5, 1),
                     pos=(x, y), scale=.05,
                     parent=self.a2dTopLeft, 
                     align=TextNode.ALeft, 
                     mayChange=True, 
                     font=self.font)
        return textnode
    
    def move_line(self, line, start, end):
        line.setPos(start)
        line.lookAt(end)
        d = (start - end).length()
        if d > 0:
            line.setScale(d)
    
    def draw_trail(self, ball):
        i = ball.index()
        trail = self.trails[i]
        start = ball.position
        start = Vec3(*start[:3])
        for i, end in enumerate(ball.positions):
            line = trail[i]
            end = Vec3(*end[:3])
            self.move_line(line, start, end)
            start = end

    def set_camera(self):
        cam_pos = self.box.center.copy()[:3]
        cam_pos[self.box.Y] = -2000
        self.camera.setPos(*cam_pos)
        self.camera.setHpr(0, 0, 0)

    def register_key_and_mouse_events(self):
        keys = ['w', 's', 'a', 'd', 'arrow_left', 'arrow_right', 'arrow_up', 'arrow_down', 'wheel_down', 'wheel_up', 'mouse1', 'mouse1-up']
        for key in keys:
            self.accept(key, self.task_move_camera, [key, "", 10])
            self.accept(key+"-repeat", self.task_move_camera,  [key, "", 10])
            self.accept("shift-"+key+"-repeat", self.task_move_camera,  [key, "", 20])

        self.accept('p', self.task_toggle_pause)
        self.accept('o', self.task_toggle_quiet)
        self.accept('k', self.task_kick)
        self.accept('c', self.task_center)
        self.accept('h', self.task_stop)
        self.accept('control-l', self.task_load)
        self.accept('control-s', self.task_save)
        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)
    
    def task_toggle_quiet(self):  
        self.quiet = not(self.quiet)
        return Task.cont

    def task_load(self): 
        path = loaddialog() 
        if path is None or len(path) == 0:
            return Task.cont

        with open(path) as file:
            self.load(file)
        
        self.set_camera()
        
        return Task.cont

    def task_save(self):  
        path = savedialog()
        if path is None or len(path) == 0:
            return Task.cont

        with open(path, "w") as file:
            self.save(file)
        return Task.cont
    
    def task_toggle_pause(self):
        self.pause = not self.pause
        return Task.cont

    def task_kick(self):
        self.box.kick_all()
        return Task.cont

    def task_center(self):
        self.box.center_all()
        return Task.cont

    def task_stop(self):
        self.box.stop_all()
        return Task.cont

    def task_move_camera(self, key="", mouse="", speed=15):

        pos = np.array(self.camera.getPos())
        dir = self.render.getRelativeVector(self.camera, Vec3.forward())

        dx = np.array([dir[1],-dir[0],0]) * speed
        dy = dir * speed
        dz = np.array([0,0,1]) * speed

        # lef right
        if key == 'a':
            pos -= dx
        elif key == 'd':
            pos += dx
        # forward backward
        elif key == 'arrow_up' or key == 'wheel_up':
            pos += dy*10
        elif key == 'arrow_down' or key == 'wheel_down':
            pos -= dy*10
        # up down
        elif key == "w":
            pos = self.up_down(pos, speed)
            # pos += dz
        elif key == "s":
            pos = self.up_down(pos, -speed)
            # pos -= dz'
        elif key == 'mouse1':
            self.mouse1_down = True
            mw = self.mouseWatcherNode
            self.mouse_x_old = mw.getMouseX()
            self.mouse_y_old = mw.getMouseY()
        elif key == 'mouse1-up':
            self.mouse1_down = False
    
        self.camera.setPos(*pos)
        self.camera.setR(0)
        self.camera.lookAt(Vec3(*self.box.center[:3]))
        return Task.cont
        
    def task_mouse_move(self, task):
        if self.mouse1_down:
            mw = self.mouseWatcherNode
            speed = 5000
            if  mw.hasMouse():
                
                if mw.is_button_down('shift'):
                    speed = 1500
                else:
                    speed = 5000

                pos = np.array(self.camera.getPos())
                dir = self.render.getRelativeVector(self.camera, Vec3.forward())

                self.mouse_x = mw.getMouseX() 
                self.mouse_y = mw.getMouseY()

                mouse_dx = self.mouse_x_old - self.mouse_x
                mouse_dy = self.mouse_y_old - self.mouse_y

                dx = np.array([dir[1],-dir[0],0]) * mouse_dx
                dpos = dx * speed
                pos += dpos 
                pos = self.up_down(pos, mouse_dy*speed)

                self.mouse_x_old = self.mouse_x
                self.mouse_y_old = self.mouse_y
            
                self.camera.setPos(*pos)
                self.camera.setR(0)
                self.camera.lookAt(Vec3(*self.box.center[:3]))
        
        return Task.cont
    
    def up_down(self, pos, speed):
        nz = np.array([0,0,1])
        center = pos - np.array(self.box.center[:3])
        vz = np.cross(center, np.cross(nz, center))
        vzn = vz/math.sqrt(vz@vz)
        pos += vzn * speed
        
        return pos
    
    def out(self):
        config = {}

        config['quiet'] = self.quiet
        config['pause'] = self.pause
        config['tickrate'] = self.tick_rate
 
        out = {'config': config}
        box = self.box.out()
        return {**out, **box}
    
    def load(self, file):
        data = yaml.load(file, Loader=yaml.FullLoader)
        config = data.get("config", {})

        # self.fps = config['fps']
        self.quiet = config.get('quiet', True)
        self.pause = config.get('pause', False)
        # self.text = config['text']
        self.tick_rate = config.get('tickrate', 1)

        box = data["box"]
        sizes = box["sizes"]
        if len(sizes) > 2:
            self.clear_box()
            self.box = load_gas(data)
            self.draw_box()
        else:
            warning("Warning", "2D boxes not supported")

    def save(self, file):
        out = self.out()
        yaml.dump(out, file, canonical=False, Dumper=yaml.Dumper, default_flow_style=False)

    def setup_box(self):
        self.quiet = True
        sizes = [1200, 1000, 1200]
        self.box = Box(sizes)

        self.box.torus = False
        self.box.merge = False
        self.box.trail = 0
        self.box.skip_trail = 1

        interaction = 000.0
        power = 2
        friction = 0.0
        gravity_strength = 0.5
        gravity_direction = self.box.nullvector.copy()
        gravity_direction[self.box.Z] = 0

        charge_colors = False

        arr = ArrangeParticles(self.box)
        balls = []

        self.box.set_interaction(interaction, power)
        self.box.set_friction(friction)
        self.box.set_gravity(gravity_strength, gravity_direction)

        # balls = arr.create_pendulum(0.2, np.array([0,0,-1]))
        # pos = self.box.center.copy()
        # pos[2] -= 100
        # balls += arr.create_simplex(size=200, position=pos, vertices=12, charge=0)
        # ball = self.box.add_particle(1, 80, self.box.center, fixed=False, charge=0)
        # balls.append(ball)

        # balls += arr.test_spring()

        # balls += arr.create_simplex(charge=0, vertices=6) # 12 = isocahedron
        # self.tick_rate = 10
        # balls += arr.shapes(radius=10, length=100)
        # balls += arr.football(radius=10, length=100)
        # balls += arr.cuboctahedral(radius=50, length=100)
        # balls += arr.create_simplex(charge=1, vertices=18)
        # balls += arr.create_simplex(charge=1, vertices=5)
        # balls += arr.create_simplex(charge=-1, vertices=5)
        # balls += arr.create_simplex(charge=-1, vertices=5)

        # balls += arr.create_kube_planes(800, 10)
        # balls += arr.create_n_mer(12, 4, star=False, circle=True, charge=None)
        # balls = arr.test_interaction_simple(10000, power)
        # balls = arr.test_interaction(40000, power, M0=40, V0=6, D=350, ratio=0.1)
        # balls = arr.test_interaction(30000/9, power, M0=40, V0=7/3, D=350, ratio=0.1)
        
        balls += arr.random_balls(nballs=80, mass=1, radius=40, max_speed=5, charge=0)
        # balls += arr.random_balls(15, 1, 40, 5, charge=-1)
 
        # balls = arr.create_kube_planes(500, 20)
        # ball = self.box.add_particle(1, 10, [15,15,15], speed=None)
        # balls.append(ball)

        # balls += arr.random_balls(nballs=15, mass=1, radius=100, max_speed=5, charge=0)
        # ball = self.box.add_particle(1, 20, self.box.center, speed=None, charge=-10, fixed=True, color=[255,255,255])
        # balls.append(ball)
        # balls = arr.random_balls(30, 1, 10, 5, charge=-1)
        # balls = arr.random_balls(12, 1, 30, 1, charge=-1)
        # for i in range(nballs):
        #     pos = self.box.random_position()
        #     speed = self.box.random(2)
        #     charge=0
        #     ball = self.box.add_particle(1, radius, pos, speed, charge=charge)
    
        # plane = Plane(self.box, [1,1,1], self.box.center)
        # self.box.planes.append(plane)

        normal = [1,1,1,1,1]
        plane = Plane(self.box, normal[:self.box.dimensions], self.box.center)
        plane.color = [0,255,0]
        self.box.planes.append(plane)

        if charge_colors:
            arr.set_charge_colors(balls)
        
        self.box.get_radi()

    def clear_box(self):
        for np in self.boxnode.children:
            np.removeNode()
            np.clear()
        
        self.trails = []
    
    def _fix2d(self,vector):
        if len(vector) < 3:
            vector = numpy.append(vector, 0)
        return vector
        
    def draw_box(self):
        # draw box
        self.boxnode = NodePath("the Box")
        self.boxnode.reparentTo(self.render)

        self.draw_edges()
        self.draw_planes()
        self.draw_spheres()
        self.draw_springs()
        self.draw_trails()
    
    def draw_edges(self):
        for (i, j) in self.box.edges:
            p1 = self.box.vertices[i]
            p2 = self.box.vertices[j]
            lines = LineSegs("edge[{},{}]".format(i,j))
            lines.setColor(0.7, 0.7, 0.7, 1)
            lines.moveTo(*p1[:3])
            lines.drawTo(*p2[:3])
            lines.setThickness(1)
            node = lines.create()
            # node.setAntiAlias(8, 1)
            np = NodePath(node)
            #np.setAntiAlias(8, 1)
            #np.setColor((1, 1, 1, 1))
            # np.reparentTo(self.render)
            np.reparentTo(self.boxnode)
           
    def draw_planes(self):
        # draw extra planes
        if self._draw_planes == True:
            for plane in self.box.planes[2*self.box.dimensions:]:
            # for plane in self.box.planes:
                if self.box.dimensions == 3:
                    vertices = plane.box_intersections
                    if not vertices:
                        continue
                    poly = Polygon(vertices)
                    node = poly.create()
                    np = NodePath(node)
                    # np.reparentTo(self.render)
                    np.reparentTo(self.boxnode)

                    np.setTwoSided(True)
                    np.setTransparency(TransparencyAttrib.M_dual, 1)
                    if not plane.color:
                        color = (0.5, 0.5, 1)
                        color = (random.random(), random.random(),random.random())
                    else:
                        color = [c/255 for c in plane.color]
                    transparency = 0.3
                    np.setColor(*color, transparency)
                    # np.setColor(0.5,0.5,1,0.3)

                for (i,j) in plane.edges:
                    p1 = plane.box_intersections[i]
                    p2 = plane.box_intersections[j]
                    lines = LineSegs("edge[{},{}]".format(i,j))
                    # lines.setColor(1, 1, 1, 1)
                    lines.moveTo(*p1[:3])
                    lines.drawTo(*p2[:3])
                    lines.setThickness(2)
                    node = lines.create()
                    np = NodePath(node)
                    # np.setColor((1, 1, 1, 1))
                    # np.reparentTo(self.render)         
                    np.reparentTo(self.boxnode)
        
    def draw_spheres(self):
        # draw spheres
        self.spheres = []
        for ball in self.box.particles:
            scale = ball.radius * 0.30
            color = [c/255 for c in ball.color] 
            color.append(1)

            sphere = self.loader.loadModel("models/Sphere_HighPoly")
            sphere.setScale(scale, scale, scale)

            material = Material()
            material.setShininess(5)
            material.setAmbient(Vec4(*color))
            material.setSpecular(Vec4(1,1,1,1))
            material.setDiffuse(Vec4(*color))
            # material.setEmission(Vec4(*color))
            sphere.setMaterial(material)

            sphere.reparentTo(self.boxnode)

            sphere.setPos(*ball.position[:3])
            if self.box.dimensions > 3:
                sphere.setTransparency(TransparencyAttrib.M_dual, 1)
            # sphere.setAntiAlias(8,1)
            ball.object = sphere
            # sphere.setColor(0, 100, 100, 10)
            self.spheres.append(sphere)
    
    def draw_springs(self):
        # draw springs
        self.springs = []
        for i, spring in enumerate(self.box.springs):
            p1 = spring.p1.object
            p2 = spring.p2.object
            line = LineSegs("spring[{}]".format(i))
            line.setColor(0.4, 0.4, 0.4, 1)
            line.moveTo((0,0,0))
            line.drawTo((0,1,0))
            line.setThickness(2)
            node = line.create(True)
            np = NodePath(node)
            # np.reparentTo(self.render)
            np.reparentTo(self.boxnode)

            # np.setColor(0,1,0,1)
            self.springs.append((np, line))
            # spring.object = (np, line)
        
    def draw_trails(self):
        # draw trails
        # self.trails = []
        for i, ball in enumerate(self.box.particles):
            trail = []
            for j in range(self.box.trail):
                line = LineSegs("trail[{},{}]".format(i, j))
                color = [c/255 for c in ball.color] 
                line.setColor(*color, 1)
                # line.setColor(0.3, 0.3, 0.3, 1)
                line.moveTo((0,0,0))
                line.drawTo((0,1,0))
                line.setThickness(1)
                
                node = line.create()
                np = NodePath(node)
                # np.reparentTo(self.render)  
                np.reparentTo(self.boxnode)
                # np.reparentTo(ball.object)
                # np.setColor(0,0.5,0,1)
                trail.append(np)
            self.trails.append(trail)   

        return self.box.particles

    def task_box_go(self, task):
        if self.pause:
            return Task.cont

        self.bounced = self.box.go(steps=self.tick_rate)

        while self.box.merged_particles:
            ball = self.box.merged_particles.pop()
            sphere = ball.object
            scale = ball.radius * 0.30
            sphere.setScale(scale, scale, scale)
            material = Material()
            material.setShininess(1.0)
            color = [c/255 for c in ball.color] 
            color.append(1)
            material.setAmbient(Vec4(*color))
            material.setSpecular(Vec4(0,1,1,1))
            sphere.setMaterial(material)

        while self.box.delete_particles:
            ball = self.box.delete_particles.pop()
            index = self.box.delete_trails.pop()
            sphere = ball.object
            trail = self.trails.pop(index)
            sphere.removeNode()
            for np in trail:
                np.removeNode()
        
        for ball in self.box.particles:
            sphere = ball.object
            pos = ball.position # - self.box.center
            sphere.setPos(*pos[:3])
            if self.box.dimensions > 3:
                transparency = pos[3]/self.box.box_sizes[3]
                color = sphere.getColor()
                color[3] = transparency
                sphere.setColor(color)
            if self.box.trail > 0:
                self.draw_trail(ball)

        for i, spring in enumerate(self.box.springs):
            p1 = spring.p1.object
            p2 = spring.p2.object
            ray, line = self.springs[i]

            if self.dynamic_string_coloring:
                color_index = 255 + 1/((spring.energy/10000) - 1/255)
                color = colormap.mpl_colormap(color_index)
                for i in range(line.getNumVertices()):
                    line.setVertexColor(i,*color[:3])

            self.move_line(ray, p1.getPos(), p2.getPos())
       
        charge = sum(p.charge for p in self.box.particles)
        output = "Ticks: {}\nDimensions: {}\nBalls: {}\nCharge: {}".format(self.box.ticks, self.box.dimensions, len(self.box.particles), charge)
        self.textnode.text = output

        if self.bounced and not(self.quiet):
            self.sound.play()

        return Task.cont


def main():
    loadPrcFile("config/Config.prc")
    world = World()
    world.run()

if __name__ == '__main__':
    main()
