from math import pi, sin, cos
import sys
import numpy as np

import tkinter
from tkinter import filedialog as fd
import yaml

from direct.showbase.ShowBase import ShowBase
# from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import OnscreenText
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
    file = fd.askopenfile(parent=root, title="Load", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")])
    root.destroy()
    return file

def savedialog():
    root = tkinter.Tk()
    root.withdraw()
    file = fd.asksaveasfile(mode="w", parent=root, title="Save", initialdir="D:\\temp", filetypes=[("YAML", "*.yml")], defaultextension=".yml")
    root.destroy()
    return file


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

class MyApp(ShowBase):
    def __init__(self):
        # ShowBase.__init__(self)
        super().__init__()
        self.pauze = False

        # Disable the camera trackball controls.
        self.disableMouse()
        # self.useDrive()

        properties = WindowProperties()
        properties.setSize(1800, 900)

        self.render.setAntialias(8|64)
        self.win.requestProperties(properties)

        self.set_main_lighting()
        # self.set_background()

        self.register_key_and_mouse_events()

        self.draw_planes = True
        self.trails = []

        sizes = [1200, 900, 1000]
        nballs = 30
        radius = 8
        self.create_box(sizes, nballs, radius)
        self.boxnode = None
        self.draw_box()

        self.taskMgr.add(self.task_box_go, 'move')

        self.set_camera()
        self.font = self.loader.load_font('fonts/CascadiaCode.ttf')
        self.textnode = self.draw_text("The Box:", 0.1, -0.1)

    def set_main_lighting(self):
        # floorTex = self.loader.loadTexture('maps/envir-ground.jpg')
        # floor = self.render.attachNewNode(PandaNode("floor"))
        # floor.setTexture(floorTex)
        # floor.flattenStrong()

        mainLight = DirectionalLight("main light")
        mainLight.setColor(Vec4(0.3, 0.3, 0.3, 1))
        self.mainLightNodePath = self.render.attachNewNode(mainLight)

        mainLight.setShadowCaster(True)

        self.mainLightNodePath.setHpr(45, -80, 0)
        self.render.setLight(self.mainLightNodePath)
        
        # self.mainLightNodePath.node().setScene(self.render)
        # self.mainLightNodePath.node().setShadowCaster(True)
        # self.mainLightNodePath.node().showFrustum()
        # self.mainLightNodePath.node().getLens().setFov(40)
        # self.mainLightNodePath.node().getLens().setNearFar(10, 100)
        # self.render.setLight(self.mainLightNodePath)
        
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        self.ambientLightNodePath = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.ambientLightNodePath)
        self.render.setShaderAuto()

        #color = np.array([0.3, 0.7, 0.9])
        color = np.array([0, 0, 0])
        # expfog = Fog("Scene-wide exponential Fog object")
        # expfog.setMode(1)
        # expfog.setColor(*color)
        # expfog.setExpDensity(0.0015)
        # self.render.setFog(expfog)
        self.setBackgroundColor(*color/3)

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

    def set_background(self):
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
    
    def draw_trails(self, ball):
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
        keys = ['w', 's', 'a', 'd', 'c', 'arrow_left', 'arrow_right', 'arrow_up', 'arrow_down']
        for key in keys:
                # params = [key]
                self.accept(key, self.task_move_camera, [key, "", 10])
                self.accept(key+"-repeat", self.task_move_camera,  [key, "", 10])
                self.accept("shift-"+key+"-repeat", self.task_move_camera,  [key, "", 20])

        self.accept('p', self.task_toggle_pauze)
        self.accept('k', self.task_kick)
        self.accept('m', self.task_center)
        self.accept('h', self.task_stop)
        self.accept('l', self.task_load)
        self.accept('o', self.task_save)
        self.accept('escape', sys.exit)

    def task_load(self):   
        with loaddialog() as file:
            self.load(file)
        
        if not file.closed:
            print("File not closed")
        return Task.cont

    def task_save(self):
        with savedialog() as file:
            self.save(file)

        if not file.closed:
            print("File not closed")
        return Task.cont
    
    def task_toggle_pauze(self):
        self.pauze = not self.pauze
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

    def task_move_camera(self, key="", mouse="", speed=10):

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
        elif key == 'arrow_up':
            pos += dy
        elif key == 'arrow_down':
            pos -= dy
        # up down
        elif key == "w":
            pos = self.up_down(pos, speed)
            # pos += dz
        elif key == "s":
            pos = self.up_down(pos, -speed)
            # pos -= dz
        
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

        out = {'config': config}
        box = self.box.out()
        return {**out, **box}
    
    def load(self, file):
        data = yaml.load(file, Loader=yaml.FullLoader)
        config = data.get("config", {})
        self.clear()
        self.box = load_gas(data)
        self.draw_box()

    def save(self, file):
        out = self.out()
        yaml.dump(out, file, canonical=False, Dumper=yaml.Dumper, default_flow_style=False)

    def create_box(self, sizes, nballs, radius):
        self.box = Box(sizes, torus=False)
        self.box.merge = False
        self.box.trail = 20
        self.box.skip_trail = 4
        arr = ArrangeParticles(self.box)
        balls = []
        # balls = arr.create_pendulum(0.05, np.array([0,0,-1]))
        # balls = arr.create_simplex()
        # balls += arr.create_kube_planes(800, 10)
        # balls = arr.create_n_mer(15, 2, charge=None)
        # balls = arr.test_interaction_simple(10000)
        # balls = arr.test_interaction(40000, M0=40, V0=6, D=300, ratio=0.1)
        balls = arr.test_interaction(30000/9, M0=40, V0=7/3, D=200, ratio=0.1)
        
        self.box.set_interaction(30000/9, 2)
        # self.box.set_friction(0.02)
        # gravity = self.box.nullvector.copy()
        # gravity[3] = 1
        # self.box.set_gravity(0.5, gravity)
        # balls += arr.random_balls(15, 1, 40, 5, charge=1)
        # balls += arr.random_balls(30, 1, 40, 5, charge=None)

        
        # balls = arr.create_kube_planes(500, 20)
        # ball = self.box.add_particle(1, 10, [15,15,15], speed=None)
        # balls.append(ball)

        # balls += arr.random_balls(30, 1, 10, 5, charge=-1)
        # ball = self.box.add_particle(1, 20, self.box.center, speed=None, charge=-10, fixed=True, color=[255,255,255])
        # balls.append(ball)
        arr.set_charge_colors(balls)
        # balls = arr.random_balls(30, 1, 10, 5, charge=-1)
        # balls = arr.random_balls(12, 1, 30, 1, charge=-1)
        # for i in range(nballs):
        #     pos = self.box.random_position()
        #     speed = self.box.random(2)
        #     charge=0
        #     ball = self.box.add_particle(1, radius, pos, speed, charge=charge)
    
        # plane = Plane(self.box, [1,1,1], self.box.center)
        # self.box.planes.append(plane)

        # plane = Plane(self.box, [1,0,0], self.box.center/2)
        # plane.color = [0,255,0]
        # self.box.planes.append(plane)

    def clear(self):
        for np in self.boxnode.children:
            np.removeNode()
            np.clear()
        
    def draw_box(self):
        # draw box
        self.boxnode = NodePath("the Box")
        self.boxnode.reparentTo(self.render)

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
        
        # draw extra planes
        if self.draw_planes == True:
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
        
        # create spheres
        self.spheres = []
        for ball in self.box.particles:
            scale = ball.radius * 0.30
            sphere = self.loader.loadModel("models/Sphere_HighPoly")
            material = Material()
            material.setShininess(1.0)

            color = [c/255 for c in ball.color] 
            color.append(1)
            # ball.color = [255*c for c in color[:3]]
            material.setAmbient(Vec4(*color))
            material.setSpecular(Vec4(0,1,1,1))
            sphere.setScale(scale, scale, scale)
            sphere.setMaterial(material)
            # sphere.reparentTo(self.render)
            sphere.reparentTo(self.boxnode)

            sphere.setPos(*ball.position[:3])
            if self.box.dimensions > 3:
                sphere.setTransparency(TransparencyAttrib.M_dual, 1)
            # sphere.setAntiAlias(8,1)
            ball.object = sphere
            # sphere.setColor(0, 100, 100, 10)
            self.spheres.append(sphere)
        
        # create springs
        self.springs = []
        for i, spring in enumerate(self.box.springs):
            p1 = spring.p1.object
            p2 = spring.p2.object
            line = LineSegs("spring[{}]".format(i))
            lines.setColor(0, 1, 0, 1)
            line.moveTo((0,0,0))
            line.drawTo((0,1,0))
            line.setThickness(2)
            node = line.create()
            np = NodePath(node)
            # np.reparentTo(self.render)
            np.reparentTo(self.boxnode)

            # np.setColor(0,1,0,1)
            self.springs.append(np)
        
        # create trails
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

        # self.taskMgr.add(self.task_box_go, 'move')
        # self.taskMgr.doMethodLater(1, self.task_box_go, 'move')

        return self.box.particles

    def task_box_go(self, task):
        if self.pauze:
            return Task.cont

        self.box.go(steps=1)

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
                self.draw_trails(ball)

        for i, spring in enumerate(self.box.springs):
            p1 = spring.p1.object
            p2 = spring.p2.object
            ray = self.springs[i]
            self.move_line(ray, p1.getPos(), p2.getPos())
            # ray.setPos(p1.getPos())
            # ray.lookAt(p2)
            # d = (p1.getPos(self.render) - p2.getPos(self.render)).length()
            # ray.setScale(d)
    
        
        charge = sum(p.charge for p in self.box.particles)
        output = "Ticks: {}\nDimensions: {}\nBalls: {}\nCharge: {}".format(self.box.ticks, self.box.dimensions, len(self.box.particles), charge)
        self.textnode.text = output

        return Task.cont

def main():
    loadPrcFile("config/Config.prc")
    app = MyApp()
    app.run()

if __name__ == '__main__':
    main()

