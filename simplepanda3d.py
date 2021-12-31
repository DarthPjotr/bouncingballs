from math import pi, sin, cos
import sys
import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import NodePath, Material, Fog
from panda3d.core import Point3
from panda3d.core import AmbientLight, Texture
from panda3d.core import Vec3, Vec4
from panda3d.core import LineSegs
from panda3d.core import DirectionalLight
from panda3d.core import WindowProperties

from gas import *


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.pauze = False

        # Disable the camera trackball controls.
        self.disableMouse()

        properties = WindowProperties()
        properties.setSize(1800, 900)

        # self.render.setAntiAlias(8, 1)
        self.win.requestProperties(properties)

        self.set_main_lighting()
        # self.set_background()

        self.register_key_and_mouse_events()

        sizes = [800,600,800]
        nballs = 30
        radius = 8
        self.create_box(sizes, nballs, radius)
        self.draw_box()

        self.set_camera()


    def set_main_lighting(self):
        mainLight = DirectionalLight("main light")
        mainLight.setColor(Vec4(0.3, 0.3, 0.3, 1))
        self.mainLightNodePath = self.render.attachNewNode(mainLight)
        self.mainLightNodePath.setHpr(45, -80, 0)
        self.render.setLight(self.mainLightNodePath)

        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        self.ambientLightNodePath = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.ambientLightNodePath)
        self.render.setShaderAuto()

        color = np.array([0.3, 0.7, 0.9])
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


    def set_camera(self):
        cam_pos = self.box.center.copy()
        cam_pos[self.box.Y] = -2000
        self.camera.setPos(*cam_pos)
        self.camera.setHpr(0, 0, 0)

    def register_key_and_mouse_events(self):
        key = 'arrow_left'
        self.accept(key, self.task_move_camera, [key, ""])
        self.accept(key+"-repeat", self.task_move_camera, [key, ""])

        key = 'arrow_right'
        self.accept(key, self.task_move_camera, [key, ""])
        self.accept(key+"-repeat", self.task_move_camera, [key, ""])

        key = 'arrow_up'
        self.accept(key, self.task_move_camera, [key, ""])
        self.accept(key+"-repeat", self.task_move_camera, [key, ""])

        key = 'arrow_down'
        self.accept(key, self.task_move_camera, [key, ""])
        self.accept(key+"-repeat", self.task_move_camera, [key, ""])
    
        key = 'c'
        self.accept(key, self.task_move_camera, [key, ""])
        # self.accept(key+"-repeat", self.move_camera, [key, ""])

        mouse = 'w'
        self.accept(mouse, self.task_move_camera, ["", mouse])
        self.accept(mouse+"-repeat", self.task_move_camera, ["", mouse])

        mouse = 's'
        self.accept(mouse, self.task_move_camera, ["", mouse])
        self.accept(mouse+"-repeat", self.task_move_camera, ["", mouse])

        mouse = 'a'
        self.accept(mouse, self.task_move_camera, ["", mouse])
        self.accept(mouse+"-repeat", self.task_move_camera, ["", mouse])

        mouse = 'd'
        self.accept(mouse, self.task_move_camera, ["", mouse])
        self.accept(mouse+"-repeat", self.task_move_camera, ["", mouse])


        self.accept('escape', sys.exit)
        self.accept('p', self.task_toggle_pauze)
        
        self.accept('k', self.task_kick)
        self.accept('m', self.task_center)
    
    def task_toggle_pauze(self):
        self.pauze = not self.pauze
        return Task.cont

    def task_kick(self):
        self.box.kick_all()

    def task_center(self):
        self.box.center_all()

    def task_move_camera(self, key="", mouse=""):
        v = 8
        pos = np.array(self.camera.getPos())
        dir = self.render.getRelativeVector(self.camera, Vec3.forward())

        dx = np.array([dir[1],-dir[0],0]) * v
        dy = dir * v
        dz = np.array([0,0,1]) * v

        if key == 'arrow_left':
            pos -= dx
        elif key == 'arrow_right':
            pos += dx
        elif key == 'arrow_up':
            pos += dy
        elif key == 'arrow_down':
            pos -= dy
        # elif key == 'c':
        #     self.camera.lookAt(Vec3(*self.box.center[:3]))

        if mouse == "w":
            pos += dz
        elif mouse == "s":
            pos -= dz
        

        # hpr = np.array(self.camera.getHpr())
        # dh = np.array([v,0,0])
        # dp = np.array([0,v,0])
        # if mouse == "w":
        #     hpr -= dp
        # elif mouse == "s":
        #     hpr += dp
        # elif mouse == 'a':
        #     hpr += dh
        # elif mouse == 'd':
        #     hpr -= dh

        self.camera.setPos(*pos)
        # self.camera.setHpr(*hpr)
        self.camera.setR(0)
        self.camera.lookAt(Vec3(*self.box.center[:3]))
        return Task.cont

    def create_box(self, sizes, nballs, radius):
        self.box = Box(sizes, torus=False)
        self.box.merge = False
        arr = ArrangeParticles(self.box)
        # balls = arr.create_pendulum(0.05, np.array([0,0,-1]))
        # balls = arr.create_simplex()
        # balls = arr.create_kube_planes(100, 10)
        # balls = arr.create_n_mer(15, 3, charge=1)
        # balls = arr.test_interaction_simple(10000)
        self.box.set_interaction(500)
        # self.box.set_friction(0.025)
        balls = arr.random_balls(30, 1, 10, 5, charge=1)
        balls = arr.random_balls(30, 1, 10, 5, charge=0)
        ball = self.box.add_particle(1, 20, self.box.center, speed=None, charge=-10, fixed=True, color=[255,255,255])
        balls.append(ball)
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
        
    def draw_box(self):
        # draw box
        for (i, j) in self.box.edges:
            p1 = self.box.vertices[i]
            p2 = self.box.vertices[j]
            lines = LineSegs()
            lines.setColor(0, 1, 0, 1)
            lines.moveTo(*p1)
            lines.drawTo(*p2)
            lines.setThickness(2)
            node = lines.create()
            # node.setAntiAlias(8, 1)
            np = NodePath(node)
            #np.setAntiAlias(8, 1)
            #np.setColor((1, 1, 1, 1))
            np.reparentTo(self.render)
        
        # draw extra planes
        for plane in self.box.planes[2*self.box.dimensions:]:
            for (i,j) in plane.edges:
                p1 = plane.box_intersections[i]
                p2 = plane.box_intersections[j]
                lines = LineSegs()
                # lines.setColor(1, 1, 1, 1)
                lines.moveTo(*p1)
                lines.drawTo(*p2)
                lines.setThickness(2)
                node = lines.create()
                np = NodePath(node)
                # np.setColor((1, 1, 1, 1))
                np.reparentTo(self.render)
        
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
            sphere.reparentTo(self.render)
            sphere.setPos(*ball.position)
            ball.object = sphere
            # sphere.setColor(0, 100, 100, 10)
            self.spheres.append(sphere)
        
        # create springs
        self.springs = []
        for i, spring in enumerate(self.box.springs):
            p1 = spring.p1.object
            p2 = spring.p2.object
            line = LineSegs()
            # lines.setColor(1, 1, 1, 1)
            line.moveTo((0,0,0))
            line.drawTo((0,1,0))
            line.setThickness(2)
            node = line.create()
            np = NodePath(node)
            np.reparentTo(self.render)
            np.setColor(0,1,0,1)
            self.springs.append(np)

        self.taskMgr.add(self.task_box_go, 'move')
        # self.taskMgr.doMethodLater(1, self.task_box_go, 'move')

        return self.box.particles

    def task_box_go(self, task):
        if self.pauze:
            return Task.cont

        self.box.go(steps=1)

        for ball in self.box.merged_particles:
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
        self.box.merged_particles.clear()

        for ball in self.box.delete_particles:
            sphere = ball.object
            sphere.removeNode()
        self.box.delete_particles.clear()
        
        for ball in self.box.particles:
            sphere = ball.object
            pos = ball.position # - self.box.center
            sphere.setPos(*pos)

        for i, spring in enumerate(self.box.springs):
            p1 = spring.p1.object
            p2 = spring.p2.object
            ray = self.springs[i]
            ray.setPos(p1.getPos())
            ray.lookAt(p2)
            d = (p1.getPos(self.render) - p2.getPos(self.render)).length()
            ray.setScale(d)

        return Task.cont


app = MyApp()
app.run()
