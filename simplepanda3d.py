from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import NodePath, Material, Fog
from panda3d.core import Point3
from panda3d.core import AmbientLight
from panda3d.core import Vec4
from panda3d.core import LineSegs
from panda3d.core import DirectionalLight
from panda3d.core import WindowProperties
from gas import *


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Disable the camera trackball controls.
        self.disableMouse()

        properties = WindowProperties()
        properties.setSize(1000, 750)
        self.win.requestProperties(properties)

        mainLight = DirectionalLight("main light")
        mainLight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        self.mainLightNodePath = self.render.attachNewNode(mainLight)
        self.mainLightNodePath.setHpr(45, -45, 0)
        self.render.setLight(self.mainLightNodePath)

        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor(Vec4(0.1, 0.1, 0.1, 1))
        self.ambientLightNodePath = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.ambientLightNodePath)
        self.render.setShaderAuto()

        # color = (0.1, 0.1, 0.1)
        # expfog = Fog("Scene-wide exponential Fog object")
        # expfog.setColor(*color)
        # expfog.setExpDensity(0.0015)
        # self.render.setFog(expfog)
        # self.setBackgroundColor(*color)


        # # Load the environment model.
        # self.box = self.loader.loadModel("models/Cube")
        # # Reparent the model to render.
        # self.box.reparentTo(self.render)
        # # self.scene.setColorScale(100, 100, 0, 10)
        # # Apply scale and position transforms on the model.
        # self.box.setScale(3, 2, 1)
        # self.box.setPos(0, 0, 0)

        # Add the spinCameraTask procedure to the task manager.
        # self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        sizes = [200,400,150]
        self.create_box(sizes)
        # Load and transform the panda actor.
        self.spheres = []
        for ball in self.box.particles:
            scale = ball.radius * 0.25
            sphere = self.loader.loadModel("models/Sphere_HighPoly")
            material = Material()
            material.setShininess(1.0) 
            color = [c/100 for c in ball.color] 
            color.append(1)
            material.setAmbient(Vec4(*color))
            sphere.setScale(scale, scale, scale)
            sphere.setMaterial(material)
            sphere.reparentTo(self.render)
            sphere.setPos(*ball.position)
            ball.object = sphere
            # sphere.setColor(0, 100, 100, 10)
            self.spheres.append(sphere)
        self.taskMgr.add(self.move, 'move')


        cam_pos = self.box.center.copy()
        cam_pos[self.box.Y] = -300
        self.camera.setPos(*cam_pos)
        self.camera.setHpr(0, 0, 0)
    
    def create_box(self, sizes):
        self.box = Box(sizes)
        for i in range(30):
            pos = self.box.random_position()
            speed = self.box.random(2)
            ball = self.box.add_particle(1, 7, pos, speed)
        
        for (i, j) in self.box.edges:
            p1 = self.box.vertices[i]
            p2 = self.box.vertices[j]
            lines = LineSegs()
            # lines.setColor(1, 1, 1, 1)
            lines.moveTo(*p1)
            lines.drawTo(*p2)
            lines.setThickness(4)
            node = lines.create()
            np = NodePath(node)
            # np.setColor((1, 1, 1, 1))
            np.reparentTo(self.render)

        return self.box.particles


    def move(self, task):
        self.box.go()
        for ball in self.box.particles:
            sphere = ball.object
            pos = ball.position # - self.box.center
            sphere.setPos(*pos)

        return Task.cont

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


app = MyApp()
app.run()
