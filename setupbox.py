
from ctypes.wintypes import HMODULE
import numpy
import time

from gas import *
from pprint import pprint as pp

class Setup():
    def __init__(self, world=None, dimensions=3) -> None:
        if world:
            world.quiet = True
            world.tick_rate = 1

        sizes = numpy.array([1500, 1500, 1200, 1000, 1000, 1000, 1000, 1000])
        self.box = Box(sizes[:dimensions])
        self.box.torus = False
        self.box.merge = False
        self.box.trail = 0 # 100
        self.box.skip_trail = 1
        self.box.optimized_collisions = True
        self.box.optimized_interaction = True
        self.box.simple_hole_bounce = False

        interaction = 0 # 5000.0
        power = 2.0
        friction = 0.0 #0.035
        gravity_strength = 0.5
        gravity_direction = self.box.nullvector.copy()
        if dimensions > 2:
            gravity_direction[self.box.Z] = -0
        else:
            gravity_direction[self.box.Y] = -0

        self.box.set_interaction(interaction, power)
        self.box.set_friction(friction)
        self.box.set_gravity(gravity_strength, gravity_direction)

        self.layout = ArrangeParticles(self.box)

        self.charge_colors = False
        self.hole_in_walls = False
        self.interaction_factor = 1
        self.neigbor_count = 20

        self.balls = []

        # self._setup_function = self._test_holes
        self._setup_function = self._test_many_holes
        # self._setup_function = None
    
    def hole_on_plane(self, plane, point, radius):
        polygon = FlatD3Shape()
        polygon.regular_polygon_vertices(36)
        polygon.rotate(plane.unitnormal)
        polygon.move(point)
        polygon.scale(radius)
        points = polygon.vertices

        on_plane = True
        for point in points:
            on_plane = on_plane and plane.on_plane(point)
            
        return on_plane

    def _test_holes(self):
        balls = self.balls
        radius = 100
        hole_size = 300

        dpos = [0, 300, 0, 0,0,0,0]
        position = self.box.center - dpos[:self.box.dimensions]
        speed = numpy.array([0.5, 1, 0.3, 0,0,0,0])*5
        ball = self.box.add_particle(mass=1, radius=radius, position=position[:self.box.dimensions], speed=speed[:self.box.dimensions], charge=1)
        balls.append(ball)

        dpos = [-300, -300, 0, 0,0,0,0]
        position = self.box.center - dpos[:self.box.dimensions]
        speed = numpy.array([0.3, 1, -0.6, 0,0,0,0])*5
        ball = self.box.add_particle(mass=1, radius=radius, position=position[:self.box.dimensions], speed=speed[:self.box.dimensions], charge=-1)
        balls.append(ball)


        normal = [0.2,1,0.2,0,0,0,0,0]
        dpos = [0, -200,0,0,0,0,0 ]
        plane = Plane(self.box, normal[:self.box.dimensions], self.box.center + dpos[:self.box.dimensions], reflect=False)

        dpoint = [-450,0,0,0,0,0,0]
        point = self.box.center + dpoint[:self.box.dimensions]
        plane.add_hole(point[:self.box.dimensions], hole_size)

        dpoint = [50,0,0,0,0,0,0]
        point = self.box.center + dpoint[:self.box.dimensions]
        plane.add_hole(point[:self.box.dimensions], hole_size)

        self.box.planes.append(plane)

    def _test_many_holes(self):
        arr = ArrangeParticles(self.box)
        balls = arr.random_balls(nballs=100, mass=1, radius=40, max_speed=3, charge=0)
    
        normal = [1,1,1,0,0,0,0,0]
        dpos = [0, -200,0,0,0,0,0 ]
        plane = Plane(self.box, normal[:self.box.dimensions], self.box.center + dpos[:self.box.dimensions], reflect=False)
        self.box.planes.append(plane)

        nholes = 30
        hole_size = 80

        points = []
        dholes = numpy.zeros(self.box.dimensions)
        dholes.fill(hole_size)

        for i in range(nholes):
            N = 100
            repeat = True
            while N > 0 and repeat:
                on_plane = False
                while not on_plane:
                    point = self.box.random_position(0)
                    point = plane.project_point(point)
                    on_plane = self.hole_on_plane(plane, point, hole_size)

                repeat = False
                for p in points:
                    dp = point - p
                    D2 = dp@dp
                    H2 = hole_size*hole_size*4.05
                    # print(H2, D2, dp)
                    if D2 < H2:
                        # print(i, "\t", N, "\tcollision")
                        repeat = True
                        break
                N -= 1
    
            if N > 0:          
                points.append(point)
                plane.add_hole(point, hole_size)
                    
    
    def _setup(self):
        balls = self.box.particles
        if self.charge_colors:
            self.layout.set_charge_colors(balls)
        
        self.box.get_radi(interaction_factor=self.interaction_factor, neighbor_count=self.neigbor_count)

        return balls

    def make(self):
        if self._setup_function:
            self._setup_function()
        self._setup()
        return (self.box, self.balls)

def main():
    print("START")
    setup = Setup()
    (box, balls) = setup.make()

    # print stuff
    print(box)
    for plane in box.planes:
        print(plane)
        print(plane.phull)
        for hole in plane._holes:
            (point, radius) = hole
            polygon = FlatD3Shape()
            polygon.regular_polygon_vertices(36)
            polygon.rotate(plane.unitnormal)
            polygon.move(point)
            polygon.scale(radius)
            points = polygon.vertices
            on_plane = True
            for point in points:
                on_plane = on_plane and plane.on_plane(point)
            print(hole, on_plane)

    # run the box
    start = time.perf_counter()
    for i in range(1000):
        box.go()
    
    ticks = box.ticks

    end = time.perf_counter()
    dtime = end - start
    print("\ntime = {:.2f}, tick/sec = {:.2f}".format(dtime, ticks/dtime))

    print("END")
    
if __name__ == "__main__":
    main()
