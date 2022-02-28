
from ctypes.wintypes import HMODULE
import numpy
import time

from gas import *

class Setup():
    def __init__(self, world=None, dimensions=3) -> None:
        if world:
            world.quiet = True
            world.tick_rate = 1

        sizes = numpy.array([1500, 1500, 1200, 1000, 1000, 1000, 1000, 1000])
        self.box = Box(sizes[:dimensions])
        self.box.torus = False
        self.box.merge = False
        self.box.trail = 100
        self.box.skip_trail = 1
        self.box.optimized_collisions = True
        self.box.optimized_interaction = True
        self.box.simple_hole_bounce = True

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

        self.charge_colors = True
        self.hole_in_walls = False
        self.interaction_factor = 1
        self.neigbor_count = 20

        self.balls = []

        self._setup_function = self._test_holes
        # self._setup_function = None

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
    setup._test_holes()
    # setup.setup()
    box = setup.box

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
