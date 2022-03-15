# pylint: disable=C,I
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable

from types import MethodType
import numpy
import time
import random
import math
import networkx as nx

from gas import *  # pylint: disable=wildcard-import, unused-wildcard-import
from gas import _Rod, _Membrane
from pprint import pprint as pp # pylint: disable=unused-import

from rotations import RotationMatrix

from p120cell import create_120cell

TEST = False

__all__ = ['ArrangeParticles', 'Setup']

class ArrangeParticles:
    """
    Standard particle arrangements
    """
    def __init__(self, box: Box) -> None:
        """
        Creates particle arrangement

        Args:
            box (Box): The box
        """
        self.box = box

    def set_charge_colors(self, balls):
        for ball in balls:
            if ball.charge >= 1:
                ball.color = [0,255,0]
            elif ball.charge <= -1:
                ball.color = [255,0,0]
            else:
                ball.color = [255,255,0]

        return balls

    def test_bounce(self):
        balls = []
        ball = self.box.add_particle(mass=10, radius=min(self.box.box_sizes)/6,color=[180,180,0])
        balls.append(ball)

        R = 15
        for i in range(R):
            c = (i*255/R) % 255
            color = [c,c,c]
            ball = self.box.add_particle(mass=1, radius=30, color=color)
            balls.append(ball)

        return balls

    def test_all(self, nplanes=1, nballs=1, nsprings=1, charge=0, extra_holes=0, reflect=True):
        balls = []

        for _ in range(nplanes):
            normal = self.box.random()
            distance = (min(self.box.center)/4) * (1-(2*random.random()))
            point = self.box.center + distance * normal
            color = [0,128,0]
            plane = Plane(self.box, normal=normal, point=point, color=None, reflect=reflect)

            self.box.planes.append(plane)
            for _ in range(extra_holes):
                # self.box.random_position()
                _range = [min(self.box.box_sizes)//5, min(self.box.box_sizes)//3]
                _range.sort()
                radius = random.randint(*_range)
                point = self.box.center + self.box.random(min(self.box.box_sizes) - 3*radius)
                plane.add_hole(point, radius)

        balls += self.random_balls(nballs, charge=charge)

        for _ in range(nsprings):
            length = 100*random.random() + 50
            strength = 0.05
            damping = 0.001

            if charge is None:
                c1 = 1
                c2 = -1
            else:
                c1 = c2 = 0

            pos1 = self.box.random_position()
            v1 = self.box.random(3)
            p1 = self.box.add_particle(1, 20, pos1, v1, c1)

            pos2 = pos1 + length * self.box.random()
            v2 = self.box.random(3)
            p2 = self.box.add_particle(1, 20, pos2, v2, c2)
            spring = Spring(length=length, strength=strength, damping=damping, p1=p1, p2=p2)
            self.box.springs.append(spring)

            balls.append(p1)
            balls.append(p2)

        return balls

    def cuboctahedral (self, radius=10, length=100, strength=0.05, damping=0.01,center=True):
        G = nx.Graph()
        edges = []
        start = 0
        size = 4
        for i in range(size):
            edge = (start+i, start+(i+1) % size)
            edges.append(edge)
        G.add_edges_from(edges)
        print(edges)

        edges = []
        start = start + size
        size = 8
        for i in range(size):
            edge = (start+i, start+(i+1) % size)
            edges.append(edge)
        G.add_edges_from(edges)

        edges = [(4,0),(4,1),(6,1),(6,2),(8,2),(8,3),(10,3),(10,0)]
        G.add_edges_from(edges)

        edges = [(5,7),(7,9),(9,11),(11,5)]
        G.add_edges_from(edges)

        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)
        return balls

    def football(self, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        G = nx.Graph()
        edges = []
        start = 0
        size = 5
        for i in range(size):
            edge = (start+i, start+(i+1) % size)
            edges.append(edge)

        print(edges)
        G.add_edges_from(edges)

        start = size
        size = 15
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)

        # print(edges)
        G.add_edges_from(edges)

        edges = [(0,5),(1,8),(2,11),(3,14),(4,17)]
        G.add_edges_from(edges)

        start = start + size
        size = 20
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)

        # print(edges)
        G.add_edges_from(edges)

        edges = [(6,20),(7,23),(9,24),(10,27),(12,28),(13,31),(15,32),(16,35),(18,36),(19,39)]
        G.add_edges_from(edges)

        start = start + size
        size = 15
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)

        # print(edges)
        G.add_edges_from(edges)

        edges = [(21,40),(22,42),(25,43),(26,45),(29,46), (30,48),(33,49),(34,51),(37,52),(38,54)]
        G.add_edges_from(edges)

        start = start + size
        size = 5
        edges = []
        for i in range(size):
            edge = (start+i, start + (i+1) % size)
            edges.append(edge)

        # print(edges)
        G.add_edges_from(edges)

        edges = [(41,55),(44,56),(47,57),(50,58),(53,59)]
        G.add_edges_from(edges)

        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)
        return balls

    def shapes(self, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        G = nx.dodecahedral_graph()
        # G = nx.graph_atlas(134)
        # G = nx.graph_atlas(1167)
        # G = nx.truncated_cube_graph()
        # G = nx.truncated_tetrahedron_graph()
        # G = nx.cycle_graph(5)
        # G = nx.circular_ladder_graph(10)
        # G = nx.circulant_graph(10,[1,4,6])
        # G = nx.frucht_graph()
        # G = nx.moebius_kantor_graph()
        # G = nx.random_tree(10, None)
        # G = nx.sudoku_graph(2)
        # G = nx.pappus_graph()
        # G = nx.octahedral_graph()
        G = nx.hypercube_graph(4)

        # dim = (4,4,4)
        # dim = (2,2,6)
        # dim = (3,3,3)
        # dim = (2,2,2,2)
        # dim = (3,3,6)
        # G = nx.grid_graph(dim=dim, periodic=False)
        # G = nx.wheel_graph(40)
        # G = nx.star_graph(21)

        # G = nx.hexagonal_lattice_graph(4,3)
        # G = nx.triangular_lattice_graph(3,4)
        # G = nx.diamond_graph()

        # G = nx.grid_2d_graph(3,4, periodic=False)
        # G = nx.hypercube_graph(2)
        # G = nx.random_geometric_graph(n=8, radius=8, dim=8, p=10)
        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)

        return balls

    def create_grid(self, dim, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        G = nx.grid_graph(dim=dim, periodic=False)
        balls = self.arrange_from_graph(G, radius, length, strength, damping, center)

        return balls

    def arrange_from_graph(self, G, radius=10, length=100, strength=0.05, damping=0.01, center=True):
        balls = []
        for node in G.nodes:
            try:
                lnode = len(node)
            except TypeError:
                lnode = 1

            if lnode > 2:
                position = numpy.array(node[:self.box.dimensions])
                position = (position * length) # + self.box.center
            else:
                position = self.box.center + self.box.random() * length

            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(mass=1, radius=radius, position=position, speed=speed, charge=1, fixed=False)
            balls.append(ball)

        for edge in G.edges:
            if len(edge) == 2:
                node1,node2 = edge
                i = list(G.nodes).index(node1)
                j = list(G.nodes).index(node2)
                p1 = balls[i]
                p2 = balls[j]
                if numpy.all(p1.position == p2.position):
                    p1.position += (self.box.onevector * 10)
                spring = Spring(length=length, strength=strength, damping=damping, p1=p1, p2=p2)
                self.box.springs.append(spring)
            else:
                print(edge)

        if center:
            self.box.center_all()

        return balls


    def add_rotation_speed(self, rotations, center=None, balls=None):
        if balls is None:
            balls = self.box.particles
        if center is None:
            center = sum([ball.position for ball in balls])/len(balls)

        rotor = RotationMatrix(self.box.dimensions)
        R = rotor.combined_rotations(rotations)
        for ball in balls:
            position = ball.position - center
            dpos = (position @ R) - position
            ball.speed += dpos

        return balls

    def random_balls(self, nballs: int, mass=None, radius=None, max_speed=5, charge=0):
        """
        Randomly places balls in the box

        Args:
            nballs (int): Number of balls
            mass (float, optional): Mass of the balls. Defaults to None.
            radius (int, optional): Radius of balls. Defaults to None.
            max_speed (numpy.array, optional): Speed of balls. Defaults to VMAX.
            charge (int, optional): Charge of the balls. Defaults to 0.

        Returns:
            list: list of balls
        """
        rand_m = False
        rand_r = False
        rand_c = False

        if mass is None:
            rand_m = True
        if radius is None:
            rand_r = True
        if charge is None:
            rand_c = True

        balls = []
        for _ in range(nballs):
            if rand_m:
                mass = random.randrange(1, 30) * 1.0
            if rand_r:
                radius = random.randrange(1, 30) * 1.0
            if rand_c:
                # charge = random.randint(-1, 1)
                charge = random.choice([-1, 1])

            speed = self.box.random(max_speed)
            ball = self.box.add_particle(mass, radius, None, speed, charge, None)
            balls.append(ball)

        return balls

    def create_simplex(self, size=200, position=None, radius=10, charge=0, vertices=None):
        """
        Creates simplex

        Args:
            size (int, optional): Size of the edges. Defaults to 200.
            position (numpy.array, optional): position. Defaults to None.
            charge (int, optional): Charge of the balls. Defaults to 0.
            vertices (int, optional): Number of vertices. Defaults to None.

        Returns:
            list: list of balls
        """
        if position is None:
            center = self.box.center
        else:
            center = position

        if vertices is None:
            vertices = self.box.dimensions+1

        balls = []
        for i in range(vertices):
            pos = center + self.box.random(size)
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, radius, pos, speed, charge)
            balls.append(ball)

        balls[0].speed = 5 * self.box.onevector.copy()
        balls[-1].speed = -5 * self.box.onevector.copy()

        for i, b1 in enumerate(balls):
            for b2 in balls[i:]:
                if b1 != b2:
                    spring = Spring(size, 0.05, 0.01, b1, b2)
                    self.box.springs.append(spring)
        return balls

    def create_box(self, size, position=None, radius=10, charge=0):
        """
        Creates a box

        Args:
            size (float): size of the box
            position (numpy.array, optional): position. Defaults to None.
            charge (int, optional): charge of the balls of the box. Defaults to 0.

        Returns:
            list: list of balls
        """
        ratio = max(self.box.box_sizes)/size
        sizes = self.box.box_sizes/ratio
        if position is None:
            center = self.box.box_sizes/2
        else:
            center = position
        box = Box(sizes)
        speed = self.box.nullvector.copy()

        balls = []
        for vertex in box.vertices:
            pos = center - (box.box_sizes/2) + vertex
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, radius, pos, speed, charge)
            balls.append(ball)

        balls[0].speed = 5 * self.box.onevector.copy()
        balls[-1].speed = -5 * self.box.onevector.copy()

        l = sum(box.box_sizes)/box.dimensions
        for edge in box.edges:
            spring = Spring(l, 0.01, 0.01, balls[edge[0]], balls[edge[1]])
            self.box.springs.append(spring)

        return balls

    def create_kube_planes(self, size, nballs):
        sizes = numpy.array(self.box.dimensions * [size*1.0])
        kube = Box(sizes)
        dcenter = self.box.center - kube.center
        # for vertix in kube.vertices:
        #     vertix += dcenter
        # for plane in kube.planes:
        #     plane.point += dcenter
        #     plane.box = self.box
        #
        for plane in kube.planes:
            point = plane.point + dcenter
            normal = plane.unitnormal
            plane_ = Plane(self.box, normal, point)
            self.box.planes.append(plane_)

        balls = []

        b = {}
        for (p1, p2) in kube.edges:
            if p1 not in b:
                pos1 = kube.vertices[p1] + dcenter
                speed = self.box.nullvector.copy()
                ball1 = self.box.add_particle(1,3, pos1, speed, 0, True, [255,255,255])
                b[p1] = ball1
                balls.append(ball1)

            if p2 not in b:
                pos2 = kube.vertices[p2] + dcenter
                speed = self.box.nullvector.copy()
                ball2 = self.box.add_particle(1,3, pos2, speed, 0, True, [255,255,255])
                b[p2] = ball2
                balls.append(ball2)

            l = math.sqrt((b[p1].position - b[p2].position) @ (b[p1].position - b[p2].position))
            spring = Spring(l, 0, 0, b[p1], b[p2])
            self.box.springs.append(spring)


        for _ in range(nballs):
            position = kube.random_position() + dcenter
            ball = self.box.add_particle(10, 50, position)
            balls.append(ball)

        return balls

    def create_n_mer(self, nballs, n=2, star=False, circle=False, charge=0):
        """
        Creates shape of n balls in a line, star of circle shape

        Args:
            nballs (int): number of balls
            n (int, optional): number of balls in the shape. Defaults to 2.
            star (bool, optional): Make a star shape. Defaults to False.
            circle (bool, optional): Make a circle. Defaults to False.
            charge (int, optional): Charge of the balls. Defaults to 0.

        Returns:
            list: list of balls
        """
        radius = 20
        lspring = 150
        balls = []
        alternate = False
        if charge is None:
            alternate = True
            charge = 1
        for _ in range(round(nballs/n)):
            pos1 = self.box.random_position()
            speed = self.box.random(3)
            if star and alternate:
                charge = n - 1
            b1 = self.box.add_particle(1, radius, pos1, speed, charge)
            bstart = b1
            balls.append(b1)
            if n > 1:
                for _ in range(n-1):
                    if alternate:
                        if star:
                            charge = -1
                        else:
                            charge = -charge
                    pos2 = pos1 + self.box.random() * (lspring)
                    speed2 = speed + self.box.random(0.5)
                    b2 = self.box.add_particle(1, radius, pos2, speed2, charge)
                    balls.append(b2)
                    spring = Spring(lspring, 0.15, 0.00, b1, b2)
                    self.box.springs.append(spring)
                    if not star:
                        b1 = b2
                if circle:
                    spring = Spring(lspring, 0.01, 0, b1, bstart)
                    self.box.springs.append(spring)
        return balls

    def create_kube(self, size, position=None, charge=0):
        """
        Creates a kube

        Args:
            size (float): size of the box
            position (numpy.array, optional): position. Defaults to None.
            charge (int, optional): charge of the balls of the box. Defaults to 0.

        Returns:
            list: list of balls
        """
        if position is None:
            center = self.box.box_sizes/2
        else:
            center = position
        sizes = [size]*self.box.dimensions
        box = Box(sizes)
        speed = self.box.nullvector.copy()

        balls = []
        for vertex in box.vertices:
            pos = center - (box.box_sizes/2) + vertex
            speed = self.box.nullvector.copy()
            ball = self.box.add_particle(1, 1, pos, speed, charge)
            balls.append(ball)

        # balls[0].speed = 5 * self.box.onevector.copy()
        # balls[-1].speed = -5 * self.box.onevector.copy()

        for i, b0 in enumerate(balls):
            for b1 in balls[i:]:
                if b1 == b0:
                    continue
                d = math.sqrt((b0.position-b1.position) @ (b0.position-b1.position))
                spring = Spring(d, 0.1, 0.02, b0, b1)
                self.box.springs.append(spring)

        return balls

    def create_pendulum(self, gravity=0.1, direction=None):
        self.box.set_gravity(gravity, direction)
        balls = []

        pos = self.box.center.copy()
        # pos[Box.Y] = self.box.box_sizes[Box.Y]
        speed = self.box.nullvector.copy()
        ancor = self.box.add_particle(1, 1, pos, speed, charge=0, fixed=True, color=[255,255,255])
        balls.append(ancor)

        lspring = 150
        pos = pos.copy()
        pos[Box.X] += lspring
        ball1 = self.box.add_particle(10,20, pos, speed, color=[0,255,255])
        balls.append(ball1)

        spring = Spring(lspring, 10, 0.01, ancor, ball1)
        self.box.springs.append(spring)

        lspring = 100
        pos = pos.copy()
        pos += self.box.random(lspring)
        # pos[Box.Z] = self.box.center[Box.Z]
        ball2 = self.box.add_particle(10,20, pos, speed, color=[0,255,255])
        balls.append(ball2)

        spring = Spring(lspring, 10, 0.01, ball1, ball2)
        self.box.springs.append(spring)

        return balls

    def test_interaction_simple(self, interaction, power=2):
        self.box.set_interaction(interaction, power)
        balls = []

        dpos = self.box.nullvector.copy()
        dpos[1] = 120
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = 3
        ball = self.box.add_particle(5, 5, position=list(pos), speed=list(speed), charge=-1)
        balls.append(ball)

        dpos = self.box.nullvector.copy()
        dpos[1] = -20
        pos = self.box.center + dpos
        speed[0] = -3/6
        ball = self.box.add_particle(30, 30, position=list(pos), speed=list(speed), charge=1)
        balls.append(ball)

        return balls

    def test_interaction(self, interaction=20000, power=2, M0=40, V0=6, D=140, ratio=0.1):
        self.box.set_interaction(interaction, power)
        balls = []

        dpos = self.box.nullvector.copy()
        dpos[1] = D * ratio
        M = M0 * (1-ratio)
        V = V0 * (ratio)
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = V
        ball = self.box.add_particle(M, int(M), position=list(pos), speed=list(speed), charge=-1, color=[255,0,0])
        balls.append(ball)

        dpos = self.box.nullvector.copy()
        dpos[1] = -D * (1-ratio)
        M = M0 * ratio
        V = -V0 * (1-ratio)
        pos = self.box.center + dpos
        speed[0] = V
        ball = self.box.add_particle(M, int(M), position=list(pos), speed=list(speed), charge=1, color=[0,255,0])
        balls.append(ball)

        return balls

    def test_springs(self, interaction=0):
        self.box.set_interaction(interaction)

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[1] = 120
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        speed[0] = 0
        #speed[1] = 1
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=-1)
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = -20
        pos = self.box.center + dpos
        speed[0] = -0
        #speed[1] = -1/6
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=1)
        balls.append(ball)
        b2 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = 50
        dpos[1] = 50
        pos = self.box.center + dpos
        speed[0] = -0
        #speed[1] = -1/6
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=1)
        balls.append(ball)
        b3 = ball

        spring = Spring(150, 0.03, 0.001, b1, b2)
        self.box.springs.append(spring)
        spring = Spring(150, 0.03, 0.001, b2, b3)
        self.box.springs.append(spring)
        spring = Spring(190, 0.02, 0.001, b1, b3)
        self.box.springs.append(spring)

        return balls

    def test_spring(self, length=150, distance=240, strength=0.03, interaction=0 , center=None,speed=None):
        self.box.set_interaction(interaction)
        if center is None:
            center = self.box.center

        if speed is None:
            speed = self.box.nullvector.copy()

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[0] = distance/2
        pos = center + dpos
        # speed = self.box.nullvector.copy()
        # speed[0] = 0

        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=-1, color=[255,0,0])
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[0] = -distance/2
        pos = center + dpos
        # speed[0] = -0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=1, color=[0,255,0])
        balls.append(ball)
        b2 = ball

        spring = Spring(length, strength, 0.000, b1, b2)
        self.box.springs.append(spring)

        return balls

    def test_fixed_charge(self, interaction=10000):
        self.box.set_interaction(interaction)

        balls = []

        ball = self.box.add_particle(1, 25, self.box.center, speed=self.box.nullvector.copy(), charge=5, fixed=True, color=[255,0,255])
        balls.append(ball)

        balls_ = self.create_n_mer(2, 2)
        for i, ball in enumerate(balls_):
            if i % 2 == 0:
                ball.charge = -1
            else:
                ball.charge = 1
        balls.extend(balls_)

        return balls

    def test_walls(self):
        normal = self.box.nullvector.copy()
        normal[0] = 1
        normal[1] = 0.3
        # wall = Plane(self.box, normal, self.box.center)
        wall = _Membrane(self.box, normal, self.box.center)
        # wall.hole_size = 15
        wall.max_speed = 4
        wall.filter = wall.maxwells_demon_filter

        self.box.planes.append(wall)

        balls = []
        position = self.box.center.copy()
        position += self.box.center/2
        # position[0] += self.box.center[0]/2
        speed = self.box.onevector.copy()
        ball = self.box.add_particle(1, 50, position, speed=speed, charge=0, fixed=False, color=[0,0,255])
        balls.append(ball)

        position = self.box.center.copy()
        position -= self.box.center/2
        speed = -3 * self.box.onevector.copy()
        ball = self.box.add_particle(1, 10, position, speed=speed, charge=0, fixed=False, color=[255,255,0])
        balls.append(ball)

        balls += self.random_balls(10, 1, 10, 2)

        return balls

    def test_rod(self, length=150, interaction=0):
        self.box.set_interaction(interaction)

        balls = []

        dpos = self.box.nullvector.copy()
        dpos[Box.X] = length/2
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        # speed = self.box.random(1.0)
        #speed[Box.X] = 1.5
        #speed[Box.Y] = 2.0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=0, color=[255,0,0])
        balls.append(ball)
        b1 = ball

        dpos = self.box.nullvector.copy()
        dpos[Box.X] = -length/2
        # dpos[Box.Y] = 50
        pos = self.box.center + dpos
        speed = self.box.nullvector.copy()
        # speed[Box.Y] = 1.0 # self.box.random(0.5)

        speed[Box.X] = 1.5
        speed[Box.Y] = -2.0
        ball = self.box.add_particle(1, 30, position=list(pos), speed=list(speed), charge=0, color=[0,255,0])
        balls.append(ball)
        b2 = ball

        rod = _Rod(length, b1, b2)
        # spring = Spring(length, 1.0, 0, b1, b2)
        # self.box.rods.append(rod)
        self.box.rods.append(rod)

        return balls

class _Test():
    def test_plane(self):
        planes = []
        box = Box([10,20,30])
        print(box)
        plane = Plane(box, [math.pi,2,5], [5,math.e,30])
        print(plane)
        planes.append(plane)


        print("\n####\n")

        # plane2 = Plane(box, points=[[5,10,15],[6,8,2],[7,12,4]])
        plane2 = Plane(box, [2,4,3], [5,10,30])
        # plane2 = Plane(box, [3,2,5], [5,10,30])
        print(plane2)
        planes.append(plane2)

        print("\n####\n")

        plane3 = Plane(box, [5,3,6], [5,15,25])
        print(plane3)
        planes.append(plane3)

        print("\n####\n")
        print(plane.intersection(planes))
        intersection = plane.intersection(planes)

        print("\n####\n")

        # print(plane.on([5,10,30]))
        # print(plane.on2([5,10,30]))

        # print(plane.on([5,13,33]))
        # print(plane.on2([5,13,33]))

        print("\n####\n")
        print(plane)
        # for p in planes:
        #     print(p.on(intersection))
        #     print(p.on2(intersection))

        projected = plane.project_point([3,5,math.sqrt(2)])
        print(projected)

        projected2 = plane.project_point(projected)
        print(projected2)

        equal = numpy.allclose(projected, projected2)
        distance = plane.distance(projected)
        onplane = numpy.isclose(distance, 0)
        print(equal, onplane)
        print(distance==0)
        print(projected==projected2)

        return 0


    def test_box(self):
        box = Box([800,600,700])
        print(box.vertices)
        print(box.edges)
        print(box.volume())
        print(box.area())
        print(box.axis)

    def test(self):
        BOX_DIMENSIONS = [800, 600, 700, 400, 300]
        NBALLS = 20
        GRAVITY = 0.0
        FRICTION = 0.00
        INTERACTION = 0000.0
        TORUS = False
        DIMENSIONS = 3
        SPEED = 4
        box = Box(BOX_DIMENSIONS[:DIMENSIONS])
        # for i in range(NBALLS):
        #     if SPEED is not None:
        #         speed = numpy.array([2*random.random()-1 for r in range(DIMENSIONS)])
        #         speed = SPEED * speed / math.sqrt(speed.dot(speed))
        #         speed = list(speed)
        #     else:
        #         speed = SPEED
        #     box.add_particle(1, 1, None, speed)

        arrangement = ArrangeParticles(box)
        # arrangement.random_balls(25, None, None, VMAX, None)

        # balls = arrangement.test_springs(10000)
        # self.add_balls(balls)

        # balls = arrangement.test_interaction(40000, M0=40, V0=6, D=300, ratio=0.1)
        balls = arrangement.test_spring(length=300, distance=240, strength=0.0001, interaction=000)
        for i in range(1000):
            try:
                box.go()
                # KE = sum(ball.energy for ball in box.particles)
                # EE = KE * (2 / DIMENSIONS) # https://en.wikipedia.org/wiki/Kinetic_theory_of_gases#Pressure_and_kinetic_energy
                # PV = box.pressure()*box.volume()
                # print("{:.2f} {:.2f} {:.2f}".format(PV, EE, PV/EE))
                # PE = sum(ball.potential_energy for ball in box.particles)
                # SE = sum(spring.energy for spring in box.springs)
                KE = box.energy["KE"]
                PE = box.energy["PE"]
                EE = box.energy["EE"]
                SE = box.energy["SE"]

                print("{};{};{};{};{};{}".format(i, KE, PE, SE, EE, KE+PE+SE).replace(".", ","))

                # print("{:,2f} {:,2f} {:,2f}".format(PV, E, PV/E))

            except KeyboardInterrupt:
                print("user interupted")
                break

        pp(box.out())

    def test_displacement(self):
        box = Box([10, 20])
        p1 = numpy.array([1,2])
        p2 = numpy.array([2,19])
        p3 = numpy.array([9,19])
        box.torus = True

        d = box.displacement(p1, p2)
        print(d, p1 - p2)

        d = box.displacement(p1, p3)
        print(d, p1 - p3)

    def test_save_yaml(self):
        FILE = "D:\\temp\\box.yml"
        BOX_DIMENSIONS = [800, 600, 700, 400, 300]
        NBALLS = 20
        GRAVITY = 0.0
        FRICTION = 0.00
        INTERACTION = 0000.0
        TORUS = False
        DIMENSIONS = 3
        box = Box(BOX_DIMENSIONS[:DIMENSIONS])
        arrangement = ArrangeParticles(box)
        balls = arrangement.test_spring(length=300, distance=240, strength=0.0001, interaction=000)

        with open(FILE) as file:
            save(box, file)

    def test_load_yaml(self):
        FILE = "D:\\temp\\box.yml"
        with open(FILE) as file:
            box = load(file)
        print(box)

        for p in box.particles:
            print(p)

        for s in box.springs:
            print(s)

        for _ in range(100):
            box.go()
            print(box.ticks)

    def plane(self):
        # sizes = [1000, 900, 1080]
        sizes = [100, 200, 300]
        box = Box(sizes)
        print(box.vertices)
        print(box.edges)
        print(box.axis)
        for p in box.planes:
            print(p)

        # plane = Plane(box, points=box.axis)
        point = box.center
        normal = numpy.array([1,2,0,1])

        normal = [1,1,1,1]

        normal = normal[:box.dimensions]
        point = box.center
        plane = Plane(box, normal=normal, point=point)
        # print(plane)
        box.planes.append(plane)
        plane = Plane(box, normal=normal, point=point)
        # print(plane)
        # box.planes.append(plane)

        print("\n####\n")
        for plane in box.planes[2*box.dimensions:]:
            points = plane.box_intersections
            print(plane)
            print("\n")
            points = numpy.array(points)
            # points.sort(axis=2)
            for point in points:
                print(point)
            print("\n")

        for plane in box.planes[2*box.dimensions:]:
            points = plane.box_intersections
            print(plane)
            for (i,j) in plane.edges:
                print(i, j, points[i], points[j])

            # print("\n")
            # results = []
            # points = []
            # for (i, j) in box.edges:
            #     v1 = box.vertices[i]
            #     v2 = box.vertices[j]
            #     V = v1 - v2
            #     intersection = plane.intersect_line(v1, V)
            #     if intersection is not None and numpy.all(intersection <= box.box_sizes) and numpy.all(intersection >= box.nullvector):
            #         skip = False
            #         for ins in points:
            #             if numpy.allclose(intersection, ins):
            #                 skip = True
            #                 break
            #         if not skip:
            #             points.append(intersection)
            #             results.append(((i,j),intersection))

            # for p in results:
            #     print(p)

    def shapes(self):
        sizes = [100, 200, 300]
        box = Box(sizes)
        arr = ArrangeParticles(box)
        balls = arr.shapes()

        for _ in range(100):
            box.go()

    def kdtree(self):
        sizes = [100, 200, 300, 500]
        box = Box(sizes)
        box.interaction = 000
        box.optimized_collisions = True
        box.optimized_interaction = True

        arr = ArrangeParticles(box)

        nballs = 100
        charge = 0
        balls = arr.random_balls(nballs=nballs, mass=1, radius=5, charge=charge)
        balls = arr.random_balls(nballs=nballs, mass=1, radius=5, charge=-charge)

        box.get_radi(interaction_factor=1, neighbor_count=0)

        box.interaction_radius = 60
        box.interaction_neighbors = 20
        # print([ball.position for ball in box.particles])
        print("nball: {}\noptimized collisions: {}\noptimized interaction: {}\ninteraction radius: {:.2f}\nneighbor count: {}".format(len(box.particles), box.optimized_collisions, box.optimized_interaction, box.interaction_radius, box.interaction_neighbors))

        for i in range(100):
            box.go()

        ticks = box.ticks
        return ticks


class Setup():
    def __init__(self, world=None, dimensions=3) -> None:
        if world:
            world.quiet = True
            world.tick_rate = 1
            world.project4d = False

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
        gravity_strength = 0
        gravity_direction = self.box.nullvector.copy()
        if dimensions > 3:
            gravity_direction[self.box.D4] = -1
        elif dimensions > 2:
            gravity_direction[self.box.Z] = -1
        else:
            gravity_direction[self.box.Y] = -1

        self.box.set_interaction(interaction, power)
        self.box.set_friction(friction)
        self.box.set_gravity(gravity_strength, gravity_direction)

        self.layout = ArrangeParticles(self.box)

        self.charge_colors = False
        self.hole_in_walls = False
        self.interaction_factor = 1
        self.neighbor_count = 20

        self.balls = []

        self.arrangement = ArrangeParticles(self.box)

        self._setup_function = None
        # self._setup_function = self.p120_cell
        self._setup_function = self._test_rotation
        # self._setup_function = self.arrangement.create_pendulum
        # self._setup_function = self.many_interactions
        # self._setup_function = self._eight_dim
        # self._setup_function = self._test_holes
        # self._setup_function = self._test_many_holes
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
            on_plane = on_plane and plane._on_plane(point)

        return on_plane

    def _eight_dim(self):
        self.box.interaction = 5000
        self.box.friction = 0.05

        balls = self.balls
        nballs = 24
        radius = 100
        arr = ArrangeParticles(self.box)

        dpos = [0, 300, 0, 0, 0, 0, 0, 0]
        position = self.box.center - dpos[:self.box.dimensions]
        speed = numpy.array([0.5, 1, 0.3, 0, 0, 0, 0])*5
        ball = self.box.add_particle(
            mass=1, radius=radius, position=position[:self.box.dimensions], speed=speed[:self.box.dimensions], charge=nballs)
        balls.append(ball)

        balls += arr.random_balls(nballs, 1, radius, charge=-1)
        arr.set_charge_colors(balls)

    def p120_cell(self):
        self.box.friction = 0.02
        self.box.interaction = 0.0
        G = create_120cell()
        ball = self.arrangement.arrange_from_graph(G)


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

        dpoint = [450,0,0,0,0,0,0]
        point = self.box.center + dpoint[:self.box.dimensions]
        plane.add_hole(point[:self.box.dimensions], hole_size)

        self.box.planes.append(plane)

    def _test_many_holes(self):
        balls = self.balls
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

        for _ in range(nholes):
            N = 100
            repeat = True
            while N > 0 and repeat:
                on_plane = False
                while not on_plane:
                    point = self.box.random_position(hole_size)
                    point = plane.project_point(point)
                    if self.box.dimensions == 3:
                        on_plane = self.hole_on_plane(plane, point, hole_size)
                    else:
                        on_plane = True

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

    def many_interactions(self):
        self.box.friction = 0.0
        self.box.interaction = 0.0
        arr = ArrangeParticles(self.box)
        nballs = 300
        radius = 30
        arr.random_balls(nballs, 1, radius, 2, charge=1)
        arr.random_balls(nballs, 1, radius, 2, charge=-1)
        arr.set_charge_colors(self.box.particles)

    def _test_rotation(self):
        self.box.interaction = 15000
        # balls = self.arrangement.create_kube(100, self.box.center, 1)
        balls = self.arrangement.create_box(size=600, position=self.box.center, radius=50, charge=1)
        # balls = self.arrangement.create_simplex(size=600, position=self.box.center, radius=50, charge=1, vertices=None)
        for ball in balls:
            ball.speed = self.box.nullvector.copy()

        angle = math.pi/360
        rotations = [[0, 1, -1*angle], [1, 2, 0.3*angle], [2, 3, 2*angle]]
        # rotations = [[0, 1, -1*angle], [2, 3, 3*angle]]
        balls = self.arrangement.add_rotation_speed(rotations, center=self.box.center, balls=balls)
        # self.box.gravity = numpy.array([0,0,0,-1])
        return balls

    def _setup(self):
        # self.box.interaction = 5000
        balls = self.box.particles
        if self.charge_colors:
            self.layout.set_charge_colors(balls)

        self.box.get_radi(interaction_factor=self.interaction_factor, neighbor_count=self.neighbor_count)
        self.balls = self.box.particles

        return balls

    def make(self):
        if isinstance(self._setup_function, MethodType):
            self._setup_function()
        self._setup()
        return (self.box, self.balls)

    def show(self):
        # print stuff
        print(self.box)
        for plane in self.box.planes:
            print(plane)
            print(plane._projected_hull)
            for hole in plane.holes:
                (point, radius) = hole
                polygon = FlatD3Shape()
                polygon.regular_polygon_vertices(36)
                polygon.rotate(plane.unitnormal)
                polygon.move(point)
                polygon.scale(radius)
                points = polygon.vertices
                on_plane = True
                for point in points:
                    on_plane = on_plane and plane._on_plane(point)
                print(hole, on_plane)


    def run(self):
        # run the box
        for _ in range(1000):
            self.box.go()

        ticks = self.box.ticks
        return ticks


def main():  # pylint: disable=function-redefined
    ticks = 0
    print("START")

    if TEST:
        test = _Test()

        start = time.perf_counter()
        ticks = test.kdtree()
        end = time.perf_counter()
        dtime = end - start

        if not isinstance(ticks, int):
            ticks = 0
    else:
        setup = Setup()
        setup.make()
        setup.show()

        start = time.perf_counter()
        ticks = setup.run()
        end = time.perf_counter()
        dtime = end - start

    if ticks:
        print("\ntime = {:.2f}, tick/sec = {:.2f}".format(dtime, ticks/dtime))
    print("END")

if __name__ == "__main__":
    main()
