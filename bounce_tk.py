from tkinter import *
from random import randint
from gas import *
import numpy

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

class Ball:
    def __init__(self, canvas, box, ball):
        R = ball.radius
        x1 = round(ball.position[0] - R)
        x2 = round(ball.position[0] + R)
        y1 = round(ball.position[1] - R)
        y2 = round(ball.position[1] + R)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.box = box
        self.ball = ball
        self.canvas = canvas
        self.object = canvas.create_oval(self.x1, self.y1, self.x2, self.y2, fill="red")

    def move_ball(self):
        old = numpy.array(self.ball.position[:])
        self.ball.move()
        self.ball.bounce(self.box)
        for b2 in self.box.particles:
            self.ball.collide(b2)
        delta = self.ball.position - old
        self.canvas.move(self.object, delta[0], delta[1])
        self.canvas.after(60, self.move_ball)

def main():
    # initialize root Window and canvas
    root = Tk()
    root.title("Balls")
    root.resizable(False,False)
    canvas = Canvas(root, width = SCREEN_WIDTH, height = SCREEN_HEIGHT)
    canvas.pack()

    box = Box([SCREEN_WIDTH, SCREEN_HEIGHT])
    for i in range(10):
        mass = random.randrange(5, 50)
        box.add_particle(mass, mass)

    balls = []
    for ball in box.particles:
        balls.append(Ball(canvas, box, ball))

    for ball in balls:
        ball.move_ball()
    root.mainloop()

if __name__ == "__main__":
    main()