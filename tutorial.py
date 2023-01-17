import pygame
from pygame.locals import *
import sys  # 외장 모듈
import math

# 초기화
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("PyGame")
clock = pygame.time.Clock()


class Character:
    def __init__(self, x, y, radius, speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed

    def draw(self):
        pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), self.radius, 0)

    def move(self, x, y):
        self.x += x
        self.y += y

    def move_to(self, x, y):
        self.x = x
        self.y = y


class Goal:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius, 0)


def collide_check(character, goal):
    distance = math.sqrt((character.x - goal.x) ** 2 + (character.y - goal.y) ** 2)
    if distance < character.radius + goal.radius:
        return True
    else:
        return False


me = Character(400, 300, 20, 5)
goal = Goal(400, 100, 20)

while True:
    clock.tick(60)
    screen.fill((0, 0, 0))

    me.draw()
    goal.draw()
    me.move_to(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
    print(collide_check(me, goal))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
