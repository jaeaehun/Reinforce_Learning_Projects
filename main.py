import pygame
from pygame.locals import *
import sys  # 외장 모듈
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

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


def select_action(k, s, g, re):
    luck = s
    reward = re

    if luck < 0.125:
        k.move(1, -1)
        reward -= 1

    elif luck < 0.25:
        k.move(1, 0)
        reward -= 1

    elif luck < 0.375:
        k.move(1, 1)
        reward -= 1

    elif luck < 0.5:
        k.move(0, 1)
        reward -= 1

    elif luck < 0.625:
        k.move(-1, 1)
        reward -= 1

    elif luck < 0.75:
        k.move(-1, 0)
        reward -= 1

    elif luck < 0.875:
        k.move(-1, -1)
        reward -= 1

    else:
        k.move(0, -1)
        reward -= 1

    if collide_check(k, g):
        reward -= 100

    return reward


class Goal:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius, 0)


class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self):
        pygame.draw.circle(screen, (0, 255, 0), (self.x, self.y), self.radius, 0)


def collide_check(character, goal):
    distance = math.sqrt((character.x - goal.x) ** 2 + (character.y - goal.y) ** 2)
    if distance < character.radius + goal.radius:
        return True
    else:
        return False


def distance(x, y):
    dx = x.x - y.x
    dy = x.y - y.y

    return dx, dy


class ActorCritic:
    def __init__(self):
        super(ActorCritic, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss_lst = []  # loss들을 모아두기 위한 배열

        self.fc1 = nn.Linear(2, 256)  # 입력 state 2개
        self.fc_pi = nn.Linear(256, 2)  # POLICY_NETWORK
        self.fc_vel = nn.Linear(256, 1)  # value_network
        model = nn.Linear(1, 1)

        self.optimizer = optim.Adam(model.parameters(), lr=0.002)  # 딥러닝 최적화 방식

    def policy_network(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        act = F.softmax(x, dim=softmax_dim)  # 가속 또는 감속을 할 확률이 나옴
        return act

    def value_network(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc_vel(x)  # 현재 state의 상태가치함수가 나옴

        return value

    def gather_loss(self, loss):
        self.loss_lst.append(loss.unsqueeze(0))  # loss를 모아둔다. unsqueeze(0)는 차원을 높여주는 것

    def train(self):
        loss_cat = torch.cat(self.loss_lst).sum
        loss_mean = loss_cat / len(self.loss_lst)  # loss 평균 내주기

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        self.loss_lst = []


me = Character(800, 600, 20, 5)
obstacle = Obstacle(400, 300, 20)
obstacle_1 = Obstacle(200, 150, 20)
obstacle_2 = Obstacle(600, 200, 20)
obstacle_3 = Obstacle(450, 150, 20)
goal = Goal(400, 100, 20)

model = ActorCritic()

gamma = 0.95

while True:

    clock.tick(60)
    screen.fill((0, 0, 0))

    me.draw()
    obstacle.draw()
    obstacle_1.draw()
    obstacle_2.draw()
    obstacle_3.draw()
    goal.draw()

    dx, dy = distance(me, goal)

    clear = collide_check(me, goal)

    fuck = collide_check(me, obstacle)
    fuck_1 = collide_check(me, obstacle_1)
    fuck_2 = collide_check(me, obstacle_2)
    fuck_3 = collide_check(me, obstacle_3)

    data = [dx, dy]

    for n in range(10000):

        score = 0
        reward = 0

        while not clear or (fuck and fuck_1 and fuck_2 and fuck_3):

            state = torch.tensor(data)
            probability = model.policy_network(state)
            value = model.value_network(state)
            pdf = Categorical(probability)
            action = pdf.sample()
            reward = select_action(me, action.item(), goal, reward)

            state_prime = torch.tensor(data)
            next_value = model.value_network(state_prime)
            delta = reward + gamma * next_value - value
            loss = -torch.log(probability) * delta.item() + delta * delta

            model.gather_loss(loss)
            score += reward

            if clear:
                break

        model.train()

        if n % 20 == 0 and n != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n, score / 10))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
