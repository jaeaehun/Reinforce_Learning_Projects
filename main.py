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

    if luck == 0:
        k.move(1, -1)
        reward -= 1

    elif luck == 1:
        k.move(1, 0)
        reward -= 1

    elif luck == 2:
        k.move(1, 1)
        reward -= 1

    elif luck == 3:
        k.move(0, 1)
        reward -= 1

    elif luck == 4:
        k.move(-1, 1)
        reward -= 1

    elif luck == 5:
        k.move(-1, 0)
        reward -= 1

    elif luck == 6:
        k.move(-1, -1)
        reward -= 1

    else:
        k.move(0, -1)
        reward -= 1

    if collide_check(k, g) == True:
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

        self.loss_lst = []  # loss들을 모아두기 위한 배열

        self.fc1 = nn.Linear(2, 128)  # 입력 state 2개
        self.fc_pi = nn.Linear(128, 8)  # POLICY_NETWORK
        self.fc_vel = nn.Linear(128, 1)  # value_network

        model = nn.Linear(1, 1)

        self.optimizer = optim.Adam(model.parameters(), lr = 0.002)  # 딥러닝 최적화 방식

    def policy_network(self, x, softmax_dim=1):
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
        #loss = torch.cat(self.loss_lst).sum

        #print(type(loss_cat))

        loss = torch.cat(self.loss_lst).mean() #/ len(self.loss_lst)   loss 평균 내주기

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_lst = []


me = Character(800, 600, 20, 5)
obstacle = Obstacle(400, 300, 20)
obstacle_1 = Obstacle(200, 150, 20)
obstacle_2 = Obstacle(600, 200, 20)
obstacle_3 = Obstacle(450, 150, 20)
goal = Goal(100, 100, 20)

model = ActorCritic()

gamma = 0.95
print(torch.cuda.is_available())

while True:
    for n in range (1000):
        clear = False
        impact = False
        impact_1 = False
        impact_2 = False
        impact_3 = False


        score = 0
        reward = 0
        me.x = 800
        me.y = 600
        t=0

        print("in for")


        #dx, dy = distance(me, goal)
        data_1 = [[700, 500]]

        while clear or impact or impact_1 or impact_2 or impact_3 == False:

            clock.tick(60)
            screen.fill((0, 0, 0))

            clear = collide_check(me, goal)

            impact = collide_check(me, obstacle)
            impact_1 = collide_check(me, obstacle_1)
            impact_2 = collide_check(me, obstacle_2)
            impact_3 = collide_check(me, obstacle_3)

            me.draw()
            obstacle.draw()
            obstacle_1.draw()
            obstacle_2.draw()
            obstacle_3.draw()
            goal.draw()

            for event in pygame.event.get():
                if n == 1000:
                    pygame.quit()
                    sys.exit()



            if me.x == 800 and me.y == 600 and t ==0:
                    state = torch.tensor(data_1).float()
                    t += 1
            else:
                state = torch.tensor(data).float()

            probability = model.policy_network(state)
            value = model.value_network(state)

            pdf = Categorical(probability)
            action = pdf.sample()
            reward = select_action(me, action.item(), goal, reward)
            #print("reward =", reward)


            dx, dy = distance(me, goal)
            data = [[dx, dy]]

            state_prime = torch.tensor(data).float()
            next_value = model.value_network(state_prime)
            delta = reward + gamma * next_value - value
            loss = -torch.log(probability) * delta.item() + delta * delta

            model.gather_loss(loss)
            score += reward
            #print("score=", score)

            if clear or impact or impact_1 or impact_2 or impact_3 == True:
                #print("impact")
                break

            pygame.display.update()


        model.train()

        if n % 20 == 0 and n != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n, score / 10))





