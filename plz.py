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
WIDTH = 800
HEIGHT = 600
# 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
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
        self.x += x * self.speed
        self.y += y * self.speed

    def move_to(self, x, y):
        self.x = x
        self.y = y


def select_action(k, s, g, re,o_1, o_2, o_3, o_4):
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
        reward += 5000

    if collide_check(k, o_1) == True:
        reward -= 500

    if collide_check(k, o_2) == True:
        reward -= 500

    if collide_check(k, o_3) == True:
        reward -= 500

    if get_out_check(o_4) == True:
        reward -= 10
        print("fuck")

    done = end_episode()
    dx, dy = distance(k, g)
    data = [[dx, dy]]

    return data, reward, done


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


def get_out_check(k):
    if k.x > WIDTH or k.x < 0 or k.y > HEIGHT or k.y < 0:
        return True
    else:
        return False


def distance(x, y):
    dx = x.x - y.x
    dy = x.y - y.y

    return dx, dy

def env_reset(k):
    k.x = 600
    k.y = 400
    num = 0
    score = 0
    reward = 0
    data_1 = [[500, 300]]

    return num, score, reward, data_1

def end_episode():
    clear = collide_check(me, goal)
    impact = collide_check(me, obstacle)
    impact_1 = collide_check(me, obstacle_1)
    impact_2 = collide_check(me, obstacle_2)
    impact_3 = collide_check(me, obstacle_3)
    get_out = get_out_check(me)

    if clear and impact and impact_1 and impact_2 and impact_3 and get_out == False:
        ans = False
        return ans

    else:
        ans = True
        return ans

def draw_env():
    me.draw()
    obstacle.draw()
    obstacle_1.draw()
    obstacle_2.draw()
    obstacle_3.draw()
    goal.draw()


class ActorCritic:
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.loss_lst = []  # loss들을 모아두기 위한 배열
        self.data = []

        self.fc1 = nn.Linear(2, 128)  # 입력 state 2개
        self.fc_pi = nn.Linear(128, 8)  # POLICY_NETWORK
        self.fc_vel = nn.Linear(128, 1)  # value_network

        model = nn.Linear(20, 10)

        self.optimizer = optim.Adam(model.parameters(), lr = 0.0001)  # 딥러닝 최적화 방식

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

    def gather_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            state, action, reward, state_prime_prime, done = transition
            s_lst.append(state)
            a_lst.append([action])
            r_lst.append([reward])
            s_prime_lst.append(state_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.value_network(s_prime) * done
        delta = td_target - self.value_network(s)

        pi = self.policy_network(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.value_network(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


me = Character(600, 400, 20, 20)
obstacle = Obstacle(400, 300, 20)
obstacle_1 = Obstacle(200, 150, 20)
obstacle_2 = Obstacle(600, 200, 20)
obstacle_3 = Obstacle(450, 150, 20)
goal = Goal(100, 100, 20)

model = ActorCritic()

gamma = 0.95
print(torch.cuda.is_available())

while True:
    for n in range (10000):

        num, score, reward, data_1 = env_reset(me)

        while end_episode() is False or score > -200:

            clock.tick(60)
            screen.fill((0, 0, 0))

            draw_env()

            for event in pygame.event.get():
                if n == 1000:
                    pygame.quit()
                    sys.exit()

            if me.x == 600 and me.y == 400 and num == 0:
                    #state = torch.tensor(data_1).float()
                    state = data_1
                    probability = model.policy_network(torch.tensor(state).float())
                    num += 1
            else:
                #state = torch.tensor(data).float()
                state = state_prime
                model.policy_network(torch.tensor(state).float())

            pdf = Categorical(probability)
            action = pdf.sample().item()
            state_prime, reward, done = select_action(me, action, goal, reward, obstacle, obstacle_1, obstacle_2, obstacle_3)

            score += reward

            #if done == True:
                #("episode end")
                #break

            pygame.display.update()

        print("yeah")
        model.train_net()

        if n % 20 == 0 and n != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n, score / 10))