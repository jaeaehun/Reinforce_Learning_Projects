import pygame
from pygame.locals import *
import sys  # 외장 모듈
import math
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
WIDTH = 800
HEIGHT = 600
# 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyGame")
clock = pygame.time.Clock()

learning_rate = 0.005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


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


def select_action(k, s, g, re, o_1, o_2, o_3, o_4):
    luck = s
    reward = re

    if luck == 0:
        k.move(1, -1)
        reward -= 0.1

    elif luck == 1:
        k.move(1, 0)
        reward -= 0.1

    elif luck == 2:
        k.move(1, 1)
        reward -= 0.1

    elif luck == 3:
        k.move(0, 1)
        reward -= 0.1

    elif luck == 4:
        k.move(-1, 1)
        reward -= 0.1

    elif luck == 5:
        k.move(-1, 0)
        reward -= 0.1

    elif luck == 6:
        k.move(-1, -1)
        reward -= 0.1

    else:
        k.move(0, -1)
        reward -= 0.1

    if collide_check(k, g) == True:
        print("success")
        reward += 1

    if collide_check(k, o_1) == True:
        reward -= 0.5

    if collide_check(k, o_2) == True:
        reward -= 0.5

    if collide_check(k, o_3) == True:
        reward -= 0.5

    if collide_check(k, o_4) == True:
        reward -= 0.5

    if get_out_check(k) == True:
        reward -= 0.5


    done = end_episode()

    data = [[distance(k, g), distance(k, o_1), distance(k, o_2), distance(k, o_3), distance(k, o_4)]]
    dx = k.x - g.x
    dy = k.y - g.y
    #data = [[dx, dy]]

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


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

            return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.fc1 = nn.Linear(5, 256)  # 입력 state 2개
        self.fc2 = nn.Linear(256, 256)  # POLICY_NETWORK
        self.fc3 = nn.Linear(256, 8)  # value_network

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)

        print("q_out =", q_out.size())
        print("a=", a.size())

        q_a = torch.gather(q_out, 1, a)
        max_q_prime = q_target(s_prime).max(1)(0).unsqueeze(1)
        target = r + gamma*max_q_prime*done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def collide_check(character, goal):
    distance = math.sqrt((character.x - goal.x) ** 2 + (character.y - goal.y) ** 2)
    if distance < character.radius + goal.radius:
        return True
    else:
        return False


def get_out_check(k):
    if k.x > WIDTH-50 or k.x < 50 or k.y > HEIGHT-50 or k.y < 50:
        return True
    else:
        return False


def distance(x, y):
    dx = x.x - y.x
    dy = x.y - y.y
    dis = math.sqrt(dx ** 2 + dy ** 2)
    return dis


def env_reset(k):
    k.x = 600
    k.y = 400
    num = 0
    score = 0
    reward = 0
    data_1 = [[math.sqrt(300 ** 2 + 300 ** 2), math.sqrt(200 ** 2 + 100 ** 2), math.sqrt(400 ** 2 + 250 ** 2), 200, math.sqrt(150 ** 2 + 250 ** 2)]]
    #data_1 = [(500, 300)]

    return num, score, reward, data_1


def end_episode():
    clear = collide_check(me, goal)
    impact = collide_check(me, obstacle)
    impact_1 = collide_check(me, obstacle_1)
    impact_2 = collide_check(me, obstacle_2)
    impact_3 = collide_check(me, obstacle_3)
    get_out = get_out_check(me)
    if clear or impact or impact_1 or impact_2 or impact_3 or get_out:
        return True
    else:
        return False


def draw_env():
    me.draw()
    obstacle.draw()
    obstacle_1.draw()
    obstacle_2.draw()
    obstacle_3.draw()
    goal.draw()


me = Character(600, 400, 20, 3)
obstacle = Obstacle(400, 300, 20)
obstacle_1 = Obstacle(200, 150, 20)
obstacle_2 = Obstacle(600, 200, 20)
obstacle_3 = Obstacle(450, 150, 20)
goal = Goal(300, 100, 20)


while True:
    for n_epi in range(10000):

        q = Qnet()
        q_target = Qnet()
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer()

        print_interval = 20
        optimizer = optim.Adam(q.parameters(), lr=learning_rate)

        num, score, reward, data_1 = env_reset(me)
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s = data_1

        while not end_episode():

            clock.tick(1200)
            screen.fill((0, 0, 0))

            draw_env()

            a = q.sample_action(torch.tensor(data_1).float(), epsilon)
            s_prime, r, done = select_action(me, a, goal, reward, obstacle, obstacle_1, obstacle_2,
                                                      obstacle_3)

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/1000.0, s_prime, done_mask))
            s = s_prime
            score += r

            if memory.size() > 200:
                print("train_start")
                train(q, q_target, memory, optimizer)

            if n_epi % print_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100))

            pygame.display.update()
