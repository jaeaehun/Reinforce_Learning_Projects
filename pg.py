import pygame
from pygame.locals import *
import sys  # 외장 모듈
import math


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

    #data = [[distance(k, g), distance(k, o_1), distance(k, o_2), distance(k, o_3), distance(k, o_4)]]
    dx = k.x - g.x
    dy = k.y - g.y
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
    #data_1 = [[math.sqrt(500 ** 2 + 300 ** 2), math.sqrt(200 ** 2 + 100 ** 2), math.sqrt(400 ** 2 + 250 ** 2), 200, math.sqrt(150 ** 2 + 250 ** 2)]]
    data_1 =  [[500, 300]]

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


class ActorCritic:
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.data = []

        self.fc1 = nn.Linear(2, 256)  # 입력 state 2개
        self.fc_pi = nn.Linear(256, 8)  # POLICY_NETWORK
        self.fc_vel = nn.Linear(256, 1)  # value_network
        self.gamma = 0.95
        #model = nn.Linear(1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)  # 딥러닝 최적화 방식

    def policy_network(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        #print("x = relu", x)
        x = self.fc_pi(x)
        #print("x = pi", x)
        act = F.softmax(x, dim=softmax_dim)  # 8 방향 중에서 어디로 갈지에 대한 확률로 나옴
        #print("act=", act)
        return act

    def policy_network_practice(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        #print("x = relu", x)
        x = self.fc_pi(x)
        #print("x = pi", x)
        act = F.softmax(x, dim=softmax_dim)  # 8 방향 중에서 어디로 갈지에 대한 확률로 나옴
        #print("act=", act)
        return act

    def value_network(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc_vel(x)  # 현재 state의 상태가치함수가 나옴

        return value

    def gather_data(self, transition):

        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        #print("s =", len(s_batch))
        #print("a_batch =", a_batch)
        #print("r =", len(r_batch))
        #print("s_p =", len(s_prime_batch))
        #print("d =", len(done_batch))

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        #print("a=", a)
        td_target = r + gamma * self.value_network(s_prime) * done

        delta = td_target - self.value_network(s)

        pi = self.policy_network_practice(s).squeeze()
        #print("pi= ", pi)
        #print("pi_sum", pi.sum())
        pi_a = torch.gather(pi, -2, a)
        #print("pi_a =", pi_a)
        loss = -torch.log(pi_a) * delta.detach() + delta*delta #F.smooth_l1_loss(self.value_network(s), td_target.detach())
        #print("loss=", loss)
        #print("loss_mean=", loss.mean())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

class PolicyGradient(nn.Module):
    def __init__(self):
        super(PolicyGradient, self).__init__()
        self.gama = 0.95

        self.data = []

        self.fc1 = nn.Linear(2, 128)  # 입력 state 2개
        self.fc2 = nn.Linear(128, 8)  # POLICY_NETWORK

        #model = nn.Linear(1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = 0.0003)  # 딥러닝 최적화 방식

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        #print("sum=", x.sum())
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_pi(self):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + R*self.gama
            loss = -log_prob*R
            print("loss =", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.data = []


me = Character(600, 400, 20, 8)
obstacle = Obstacle(400, 300, 20)
obstacle_1 = Obstacle(200, 150, 20)
obstacle_2 = Obstacle(600, 200, 20)
obstacle_3 = Obstacle(450, 150, 20)
goal = Goal(100, 100, 20)

#model = ActorCritic()
pi = PolicyGradient()


while True:
    for n in range(10000):
        num, score, reward, data_1 = env_reset(me)

        while not end_episode() and score > -5000000:

            clock.tick(12000)
            screen.fill((0, 0, 0))

            draw_env()

            for event in pygame.event.get():
                if n == 1000:
                    pygame.quit()
                    sys.exit()

            if me.x == 600 and me.y == 400 and num == 0:
                    state = data_1
                    #probability = model.policy_network(torch.tensor(state).float())
                    out = pi(torch.tensor(state).float()) # +0.000000001
                    num += 1
            else:

                state = state_prime

                #probability = model.policy_network(torch.tensor(state).float())
                out = pi(torch.tensor(state).float()) #+0.000000001


            #pdf = Categorical(probability)
            m = Categorical(out)
            act = m.sample()
            act_1 = act.item()

            #action = pdf.sample().item()

           # state_prime, reward, done = select_action(me, action, goal, reward, obstacle, obstacle_1, obstacle_2, obstacle_3)
            state_prime, reward, done = select_action(me, act_1, goal, reward, obstacle, obstacle_1, obstacle_2,
                                                      obstacle_3)

            #model.gather_data((state, action, reward, state_prime, done))

            #print("state_size=", torch.tensor(state).float().size())
            #print("out =", out)
            #print("out_size=", out.size())
            #print("act =", act)
            #print("act =", act.size())

            k = torch.nan_to_num(torch.log(out[act]), nan=0.0, posinf=1e10, neginf=-1e10)
            print("log =", k)

            pi.put_data((reward, k))

            #print(torch.log(out[act]))

            score += reward

            pygame.display.update()

        print(n+1, "train_start")
        #model.train_net()
        pi.train()
        print("# of episode :{}, avg score : {:.1f}".format(n, score / 20))