'''강화학습_라이브러리_호출'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
"""---------------------"""

'''게임 호출'''
import engine
"""--------"""

"""Hyperparameters"""
learning_rate = 1
gamma = 1
n_rollout = 1
"""---------------"""


class ACTOR_CRITIC:
    def __init__(self):
        super(ACTOR_CRITIC, self).__init__()
        self.loss_lst = [] # loss들을 모아두기 위한 배열

        self.fc1 = nn.linear(6, 256) #입력 state 6개
        self.fc_pi = nn.linear(256, 2) #POLICY_NETWORK
        self.fc_vel = nn.linear(256, 1) #value_network

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate) #딥러닝 최적화 방식


    def policy_network(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        act = F.softmax(x, dim = softmax_dim) #가속 또는 감속을 할 확률이 나옴

        return act


    def value_network(self, x,):
        x = F.relu(self.fc1)
        value = self.fc_vel(x) #현재 state의 상태가치함수가 나옴

        return value


    def gather_loss(self, loss):
        self.loss_lst.append(loss.unsqueeze(0)) # loss를 모아둔다. unsqueeze(0)는 차원을 높여주는 것


    def train(self):
        loss = torch.cat(self.loss_lst).sum
        loss = loss/len(self.loss_lsts) # loss 평균 내주기

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_lst = []

class project:

    def __init__(self):
        enviroment = engine #게임에서 차량(ego, 상대) 위치 numpy 형태로 부르기
        model = ACTOR_CRITIC()
        self.episode_time = 10000

    def start(self):

        for n_time in range(self.episode_time): #에피소드가 10000번 실행할때까지

            while(#사고가 나기전까지):
                state = torch.tensor(enviroment) #numpy 형태였던 state를 tensor형태로 바꾸기

                probability = model.policy_network(state) #policy_network에서 가속, 감속을 할 확률을 받음
                value = model.value_network(state) #상태가치함수를 받음

                Pdf = Categorical(probability) #가감속 할 확률을 확률밀도함수로 만듬
                action = pdf.sample() #확률밀도함수에서 가속이나 감속을 선택해서 운전함





