# -*- coding: utf-8 -*-

from controller import Supervisor
import statistics
import math
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
summary = SummaryWriter()

writer = SummaryWriter(logdir='DQN')
loss_maen_lst = []

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

timetime = 32

robot_node = robot.getFromDef('car')
if robot_node is None:
    print("No DEF MY_ROBOT node found in the current world file\n")    

left1 = robot.getDevice('motor_1')
right1= robot.getDevice('motor_2')

left2 = robot.getDevice('motor_3')
right2= robot.getDevice('motor_4')

left1.setPosition(float('inf'))
right1.setPosition(float('inf'))

left2.setPosition(float('inf'))
right2.setPosition(float('inf'))

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

imu = robot.getDevice('inertial unit')
imu.enable(timestep)

learning_rate = 0.00025
gamma = 0.98
buffer_limit = 500000
batch_size = 64
itteration = 1000000

start_time = robot.getTime()

class Enviroment:

    def __init__(self) -> None:
        pass

    def prepare_episode(self):

        left1.setVelocity(0.0)
        right1.setVelocity(0.0) 
        left2.setVelocity(0.0)
        right2.setVelocity(0.0)
        translation_field = robot_node.getField('translation')
        rotation_field = robot_node.getField('rotation')

        new_value = [0, 0, 0.258]
        rotation_value = [0, 0, 1, 1.5708]
        translation_field.setSFVec3f(new_value)
        rotation_field.setSFRotation(rotation_value)

    def prepare_state(self, goal_x, goal_y):

        robot.step(1)

        lidar_point = lidar.getPointCloud()
        _, _, yaw = imu.getRollPitchYaw() 

        car_x, car_y = self.car_position()
        distance, robot_angle = self.distance(car_x, car_y, goal_x, goal_y, yaw)
        lidar_state, min_dis, obs_angle = self.point_cloud(lidar_point)

        prepare_state= (distance, robot_angle, min_dis, obs_angle) + tuple(lidar_state)
        state = prepare_state

        return state, distance

    def end_episode(self, num):

        if num >= 2999 :
            return True

        else:
            return False

    def done_mask(self, colli):
        
        if colli == True:

            return True
        
        else:
            return False

    def car_position(self):

        self.pos = robot_node.getPosition()        
        
        x = self.pos[0]
        y = self.pos[1]

        return x, y

    def distance(self, cx, cy, gx, gy, yaw):

        distance = math.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)

        trans_matrix = np.array([
                [math.cos(yaw), -math.sin(yaw), cx],
                [math.sin(yaw),math.cos(yaw), cy],
                [0 ,0 ,1 ]])
        local_point = np.linalg.inv(trans_matrix).dot([gx, gy ,1])
        goal_angle = math.atan2(local_point[1], local_point[0])

        return distance, goal_angle

    def point_cloud(self, point):
        lidar_distance_1 = []
        lidar_angle_1 = []

        for i in range(len(point)):
            point_distance = math.sqrt(point[i].x**2 + point[i].y**2) # + point[i].z**2)
            point_angle = math.atan2(point[i].y , point[i].x)

            if point_distance == float('inf'):
                lidar_distance_1.append(10)
                lidar_angle_1.append(point_angle)

            elif point_distance == np.nan:
                lidar_distance_1.append(0)
                lidar_angle_1.append(point_angle)

            else:

                lidar_distance_1.append(point_distance)
                lidar_angle_1.append(point_angle)
        
        #len(lidar_angle_1)
        min_dis_1 = min(lidar_distance_1)
        min_pos_1 = lidar_distance_1.index(min_dis_1)
        angle_1 =lidar_angle_1[min_pos_1]

        return lidar_distance_1, min_dis_1, angle_1

    def goal_check(self, distance, n):
        global reward

        if distance < 0.71 and n == 0:
            print("goal")
            reward = 1000
            n = 1

            return reward, True, n

        else:
            reward = 0
            n = 0
            return reward, False, n

    def collision(self, dis1, n):
        global reward

        #print("dis1 = {}, dis2 = {}, dis3 = {}, dis4 = {}".format(dis1, dis2, dis3 ,dis4))

        if dis1< 0.3 and n == 0:
            #print("collision")
            n = 1
            reward = -1000

            return reward, True, n

        else:
            reward = 0.0          
            n = 0
            return reward, False, n
            
    def goal_dis_reward(self, dis1, dis2, angle):

        global reward
        
        reward = 0

        rate = dis2/dis1
        
        angle_rate = angle/1.4
       
        #print("dis1 = {}, dis2 = {}, angle = {}, rate = {}".format(dis1, dis2, angle, rate))

        if rate < 1 and angle_rate < 1 :
            #print("near")

            reward = (100*(abs(dis1-dis2))/dis2)*(1- angle_rate)#/abs(angle)#*(1-math.cos(angle)/1.5708)

            #print("dis<1 r = {}".format(reward))

            return reward
          
            
        else:
            reward = 0
            
            return reward
            
    def obs_dis_reward(self, obs1, obs2):
        global reward
        
        reward = 0

        #print("obs1 = {}, obs2 = {}".format(obs1, obs2))

        rate = obs2/obs1

        safe_dis = 1.2

        if obs1 < safe_dis and rate > 1: 
            
            reward = 1
        
            #print("obs > 0 R = {}, rate = {}".format(reward, rate))

            return reward

        elif obs2 < safe_dis and rate <= 1:

            reward = -3.5/(obs2)
            #print("obs < 0 R = {}".format(reward))
            
            return reward

        else:
            reward = 0

            return reward

    def goal_position(self, num):

        if num == 0:

            x = 2#-2.5
            y = 2#3.5

            return x, y

        elif num == 1:

            x = 1#-5
            y = 4#0

            return x, y

        elif num == 2:

            x = -1#-1.5
            y = 4#-4

            return x, y

        elif num == 3:

            x = -2#3
            y = 2#-4.5

            return x, y

        elif num == 4:

            x = -5#3.25
            y = 3#1.5

            return x, y

        elif num == 5:

            x = 5#3
            y = 3#4

            return x, y

        elif num == 6:

            x = -2#-0.5
            y = -2#5

            return x, y

        else:
            x = 2 #5.25
            y = -3#5.25

            return x, y

    def graph(self, episode, score_list, mean_score_list):

        plt.cla()
        plt.xlabel("episode")
        plt.ylabel("score")
        #plt.subplot(222)
        plt.plot(episode, score_list, label="score")
        plt.plot(episode, mean_score_list, label="mean_score")
        plt.legend()

        plt.show(block=False)
        plt.pause(1)
        #plt.close()

    def graph_loss(self, episode, loss):

        plt.cla()
        plt.xlabel("episode")
        plt.ylabel("loss_mean")

        #plt.subplot(221)
        plt.plot(episode, loss, label="loss_mean")

        plt.show(block=False)
        plt.pause(1)
        #plt.close()

class Agent:

    def __init__(self) -> None:
        pass

    def action(self, num, goal_x, goal_y, count_g, count_c, init_dis, min_dis_0, ddd):
        
        if num == 0: # 0.50 m/s
            left1.setVelocity(3.33)
            right1.setVelocity(3.33)
            left2.setVelocity(3.33)
            right2.setVelocity(3.33)

        elif num == 1: # 0.5 rad/s
            left1.setVelocity(4.166) 
            right1.setVelocity(2.5)
            left2.setVelocity(4.166)
            right2.setVelocity(2.5)

        elif num == 2: # 0.5 rad/s
            left1.setVelocity(2.5)
            right1.setVelocity(4.166)
            left2.setVelocity(2.5)
            right2.setVelocity(4.166)

        elif num == 3: # 1rad/s
            left1.setVelocity(5)
            right1.setVelocity(1.666)
            left2.setVelocity(5)
            right2.setVelocity(1.666)

        elif num == 4: # 1rad/s
            left1.setVelocity(1.666)
            right1.setVelocity(5)
            left2.setVelocity(1.666)
            right2.setVelocity(5)

        elif num == 5: # 1.5rad/s
            left1.setVelocity(5.833)
            right1.setVelocity(0.833)
            left2.setVelocity(5.833)
            right2.setVelocity(0.833)

        elif num == 6: # 1.5 rad/s
            left1.setVelocity(0.833)
            right1.setVelocity(5.833)
            left2.setVelocity(0.833)
            right2.setVelocity(5.833)

        robot.step(timetime)

        #nxt_vel_x, nxt_vel_y, _, _, _, nxt_angular_vel = robot_node.getVelocity()

        nxt_car_x, nxt_car_y = env.car_position()
        lidar_point = lidar.getPointCloud()
        _, _, next_yaw = imu.getRollPitchYaw()
        next_dis, next_robot_angle = env.distance(nxt_car_x, nxt_car_y, goal_x, goal_y, next_yaw)
        next_lidar_state, min_dis_1, nxt_angle_obs= env.point_cloud(lidar_point)

        _, goal ,_ = env.goal_check(next_dis, count_g)

        _, collide, _ = env.collision(min_dis_1, count_c)

        goal_distance_reward = env.goal_dis_reward(init_dis, next_dis, next_robot_angle)
        #angle_reward = env.goal_angle_reward(next_robot_angle)
        obstacle_distance_reward = env.obs_dis_reward(min_dis_0, min_dis_1)
        goal_reward, _, count_g = env.goal_check(next_dis, count_g)
        collision_reward, _, count_c = env.collision(min_dis_1, count_c)

        if collide == True:
            env.prepare_episode()
            count_c = 0

        done = env.done_mask(collide)
        done_num = 0.0 if done else 1.0

        
        test_reward = goal_distance_reward + obstacle_distance_reward + goal_reward + collision_reward
        prepare_state_prime = (next_dis, next_robot_angle, min_dis_1, nxt_angle_obs)+tuple(next_lidar_state)
        state_prime_r = prepare_state_prime

        #print("goal_dis_R = {}, obs_dis_R = {}, test_reward = {}".format(goal_distance_reward, obstacle_distance_reward, test_reward))

        return state_prime_r, test_reward, done_num, goal, count_g, collide

    def sample_action(self, obs, epsilon):
        #q.eval()
        #with torch.no_grad():
            #print(q.eval())
        out = q.forward(obs)

        coin = random.random()

        if coin < epsilon:

            return  random.randrange(0, 7)
        else:

            return out.argmax().item()

class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()

        self.fc1 = nn.Linear(132, 128)  # 입력 state 4개
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 128) 
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 7)  

        self.bn1 = nn.BatchNorm1d(128)#, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(128)#, track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(128)#, track_running_stats=True)
        #self.bn4 = nn.BatchNorm1d(256, track_running_stats=True)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):


        x = F.relu(self.fc1(x))
       # x = self.bn1(x)
        #x = self.dropout(x)

        x = F.relu(self.fc2(x))
       # x = self.bn2(x)
       # x = self.dropout(x)

        x = F.relu(self.fc3(x))
      #  x = self.bn3(x)
      #  x = self.dropout(x)

        x = F.relu(self.fc4(x))

        x = self.fc5(x)

        return x



    def study(self, q, q_target, memory, optimizer, batch_size):
        #q.train()
        #s, a, r, s_prime, done_mask = memory.sample(batch_size)
        #print("q_target_s_prime=", q_target(s_prime))
        loss_lis = []
        for i in range(30):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)

            #print("s shape =", s.shape)

            q_out = q(s)
            
            #print("q_out =", q_out)

            q_a = torch.gather(q_out, 1, a)
            
            #print("action =", a)
            #print("q_a=", q_a)

            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)#torch.max(q_target(s_prime))
            #print(max_q_prime )
            target = r + gamma*max_q_prime*done_mask
            loss = F.smooth_l1_loss(target, q_a)
            loss_lis.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #loss_mean = statistics.mean(loss_lis)
        #loss_maen_lst.append(loss_mean)
        
        #env.graph_loss(episode_lst, loss_maen_lst)
        #writer.add_scalar('training loss', loss , i)
        #writer.close()
        #print("train_finish")


class ReplayBuffer:

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


best_score = 0

env = Enviroment()
car = Agent()
q = Qnet()
q_target = Qnet()
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
q_target.load_state_dict(q.state_dict())

#q = torch.load('/home/jaehun/DQN_network' + 'model.pt') 

score_lst = []
mean_score_lst = []
episode_lst = []

for n_epi in range(itteration):

    if n_epi % 20 ==0 and n_epi !=0:
        print("target_network_update")
        q_target.load_state_dict(q.state_dict())
        
    print("memory = ", memory.size())
    env.prepare_episode()
    goal_num = random.randrange(0, 8)
    goal_x, goal_y = env.goal_position(goal_num)
    state, init_dis =env.prepare_state(goal_x, goal_y)
    print("episode_state =", state)
    epsilon = max(0.01, 0.30 - 0.01*(n_epi/100))
    collide_count = False
    reward = 0
    score = 0
    step = 0
    count_c = 0
    count_g = 0
    goal_count_lst = []
    collision_count_lst = []    

    print("episode = {}, goal_number = {}, goal_x = {},goal_y={}, best_score = {}, epsilon = {}".format(n_epi +1, goal_num, goal_x,goal_y,best_score, epsilon*100))

    while robot.step(timestep) != -1 and collide_count == False and step < 3000:
        
        lidar_point = lidar.getPointCloud()
        _, _, yaw = imu.getRollPitchYaw() 
        #vel_x, vel_y, _, _, _, angular_vel = robot_node.getVelocity()

        input = torch.tensor(state).float()
        #print("input =", input)
        select_action = car.sample_action(input, epsilon)
        state_prime, input_reward, done_num, g_check, count_g, collide_count = car.action(select_action, goal_x, goal_y, count_g, count_c, state[0], state[2], init_dis)   
        score += input_reward
        
        memory.put((state, select_action, input_reward, state_prime, done_num))

        #print("state_prime =", state_prime)
        
        state = state_prime

        #print("after_state = ", state)

        if collide_count == True:
            collision_count_lst.append(1)

            break

        if count_g == 1:
            goal_count_lst.append(1)
            key = 0
            while key != 1:
                new_num = random.randrange(0, 8)
                if new_num != goal_num:
                    key = 1

                else: 
                    key =0

            goal_x, goal_y = env.goal_position(new_num)
            goal_num = new_num
            print("new_num ={}, goal_x = {}, goal_y = {}".format(new_num, goal_x, goal_y))
            count_g = 0
            g_check == False
            #car_x, car_y = env.car_position()
            #distance, robot_angle = env.distance(car_x, car_y, goal_x, goal_y, yaw)

        step += 1
            
    if memory.size() > 2000:
        q.study(q, q_target, memory, optimizer, batch_size)




































    episode_lst.append(n_epi+1)
    
   

    if len(collision_count_lst)+len(goal_count_lst) !=0:
        if len(goal_count_lst)*100/(len(collision_count_lst)+len(goal_count_lst)) >90 or n_epi % 2 ==0:
            with torch.no_grad():
                print("save_trained_parameter")
                path = '/home/jaehun/DQN_network'
                torch.save(q, path + 'model.pt')
        

    if score > best_score:
        best_score = score

            
    val = len(collision_count_lst)+len(goal_count_lst)
    
    if val != 0:           
 
        print("# of episode :{}, score : {:.1f}, goal_rate = {}, collision_rate = {}, goal_count = {}, collide_count = {}".format(n_epi+1, score, len(goal_count_lst)*100/(len(collision_count_lst)+len(goal_count_lst)), len(collision_count_lst)*100/(len(collision_count_lst)+len(goal_count_lst)), len(goal_count_lst), len(collision_count_lst)))

    elif val == 0:
        print("NO collide and No goal")
        
    

    score_lst.append(score)
    mean_score_lst.append(statistics.mean(score_lst))
    #env.graph_loss(episode_lst, loss_maen_lst)
    env.graph(episode_lst, score_lst, mean_score_lst)


