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


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getFromDef('car')
if robot_node is None:
    print("No DEF MY_ROBOT node found in the current world file\n")
    
if robot_node is True:
    print("fuck")

left1 = robot.getDevice('motor_1')
right1= robot.getDevice('motor_2')

left1.setPosition(float('inf'))
right1.setPosition(float('inf'))

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

imu = robot.getDevice('inertial unit')
imu.enable(timestep)

learning_rate = 0.0002
gamma = 0.98
buffer_limit = 5000000
batch_size = 128
itteration = 1000
final_step = 3000


class Enviroment:

    def __init__(self) -> None:
        pass

    def prepare_episode(self):

        left1.setVelocity(0.0)
        right1.setVelocity(0.0) 
    
        translation_field = robot_node.getField('translation')

        new_value = [0, 0, 0]
        translation_field.setSFVec3f(new_value)

    def end_episode(self, num):

        if num >=5999:
            return True

        else:
            return False

    def done_mask(self, step_end, colli):
        
        if step_end or colli == True:

            return True
        
        else:
            return False

    def car_position(self):

        self.pos = robot_node.getPosition()
        
        x = self.pos[0]
        y = self.pos[1]

        return x, y

    def distance(self, cx, cy, gx, gy):

        distance = math.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
        goal_angle = math.atan2((gy-cy),(gx-cx))

        return distance, goal_angle

    def point_cloud(self, point, num):
        lidar_distance = []
        lidar_angle = []

        for i in range(num):
            point_distance = math.sqrt(point[i].x**2 + point[i].y**2 + point[i].z**2)
            point_angle = math.atan(point[i].y / point[i].x)

            lidar_distance.append(point_distance)
            lidar_angle.append(point_angle)

        min_dis = min(lidar_distance)

        min_pos = lidar_distance.index(min_dis)

        angle = lidar_angle[min_pos]

        return min_dis, angle


    def goal_check(self, distance):
        global reward

        if distance < 0.71:
            print("goal")
            reward = 2000

            return reward, True

        else:
            reward = 0
            return reward, False

    def collision(self, lidar_distance):
        global reward

        if lidar_distance < 0.3:
            print("collision")
            reward = -500

            return reward, True

        else:
            reward = 0            
            return reward, False
            
    def goal_dis_reward(self, dis1, dis2):
        global reward

        if dis2/dis1 < 1:
            reward = 5
            
            return reward

        else:
            reward = 1

            return reward

    def goal_angle_reward(self, angle, yaw):

        global reward

        if abs(angle - yaw) < 1:

            reward = 5

        else:

            reward = -1

        return reward

    def obs_dis_reward(self, dis1, dis2):
        global reward

        if dis1 < 0.6:
            if dis2/dis1 < 1:

                reward = -3

            return reward

        else:
            reward = 0

            return reward

    def nerd(self, num, x, y):
        global reward
        if num >= 500:
            if math.sqrt(x**2 + y**2) < 3:
                #print("i'm nerd")
                reward = -1000
            
            return reward, True

        else:

            reward = 0

            return reward, False


    def re_goal_pos(self, past_num, current_num):

        if past_num == current_num:
            while current_num != past_num:
                current_num = random.randrange(0, 8)

            return current_num

        else:
            return current_num

    def goal_position(self, num):

        if num == 0:

            x = 7
            y = 4

            return x, y

        elif num == 1:

            x = 3
            y = 5

            return x, y

        elif num == 2:

            x = -5
            y = 5

            return x, y

        elif num == 3:

            x = -3
            y = 7

            return x, y

        elif num == 4:

            x = -7
            y = -5

            return x, y

        elif num == 5:

            x = -3
            y = -5

            return x, y

        elif num == 6:

            x = 6
            y = -7

            return x, y

        else:
            x = 9
            y = -3

            return x, y


class Agent:

    def __init__(self) -> None:
        pass

    def action(self, num):

        if num == 0:
            left1.setVelocity(5.0)
            right1.setVelocity(5.0)

        elif num == 1:
            left1.setVelocity(5.0)
            right1.setVelocity(0.0)

        elif num == 2:
            left1.setVelocity(0.0)
            right1.setVelocity(5.0)

        elif num == 3:
            left1.setVelocity(-5.0)
            right1.setVelocity(5.0)

        elif num == 4:
            left1.setVelocity(0.0)
            right1.setVelocity(0.0)

    def sample_action(self, obs, epsilon):

        out = q.forward(obs)
        coin = random.random()
        if coin < epsilon:

            return  random.randrange(0,5)
        else:

            return out.argmax().item()


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()

        self.fc1 = nn.Linear(4, 256)  # 입력 state 4개
        self.fc2 = nn.Linear(256, 256)  
        self.fc3 = nn.Linear(256, 5)  

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        #x = self.bn1(x.unsqueeze(1))
        x = F.relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn2(x.unsqueeze(1))
        x = self.fc3(x)

        return x

    def train(self, q, q_target, memory, optimizer, batch_size):
        for i in range(50):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)

            q_out = q(s)
            q_a = torch.gather(q_out, 1, a)

            max_q_prime = torch.max(q_target(s_prime))
            target = r + gamma*max_q_prime*done_mask
            loss = F.smooth_l1_loss(target, q_a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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

for n_epi in range(itteration):

    env = Enviroment()
    car = Agent()

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    env.prepare_episode()
    goal_num = random.randrange(0, 8)
    new_goal_num = goal_num
    goal_x, goal_y = env.goal_position(goal_num)
    epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
    step_finish = False
    reward = 0
    score = 0
    step = 0
    
    score_lst = []
    best_score_lst = []
    episode_lst = []
    episode_lst.append(n_epi + 1)
    print("episode = {}, goal_number = {}, best_score = {}".format(n_epi +1, goal_num, best_score))

    action_lst = []
        
    while robot.step(timestep) != -1 and step < 6000:
        
        lidar_point = lidar.getPointCloud()
        _, _, yaw = imu.getRollPitchYaw() 

        car_x, car_y = env.car_position()
        distance, robot_angle = env.distance(car_x, car_y, goal_x, goal_y)
        obstacle_distance, obstacle_angle = env.point_cloud(lidar_point, len(lidar_point))
        dif_angle = abs(robot_angle - yaw)

        state = (distance, dif_angle, obstacle_distance, obstacle_angle)
        print("dis = {}, dif_angle = {}, obs_dis = {}, obs_angle".format(distance, dif_angle, obstacle_distance,obstacle_angle))

        select_action = car.sample_action(torch.tensor(state).float(), epsilon)
        action_lst.append(select_action)

        action = car.action(select_action)        

        nxt_car_x, nxt_car_y = env.car_position()
        _, _, next_yaw = imu.getRollPitchYaw()
        next_dis, next_robot_angle = env.distance(nxt_car_x, nxt_car_y, goal_x, goal_y)
        next_obs, next_angle = env.point_cloud(lidar_point, len(lidar_point))
        next_dif_angle = abs(next_robot_angle - next_yaw)

        state_prime = (next_dis, next_dif_angle, next_obs, next_angle)
        print("nx_dis = {}, nx_dif_angle = {}, nx_obs_dis = {}, nx_obs_angle".format(next_dis, next_dif_angle, next_obs, next_angle))
        print("x = {}, y = {}, z = {}, w = {}".format(distance - next_dis, dif_angle-next_dif_angle,obstacle_distance-next_obs, obstacle_angle-next_angle))

        _, goal = env.goal_check(next_dis)

        if goal == True and step_finish == False:

            new_goal_pos = random.randrange(0, 8)
            new_goal_num = env.re_goal_pos(goal_num, new_goal_pos)
            goal_x, goal_y = env.goal_position(new_goal_num)
            goal == False

        _, collide = env.collision(next_obs)

        if collide == True:
            env.prepare_episode()

        goal_distance_reward = env.goal_dis_reward(distance, next_dis)
        angle_reward = env.goal_angle_reward(robot_angle, yaw)
        obstacle_distance_reward = env.obs_dis_reward(obstacle_distance, next_obs)
        goal_reward, _ = env.goal_check(next_dis)
        collision_reward, _ = env.collision(next_obs)
        nerd_reward, nerd = env.nerd(len(action_lst), car_x, car_y)
        if nerd == True:
            action_lst = []

        #print("ANGLE: {}, YAW: {}, DIF: {}".format(next_robot_angle, next_yaw, next_dif_angle))

        total_reward = goal_distance_reward*angle_reward + obstacle_distance_reward + goal_reward + collision_reward + nerd_reward
        #print("goal_dis_reward= {}, goal_angle_reward = {}, goal_reward = {}, obstacle_reward ={}, collision_reward = {}, nerd_reward ={}".format(goal_distance_reward,angle_reward, goal_reward, obstacle_distance_reward, collision_reward, nerd_reward))

        score += total_reward
        step += 1

        done = env.done_mask(step_finish, collide)
        done_num = 0.0 if done else 1.0

        memory.put((state, select_action, reward*0.01, state_prime, done_num))

        if score > best_score:
            best_score = score
            best_score_lst.append(best_score)
            path = '/home/jaehun/DQN_network'
            torch.save(q, path + 'model.pt') 

        step_finish = env.end_episode(step)

        if step_finish == True:
            print("train start")
            score_lst.append(score)
            q.train(q, q_target, memory, optimizer, batch_size)

            break
            
    plt.plot(episode_lst, score_lst)
    print("# of episode :{}, score : {:.1f}".format(n_epi+1, score))

