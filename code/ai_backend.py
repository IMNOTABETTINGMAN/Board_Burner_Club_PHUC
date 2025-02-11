import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from collections import deque
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# DEVICE = "cpu"

MODEL_FOLDER_PATH = './models'
NN_FILE_NAME = 'phuc_ball.pth'
CAM_NN_FILE_NAME = 'phuc_cam_model.pth'
MAX_MEMORY = 10000
BATCH_SIZE = 4000
RECURRENT_STATES = 10
LR = 0.0001
EPSILON = 0
GAMMA = 0.9

# torch.set_flush_denormal(True)

# model to train camera date.  Fed into state of Linear_Model
class Conv2DModel(nn.Module):
    # input_size 20x20 image
    # output_size
    def __init__(self, input_size, output_size):
        super().__init__()
        self.classes = ['target_ball', 'bad_ball', 'box', 'wall', 'floor', 'background']
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(5,5,3)
        self.fc1 = nn.Linear(5*6*6, 200)
        self.fc2 = nn.Linear(200, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 5*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def load(self, file_name=CAM_NN_FILE_NAME):
        pass

    def save(self, file_name=CAM_NN_FILE_NAME):
        pass

# model for the PHUC state, action, reward
class LinearModelOneHidden(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).to(dtype=torch.float)
        self.linear2 = nn.Linear(hidden_size, output_size).to(dtype=torch.float)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def load(self, file_name=NN_FILE_NAME):
        model_folder_path = MODEL_FOLDER_PATH
        if os.path.exists(model_folder_path + '/' + file_name):
            pass
            # torch.load(self.state_dict(),file_name)

    def save(self, file_name=NN_FILE_NAME):
        model_folder_path = MODEL_FOLDER_PATH
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class LinearModelTwoHidden(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1).to(dtype=torch.float)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2).to(dtype=torch.float)
        self.linear3 = nn.Linear(hidden_size_2, output_size).to(dtype=torch.float)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def load(self, file_name=NN_FILE_NAME):
        model_folder_path = MODEL_FOLDER_PATH
        if os.path.exists(model_folder_path + '/' + file_name):
            pass
            # torch.load(self.state_dict(),file_name)

    def save(self, file_name=NN_FILE_NAME):
        model_folder_path = MODEL_FOLDER_PATH
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class PHUCBallModel:
    def __init__(self, phuc, keyboard, game_controller, training):
        self.phuc = phuc
        self.keyboard = keyboard
        self.game_controller = game_controller

        # self.model = linear_model(9,256, 5)
        # self.input_size = self.phuc.camera_width*self.phuc.camera_height_modified*3 + 9
        self.input_size = int(self.phuc.world.perception_field_len+9)
        self.model = LinearModelOneHidden(self.input_size,500,400).to(device)
        # training = True or False
        self.training = training

        self.n_games = 0

        self.final_move = [0.0, 0.0]

        self.lr = LR
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()

    # Main loop for running neural network
    def update(self):
        if self.training:
            old_state = self.phuc.get_state()
            # print(old_state)
            final_move = self.get_action(old_state)
            # print("final move: " + str(final_move))
            self.phuc.update()
            new_state = self.phuc.get_state()
            reward = self.phuc.world.determine_reward()
            # if PHUC accumulated 100 points or hits a wall done = True
            done = self.phuc.world.episode_finished()
            self.train_short_memory(old_state, final_move, reward, new_state, done)

            # remember
            self.remember(old_state, final_move, reward, new_state, done)

            if done:
                # train long memory, plot result
                self.phuc.reset()
                self.phuc.update()
                self.n_games += 1
                self.train_long_memory()
                self.phuc.start()
                if self.phuc.world.points > self.phuc.world.record:
                    self.phuc.world.record = self.phuc.world.score
                    # self.save()
                print('Game', self.n_games, 'Score', self.phuc.world.points, 'Record:', self.phuc.world.record)
        else:
            state = self.phuc.get_state()
            self.get_action(state)

    def train_step(self, state, action, reward, next_state, done):
        # (n, x)
        state = torch.tensor(state).to(dtype=torch.float, device=device)
        next_state = torch.tensor(next_state).to(dtype=torch.float, device=device)
        action = torch.tensor(action).to(dtype=torch.float, device=device)
        reward = torch.tensor(reward).to(dtype=torch.float, device=device)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        # pred = self.model(state).to(device)
        pred = self.model(state)

        # target = pred.clone().to(device)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = [reward[idx][0], reward[idx][1]]
            if done[idx] < 0.5:
                Q_new[0] = reward[idx][0] + self.gamma * torch.max(self.model(next_state[idx])[:200])
                Q_new[1] = reward[idx][1] + self.gamma * torch.max(self.model(next_state[idx])[200:])
            # reward the primary wheel setting
            lw_index = 100 + int(action[idx][0]*100)
            rw_index = 300 + int(action[idx][1]*100)
            # target[idx][lw_index] = Q_new[0]
            # target[idx][rw_index] = Q_new[1]
            for i in range(5):
                if (lw_index - i) >= 0:
                    target[idx][lw_index - i] = Q_new[0] * (100 -i*2) /100
                if (lw_index + i) < 200:
                    target[idx][lw_index + i] = Q_new[0] * (100 -i*2) /100
                if (rw_index - i) >= 200:
                    target[idx][rw_index - i] = Q_new[1] * (100 -i*2) /100
                if (rw_index + i) < 400:
                    target[idx][rw_index + i] = Q_new[1] * (100 -i*2) /100


            # target[idx][99 + torch.argmax(action[idx][0]).item() * 100] = Q_new[0]
            # target[idx][299 + torch.argmax(action[idx][1]).item() * 100] = Q_new[1]

        #target = pred.clone()
        #Q_new = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(DEVICE)
        #for idx in range(len(done)):
        #    if not done[idx]:
        #        next_pred = self.model(state[idx])
                # next_pred = self.model(next_state[idx])
                # next_pred.cpu()
        #        Q_new[0] = reward[idx][0]*(1-self.gamma)*action[idx][0] + (1-abs(reward[idx][0]))*self.gamma*next_pred[0]
        #        Q_new[1] = reward[idx][0]*(1-self.gamma)*action[idx][1] + (1-abs(reward[idx][0]))*self.gamma*next_pred[1]
        #        Q_new[2] = reward[idx][1]*(1-self.gamma)*action[idx][2] + (1-abs(reward[idx][1]))*self.gamma*next_pred[2]
        #        Q_new[3] = reward[idx][1]*(1-self.gamma)*action[idx][3] + (1-abs(reward[idx][1]))*self.gamma*next_pred[3]
        #        Q_new[4] = reward[idx][1]*(1-self.gamma)*action[idx][4] + (1-abs(reward[idx][1]))*self.gamma*next_pred[4]
                #Q_new[0:2] = reward[idx][0]*(1-self.gamma)*action[0:2] + (1-reward[idx][0])*self.gamma*next_pred[0:2]
                #Q_new[2:] = reward[idx][1]*(1-self.gamma)*action[2:] + (1-reward[idx][1])*self.gamma*next_pred[2:]
                #Q_new[0:2] = torch.tensor([reward[idx][0], dtype=torch.float) + self.gamma * self.model(next_state[idx]).cpu()
        #    else:
        #        Q_new = torch.tensor([reward[idx][0]*action[idx][0], reward[idx][0]*action[idx][1],
        #                              reward[idx][1]*action[idx][2], reward[idx][1]*action[idx][3],
        #                              reward[idx][1] * action[idx][4]], dtype=torch.float).to(DEVICE)


        #    target[idx] = Q_new

        # target.to(device)
        self.optimizer.zero_grad()
        # loss = self.criterion(target, pred).to(device)
        # loss = self.criterion(pred, target).to(device)
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimizer.step()

        #if device == torch.device("cuda"):
        #    torch.cuda.empty_cache()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        #self.memory_[self.memory_index] = (state, action, reward, next_state, done)
        #self.memory_index += 1
        #if self.memory_index > MAX_MEMORY_ - 1:
        #    self.memory_index = 0
        #    self.memory_full = True
        # print(self.memory_index)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        #if self.memory_full:
        #    mini_sample = random.sample(self.memory_, BATCH_SIZE)  # list of tuples
            #print("filled memory")
            #if len(self.memory_) > BATCH_SIZE:
            #    mini_sample = random.sample(self.memory_, BATCH_SIZE)  # list of tuples
            #else:
            #    mini_sample = self.memory_
        #else:
        #    if self.memory_index > BATCH_SIZE:
                # print("unfilled memory larger than batch")
        #        mini_sample = random.sample(self.memory_[:self.memory_index], BATCH_SIZE)  # list of tuples
        #    else:
                # print("unfilled memory smaller than batch")
        #        mini_sample = self.memory_[:self.memory_index]

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.training:
            self.epsilon = 100 - self.n_games

            # user game_controller override
            if self.game_controller.right_trigger > -0.95:
                self.final_move[0] = self.phuc.left_wheel_speed
                self.final_move[1] = self.phuc.right_wheel_speed
                # print('cont move: ' + str(self.final_move))
            elif self.keyboard.forward or self.keyboard.reverse or self.keyboard.turn_left or self.keyboard.turn_right:
                self.final_move[0] = self.keyboard.left_wheel_speed
                self.final_move[1] = self.keyboard.right_wheel_speed
                self.phuc.set_wheel_speed('lw', self.final_move[0])
                self.phuc.set_wheel_speed('rw', self.final_move[1])
                # print('cont move: ' + str(self.final_move))
            elif random.randint(0, 250) < self.epsilon:
                lw_move = np.random.normal(0.3, 0.2, 1).tolist()[0]
                if lw_move > 0.99:
                    lw_move = 0.99
                elif lw_move < -1.0:
                    lw_move = -1.0
                self.final_move[0] = lw_move
                self.phuc.set_wheel_speed('lw', lw_move)

                rw_move = np.random.normal(0.3, 0.2, 1).tolist()[0]
                if rw_move > 0.99:
                    rw_move = 0.99
                elif rw_move < -1.0:
                    rw_move = -1.0
                self.final_move[1] = rw_move
                self.phuc.set_wheel_speed('rw', rw_move)
                # print('rand move: ' + str(self.final_move))
            else:
                state0 = torch.tensor(state).to(dtype=torch.float, device=device)
                # This is where the neural network produces output
                output = self.model(state0).to(dtype=torch.float, device=device)
                prediction_lw = torch.argmax(output[:200]).item()
                prediction_rw = torch.argmax(output[200:]).item()
                self.final_move[0] = (prediction_lw - 100)/100
                self.final_move[1] = (prediction_rw - 100)/100
                self.phuc.set_wheel_speed('lw', self.final_move[0])
                self.phuc.set_wheel_speed('rw', self.final_move[1])
                # print("prediction lw: " + str(prediction_lw) + " rw: " + str(prediction_rw))
                # print('nnet move: ' + str(self.final_move))
        else:
            state0 = torch.tensor(state).to(dtype=torch.float, device=device)
            # This is where the neural network produces output
            with torch.no_grad():
                output = self.model(state0).to(dtype=torch.float, device=device)
            prediction_lw = torch.argmax(output[:200]).item()
            prediction_rw = torch.argmax(output[200:]).item()
            self.final_move[0] = (prediction_lw - 100) / 100
            self.final_move[1] = (prediction_rw - 100) / 100
            self.phuc.set_wheel_speed('lw', self.final_move[0])
            self.phuc.set_wheel_speed('rw', self.final_move[1])
        return self.final_move

class PHUCBallModelMemory:
    def __init__(self, phuc, keyboard, game_controller, training):
        self.phuc = phuc
        self.keyboard = keyboard
        self.game_controller = game_controller

        # self.model = linear_model(9,256, 5)
        # self.input_size = self.phuc.camera_width*self.phuc.camera_height_modified*3 + 9
        self.input_size = int((self.phuc.world.perception_field_len+9)*RECURRENT_STATES)
        self.model = LinearModelTwoHidden(self.input_size,600,600,400).to(device)
        # training = True or False
        self.training = training

        self.n_games = 0

        self.final_move = [0.0, 0.0]

        self.lr = LR
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.recurrent_perception = deque(maxlen=RECURRENT_STATES)
        for i in range(RECURRENT_STATES):
            self.recurrent_perception.append(np.zeros(self.phuc.world.perception_field_len+9, dtype=np.float32).tolist())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.prediction = []

    # Main loop for running neural network
    def update(self):
        if self.training:
            old_state = self.phuc.get_state()
            _old_state = self.update_input_array(old_state)
            # print(_old_state)
            final_move = self.get_action(_old_state)
            # print("final move: " + str(final_move))
            self.phuc.update()
            new_state = self.phuc.get_state()
            _new_state = self.update_input_array(new_state)
            reward = self.phuc.world.determine_reward()
            # if PHUC accumulated 100 points or hits a wall done = True
            done = self.phuc.world.episode_finished()
            self.train_short_memory(_old_state, final_move, reward, _new_state, done)

            # remember
            self.remember(_old_state, final_move, reward, _new_state, done)

            if done:
                # train long memory, plot result
                self.phuc.reset()
                self.phuc.update()
                self.n_games += 1
                self.train_long_memory()
                self.reset_input_array()
                self.phuc.start()
                if self.phuc.world.points > self.phuc.world.record:
                    self.phuc.world.record = self.phuc.world.score
                    # self.save()
                print('Game', self.n_games, 'Score', self.phuc.world.points, 'Record:', self.phuc.world.record)
        else:
            state = self.phuc.get_state()
            _state = self.update_input_array(state)
            self.get_action(_state)

    def train_step(self, state, action, reward, next_state, done):
        # (n, x)
        state = torch.tensor(state).to(dtype=torch.float, device=device)
        action = torch.tensor(action).to(dtype=torch.float, device=device)
        reward = torch.tensor(reward).to(dtype=torch.float, device=device)
        next_state = torch.tensor(next_state).to(dtype=torch.float, device=device)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        # pred = self.model(state).to(device)
        self.prediction = self.model(state)

        # target = pred.clone().to(device)
        target = self.prediction.clone()
        for idx in range(len(done)):
            Q_new = [reward[idx][0], reward[idx][1]]
            if done[idx] < 0.5:
                Q_new[0] = reward[idx][0] + self.gamma * torch.max(self.model(next_state[idx])[:200])
                Q_new[1] = reward[idx][1] + self.gamma * torch.max(self.model(next_state[idx])[200:])
            # reward the primary wheel setting
            lw_index = 100 + int(action[idx][0]*100)
            rw_index = 300 + int(action[idx][1]*100)
            # print("lw index: " + str(lw_index))
            # print("rw index: " + str(rw_index))
            target[idx][lw_index] = Q_new[0]
            target[idx][rw_index] = Q_new[1]
            for i in range(1,3):
                if (lw_index - i) >= 0:
                    target[idx][lw_index - i] = Q_new[0] * (100 -i*2) /100
                if (lw_index + i) < 200:
                    target[idx][lw_index + i] = Q_new[0] * (100 -i*2) /100
                if (rw_index - i) >= 200:
                    target[idx][rw_index - i] = Q_new[1] * (100 -i*2) /100
                if (rw_index + i) < 400:
                    target[idx][rw_index + i] = Q_new[1] * (100 -i*2) /100


            # target[idx][99 + torch.argmax(action[idx][0]).item() * 100] = Q_new[0]
            # target[idx][299 + torch.argmax(action[idx][1]).item() * 100] = Q_new[1]

        #target = pred.clone()
        #Q_new = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(DEVICE)
        #for idx in range(len(done)):
        #    if not done[idx]:
        #        next_pred = self.model(state[idx])
                # next_pred = self.model(next_state[idx])
                # next_pred.cpu()
        #        Q_new[0] = reward[idx][0]*(1-self.gamma)*action[idx][0] + (1-abs(reward[idx][0]))*self.gamma*next_pred[0]
        #        Q_new[1] = reward[idx][0]*(1-self.gamma)*action[idx][1] + (1-abs(reward[idx][0]))*self.gamma*next_pred[1]
        #        Q_new[2] = reward[idx][1]*(1-self.gamma)*action[idx][2] + (1-abs(reward[idx][1]))*self.gamma*next_pred[2]
        #        Q_new[3] = reward[idx][1]*(1-self.gamma)*action[idx][3] + (1-abs(reward[idx][1]))*self.gamma*next_pred[3]
        #        Q_new[4] = reward[idx][1]*(1-self.gamma)*action[idx][4] + (1-abs(reward[idx][1]))*self.gamma*next_pred[4]
                #Q_new[0:2] = reward[idx][0]*(1-self.gamma)*action[0:2] + (1-reward[idx][0])*self.gamma*next_pred[0:2]
                #Q_new[2:] = reward[idx][1]*(1-self.gamma)*action[2:] + (1-reward[idx][1])*self.gamma*next_pred[2:]
                #Q_new[0:2] = torch.tensor([reward[idx][0], dtype=torch.float) + self.gamma * self.model(next_state[idx]).cpu()
        #    else:
        #        Q_new = torch.tensor([reward[idx][0]*action[idx][0], reward[idx][0]*action[idx][1],
        #                              reward[idx][1]*action[idx][2], reward[idx][1]*action[idx][3],
        #                              reward[idx][1] * action[idx][4]], dtype=torch.float).to(DEVICE)


        #    target[idx] = Q_new

        # target.to(device)
        self.optimizer.zero_grad()
        # loss = self.criterion(target, pred).to(device)
        # loss = self.criterion(pred, target).to(device)
        loss = self.criterion(self.prediction, target)
        loss.backward()

        self.optimizer.step()

        #if device == torch.device("cuda"):
        #    torch.cuda.empty_cache()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        #self.memory_[self.memory_index] = (state, action, reward, next_state, done)
        #self.memory_index += 1
        #if self.memory_index > MAX_MEMORY_ - 1:
        #    self.memory_index = 0
        #    self.memory_full = True
        # print(self.memory_index)

    def update_input_array(self, array):
        self.recurrent_perception.append(array)
        flatten = np.array(self.recurrent_perception).flatten()
        # print(flatten[0:30])
        return np.array(flatten)

    def reset_input_array(self):
        for i in range(RECURRENT_STATES):
            self.recurrent_perception.append(np.zeros(self.phuc.world.perception_field_len+9, dtype=np.float32).tolist())

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        #if self.memory_full:
        #    mini_sample = random.sample(self.memory_, BATCH_SIZE)  # list of tuples
            #print("filled memory")
            #if len(self.memory_) > BATCH_SIZE:
            #    mini_sample = random.sample(self.memory_, BATCH_SIZE)  # list of tuples
            #else:
            #    mini_sample = self.memory_
        #else:
        #    if self.memory_index > BATCH_SIZE:
                # print("unfilled memory larger than batch")
        #        mini_sample = random.sample(self.memory_[:self.memory_index], BATCH_SIZE)  # list of tuples
        #    else:
                # print("unfilled memory smaller than batch")
        #        mini_sample = self.memory_[:self.memory_index]

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.training:
            self.epsilon = 100 - self.n_games

            # user game_controller override
            if self.game_controller.right_trigger > -0.95:
                self.final_move[0] = self.phuc.left_wheel_speed
                self.final_move[1] = self.phuc.right_wheel_speed
                # print('cont move: ' + str(self.final_move))
            elif self.keyboard.forward or self.keyboard.reverse or self.keyboard.turn_left or self.keyboard.turn_right:
                self.final_move[0] = self.keyboard.left_wheel_speed
                self.final_move[1] = self.keyboard.right_wheel_speed
                self.phuc.set_wheel_speed('lw', self.final_move[0])
                self.phuc.set_wheel_speed('rw', self.final_move[1])
                # print('cont move: ' + str(self.final_move))
            elif random.randint(0, 250) < self.epsilon:
                lw_move = np.random.normal(0.3, 0.2, 1).tolist()[0]
                if lw_move > 0.99:
                    lw_move = 0.99
                elif lw_move < -1.0:
                    lw_move = -1.0
                self.final_move[0] = lw_move
                self.phuc.set_wheel_speed('lw', lw_move)

                rw_move = np.random.normal(0.3, 0.2, 1).tolist()[0]
                if rw_move > 0.99:
                    rw_move = 0.99
                elif rw_move < -1.0:
                    rw_move = -1.0
                self.final_move[1] = rw_move
                self.phuc.set_wheel_speed('rw', rw_move)
                # print('rand move: ' + str(self.final_move))
            else:
                state0 = torch.tensor(state).to(dtype=torch.float, device=device)
                # This is where the neural network produces output
                output = self.model(state0).to(dtype=torch.float, device=device)
                prediction_lw = torch.argmax(output[:200]).item()
                prediction_rw = torch.argmax(output[200:]).item()
                # print('prediction lw: ' + str(prediction_lw))
                # print('prediction rw: ' + str(prediction_rw))
                self.final_move[0] = (prediction_lw - 100)/100
                self.final_move[1] = (prediction_rw - 100)/100
                self.phuc.set_wheel_speed('lw', self.final_move[0])
                self.phuc.set_wheel_speed('rw', self.final_move[1])
                # print("prediction lw: " + str(prediction_lw) + " rw: " + str(prediction_rw))
                # print('nnet move: ' + str(self.final_move))
        else:
            state0 = torch.tensor(state).to(dtype=torch.float, device=device)
            # This is where the neural network produces output
            with torch.no_grad():
                output = self.model(state0).to(dtype=torch.float, device=device)
            prediction_lw = torch.argmax(output[:200]).item()
            prediction_rw = torch.argmax(output[200:]).item()
            self.final_move[0] = (prediction_lw - 100) / 100
            self.final_move[1] = (prediction_rw - 100) / 100
            self.phuc.set_wheel_speed('lw', self.final_move[0])
            self.phuc.set_wheel_speed('rw', self.final_move[1])
        return self.final_move

class PHUCBallModelConvMemory:
    def __init__(self, phuc, keyboard, game_controller, training):
        self.phuc = phuc
        self.keyboard = keyboard
        self.game_controller = game_controller

        # self.model = linear_model(9,256, 5)
        # self.input_size = self.phuc.camera_width*self.phuc.camera_height_modified*3 + 9
        self.input_size = int((self.phuc.world.perception_field_len+9)*RECURRENT_STATES)
        self.vision_model = Conv2DModel(20, 4)
        self.phuc_model = LinearModelTwoHidden(self.input_size,500,500,400).to(device)
        # training = True or False
        self.training = training

        self.n_games = 0

        self.final_move = [0.0, 0.0]

        self.lr = LR
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.recurrent_perception = deque(maxlen=RECURRENT_STATES)
        for i in range(RECURRENT_STATES):
            self.recurrent_perception.append(np.zeros(self.phuc.world.perception_field_len+9, dtype=np.float32).tolist())

        self.optimizer_vision = optim.Adam(self.vision_model.parameters(), lr=self.lr)
        self.optimizer_net = optim.Adam(self.phuc_model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()

    # Main loop for running neural network
    def update(self):
        if self.training:
            old_state = self.phuc.get_state()
            _old_state = self.update_input_array(old_state)
            # print(_old_state)
            final_move = self.get_action(_old_state)
            # print("final move: " + str(final_move))
            self.phuc.update()
            new_state = self.phuc.get_state()
            _new_state = self.update_input_array(new_state)
            reward = self.phuc.world.determine_reward()
            # if PHUC accumulated 100 points or hits a wall done = True
            done = self.phuc.world.episode_finished()
            self.train_short_memory(_old_state, final_move, reward, _new_state, done)

            # remember
            self.remember(_old_state, final_move, reward, _new_state, done)

            if done:
                # train long memory, plot result
                self.phuc.reset()
                self.phuc.update()
                self.n_games += 1
                self.train_long_memory()
                self.reset_input_array()
                self.phuc.start()
                if self.phuc.world.points > self.phuc.world.record:
                    self.phuc.world.record = self.phuc.world.score
                    # self.save()
                print('Game', self.n_games, 'Score', self.phuc.world.points, 'Record:', self.phuc.world.record)
        else:
            state = self.phuc.get_state()
            _state = self.update_input_array(state)
            self.get_action(_state)

    def train_step(self, state, action, reward, next_state, done):
        # (n, x)
        state = torch.tensor(state).to(dtype=torch.float, device=device)
        next_state = torch.tensor(next_state).to(dtype=torch.float, device=device)
        action = torch.tensor(action).to(dtype=torch.float, device=device)
        reward = torch.tensor(reward).to(dtype=torch.float, device=device)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        # pred = self.model(state).to(device)
        pred = self.model(state)

        # target = pred.clone().to(device)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = [reward[idx][0], reward[idx][1]]
            if done[idx] < 0.5:
                Q_new[0] = reward[idx][0] + self.gamma * torch.max(self.model(next_state[idx])[:200])
                Q_new[1] = reward[idx][1] + self.gamma * torch.max(self.model(next_state[idx])[200:])
            # reward the primary wheel setting
            lw_index = 100 + int(action[idx][0]*100)
            rw_index = 300 + int(action[idx][1]*100)
            # target[idx][lw_index] = Q_new[0]
            # target[idx][rw_index] = Q_new[1]
            for i in range(5):
                if (lw_index - i) >= 0:
                    target[idx][lw_index - i] = Q_new[0] * (100 -i*2) /100
                if (lw_index + i) < 200:
                    target[idx][lw_index + i] = Q_new[0] * (100 -i*2) /100
                if (rw_index - i) >= 200:
                    target[idx][rw_index - i] = Q_new[1] * (100 -i*2) /100
                if (rw_index + i) < 400:
                    target[idx][rw_index + i] = Q_new[1] * (100 -i*2) /100


            # target[idx][99 + torch.argmax(action[idx][0]).item() * 100] = Q_new[0]
            # target[idx][299 + torch.argmax(action[idx][1]).item() * 100] = Q_new[1]

        #target = pred.clone()
        #Q_new = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(DEVICE)
        #for idx in range(len(done)):
        #    if not done[idx]:
        #        next_pred = self.model(state[idx])
                # next_pred = self.model(next_state[idx])
                # next_pred.cpu()
        #        Q_new[0] = reward[idx][0]*(1-self.gamma)*action[idx][0] + (1-abs(reward[idx][0]))*self.gamma*next_pred[0]
        #        Q_new[1] = reward[idx][0]*(1-self.gamma)*action[idx][1] + (1-abs(reward[idx][0]))*self.gamma*next_pred[1]
        #        Q_new[2] = reward[idx][1]*(1-self.gamma)*action[idx][2] + (1-abs(reward[idx][1]))*self.gamma*next_pred[2]
        #        Q_new[3] = reward[idx][1]*(1-self.gamma)*action[idx][3] + (1-abs(reward[idx][1]))*self.gamma*next_pred[3]
        #        Q_new[4] = reward[idx][1]*(1-self.gamma)*action[idx][4] + (1-abs(reward[idx][1]))*self.gamma*next_pred[4]
                #Q_new[0:2] = reward[idx][0]*(1-self.gamma)*action[0:2] + (1-reward[idx][0])*self.gamma*next_pred[0:2]
                #Q_new[2:] = reward[idx][1]*(1-self.gamma)*action[2:] + (1-reward[idx][1])*self.gamma*next_pred[2:]
                #Q_new[0:2] = torch.tensor([reward[idx][0], dtype=torch.float) + self.gamma * self.model(next_state[idx]).cpu()
        #    else:
        #        Q_new = torch.tensor([reward[idx][0]*action[idx][0], reward[idx][0]*action[idx][1],
        #                              reward[idx][1]*action[idx][2], reward[idx][1]*action[idx][3],
        #                              reward[idx][1] * action[idx][4]], dtype=torch.float).to(DEVICE)


        #    target[idx] = Q_new

        # target.to(device)
        self.optimizer.zero_grad()
        # loss = self.criterion(target, pred).to(device)
        # loss = self.criterion(pred, target).to(device)
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimizer.step()

        #if device == torch.device("cuda"):
        #    torch.cuda.empty_cache()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        #self.memory_[self.memory_index] = (state, action, reward, next_state, done)
        #self.memory_index += 1
        #if self.memory_index > MAX_MEMORY_ - 1:
        #    self.memory_index = 0
        #    self.memory_full = True
        # print(self.memory_index)

    def update_input_array(self, array):
        self.recurrent_perception.append(array)
        flatten = np.array(self.recurrent_perception).flatten()
        # print(flatten[0:30])
        return np.array(flatten)

    def reset_input_array(self):
        for i in range(RECURRENT_STATES):
            self.recurrent_perception.append(np.zeros(self.phuc.world.perception_field_len+9, dtype=np.float32).tolist())

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        #if self.memory_full:
        #    mini_sample = random.sample(self.memory_, BATCH_SIZE)  # list of tuples
            #print("filled memory")
            #if len(self.memory_) > BATCH_SIZE:
            #    mini_sample = random.sample(self.memory_, BATCH_SIZE)  # list of tuples
            #else:
            #    mini_sample = self.memory_
        #else:
        #    if self.memory_index > BATCH_SIZE:
                # print("unfilled memory larger than batch")
        #        mini_sample = random.sample(self.memory_[:self.memory_index], BATCH_SIZE)  # list of tuples
        #    else:
                # print("unfilled memory smaller than batch")
        #        mini_sample = self.memory_[:self.memory_index]

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.training:
            self.epsilon = 100 - self.n_games

            # user game_controller override
            if self.game_controller.right_trigger > -0.95:
                self.final_move[0] = self.phuc.left_wheel_speed
                self.final_move[1] = self.phuc.right_wheel_speed
                # print('cont move: ' + str(self.final_move))
            elif self.keyboard.forward or self.keyboard.reverse or self.keyboard.turn_left or self.keyboard.turn_right:
                self.final_move[0] = self.keyboard.left_wheel_speed
                self.final_move[1] = self.keyboard.right_wheel_speed
                self.phuc.set_wheel_speed('lw', self.final_move[0])
                self.phuc.set_wheel_speed('rw', self.final_move[1])
                # print('cont move: ' + str(self.final_move))
            elif random.randint(0, 250) < self.epsilon:
                lw_move = np.random.normal(0.3, 0.2, 1).tolist()[0]
                if lw_move > 0.99:
                    lw_move = 0.99
                elif lw_move < -1.0:
                    lw_move = -1.0
                self.final_move[0] = lw_move
                self.phuc.set_wheel_speed('lw', lw_move)

                rw_move = np.random.normal(0.3, 0.2, 1).tolist()[0]
                if rw_move > 0.99:
                    rw_move = 0.99
                elif rw_move < -1.0:
                    rw_move = -1.0
                self.final_move[1] = rw_move
                self.phuc.set_wheel_speed('rw', rw_move)
                # print('rand move: ' + str(self.final_move))
            else:
                state0 = torch.tensor(state).to(dtype=torch.float, device=device)
                # This is where the neural network produces output
                output = self.model(state0).to(dtype=torch.float, device=device)
                prediction_lw = torch.argmax(output[:200]).item()
                prediction_rw = torch.argmax(output[200:]).item()
                self.final_move[0] = (prediction_lw - 100)/100
                self.final_move[1] = (prediction_rw - 100)/100
                self.phuc.set_wheel_speed('lw', self.final_move[0])
                self.phuc.set_wheel_speed('rw', self.final_move[1])
                # print("prediction lw: " + str(prediction_lw) + " rw: " + str(prediction_rw))
                # print('nnet move: ' + str(self.final_move))
        else:
            state0 = torch.tensor(state).to(dtype=torch.float, device=device)
            # This is where the neural network produces output
            with torch.no_grad():
                output = self.model(state0).to(dtype=torch.float, device=device)
            prediction_lw = torch.argmax(output[:200]).item()
            prediction_rw = torch.argmax(output[200:]).item()
            self.final_move[0] = (prediction_lw - 100) / 100
            self.final_move[1] = (prediction_rw - 100) / 100
            self.phuc.set_wheel_speed('lw', self.final_move[0])
            self.phuc.set_wheel_speed('rw', self.final_move[1])
        return self.final_move
