import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Func
import os


# linear deep q-learning class
class Linear_QNet(nn.Module):
    # initializer
    def __init__(self, input_layer_length, hidden_layer_length, output_layer_length):
        # this model is a feedfoward neural network model with input, 1 hidden and output layer
        super().__init__()
        # sets values for layer 1 and 2
        self.linear_layer_1 = nn.Linear(input_layer_length, hidden_layer_length)
        self.linear_layer_2 = nn.Linear(hidden_layer_length, output_layer_length)

    # input: x - tensor
    # this is the prediction function
    def forward(self, x):
        # activation function relu is taken from the functional module
        # layer one is passed as a parameter
        x = Func.relu(self.linear_layer_1(x))
        # second layer os applied. No activation function is used
        x = self.linear_layer_2(x)
        return x

    # helper function to save the model
    def save_model(self, file_name='saved_model.pth'):
        # if the saved_model folder does not exist
        if not os.path.exists('./saved_model'):
            # making a saved_model directory
            os.makedirs('./saved_model')
        # model is saved
        torch.save(self.state_dict(), os.path.join('./saved_model', file_name))


# to do the actual training and optimization, QTrainer class is implemented
class QTrainer:
    # initializer
    # input: model, learning rate, discount_rate
    # discount rate must be smaller than 1
    def __init__(self, model, learning_rate, discount_rate):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.model = model
        # pytorch's optimizer Adam is used
        self.optim = optim.Adam(model.parameters(), lr=self.learning_rate)
        # loss function
        self.criterion = nn.MSELoss()

    def train_step(self, current_state, action, reward, new_state, game_over):
        # each of these values is converted to a pytorch tensor
        current_state = torch.tensor(current_state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # if the length of the tensor os only one
        if len(current_state.shape) == 1:
            # It is changed to the form (1,x)
            # If there are multiple values (i.e., length is greater than 1),
            # it is of the form (n,x)
            current_state = torch.unsqueeze(current_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # Bellman equation is implemented
        # get predicted q value with the current state
        predicted = self.model(current_state)
        
        target = predicted.clone()
        # this has to be iterated with the length of tensor
        for index in range(len(game_over)):
            # if the game is over, we just take the reward
            Q_new = reward[index]
            # if the game is not done
            if not game_over[index]:
                 # new q = reward + (discount_rate * max(next_predicted Q value))
                Q_new = reward[index] + self.discount_rate * torch.max(self.model(new_state[index]))
            # to set the target as the maximum value of the action to new Q
            target[index][torch.argmax(action[index]).item()] = Q_new

        # loss function needs to be applied
        # zero_grad function empties the gradient
        self.optim.zero_grad()
        # target = Q new, predicted = Q
        loss = self.criterion(target, predicted)
        # backpropogation is used to update the gradient
        loss.backward()
        self.optim.step()