import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pdb
import numpy as np
from PreProcessing import PreprocessData

# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)
InputSize = 6  # Input Size
batch_size = 1  # Batch Size Of Neural Network
NumClasses = 1  # Output Size

############################################# FOR STUDENTS #####################################
NumEpochs = 25
HiddenSize = 35

m= nn.Sigmoid()

# Create The Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): # Network constructor
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)# Define the first fully connected layer
        self.relu1 = nn.ReLU()# Define the ReLU activation function for the first layer
        self.fc2 = nn.Linear(hidden_size, num_classes) # Define the second fully connected layer
        self.relu2 = nn.ReLU()# Define the ReLU activation function for the second layer 
        self.fc3 = nn.Linear(input_size, hidden_size) # Define the third fully connected layer
        self.relu3 = nn.ReLU()# Define the ReLU activation function for the third layer
        self.m = nn.Sigmoid()# Define the sigmoid activation function for the final layer
        self.fc4 = nn.Linear(hidden_size, num_classes) # Define the fourth fully connected layer

    def forward(self, x):
        
        out = self.fc1(x) # Pass the input through the first fully connected layer
        out = self.relu1(out) # Apply the ReLU activation function for the first layer
        out = self.fc2(out) # Pass the result through the second fully connected layer
        out = self.relu2(out) # Apply the ReLU activation function for the second layer
        out = self.fc3(x)  # Pass the input through the third fully connected layer
        out = self.relu3(out) # Apply the ReLU activation function for the third layer
        out = self.m(out)  # Apply the sigmoid activation function for the final layer
        out = self.fc4(out) # Pass the result through the fourth fully connected layer

        return out

net = Net(InputSize,HiddenSize, NumClasses)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005)

##################################################################################################

if __name__ == "__main__":

    TrainSize, SensorNNData, SensorNNLabels = PreprocessData()
    for j in range(NumEpochs):
        losses = 0
        for i in range(TrainSize):
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        print('Epoch %d, Loss: %.4f' % (j + 1, losses / SensorNNData.shape[0]))
        torch.save(net.state_dict(), './SavedNets/NNBot.pkl')
