import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()
        # TODO YOUR CODE HERE FOR INITIALIZING THE MODEL
        # following piazza advice for 2 hidden layers of 64 neurons
        self.hd1 = nn.Linear(state_size,64)
        self.hd2 = nn.Linear(64,64)
        self.output = nn.Linear(64,action_size)


    def forward(self, x):
        # TODO YOUR CODE HERE FOR THE FORWARD PASS
        
        # pass through network
        x = F.relu(self.hd1(x.float()))
        x = F.relu(self.hd2(x))
        x = self.output(x)
        return x
        raise NotImplementedError()

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
