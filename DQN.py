import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
    
        self.relu_stack = nn.Sequential(
             nn.Linear(n_observations, 128),
             nn.ReLU(),
             nn.Linear(128, 128),
             nn.ReLU(),
             nn.Linear(128,n_actions)
        )

    def forward(self, x):
        out = self.relu_stack(x)

        return out