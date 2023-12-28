import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    
    def __init__(self,inputsize) :
        super().__init__()

        self.relu_stack = nn.Sequential(
            nn.Linear(inputsize,1000),
            nn.ReLU(),
            nn.Linear(1000,4)
            )
    def forward(self,x):
        out = self.relu_stack(x)
        return out


