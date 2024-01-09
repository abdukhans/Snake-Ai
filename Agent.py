from SnakeEnv import SnakeEnv 
from DQN import DQN
from RepalyMemory import ReplayMemory
from QTrainer import QTrainer
import numpy as np
import random 
import torch
import math
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 0.5
BATCH_SIZE = 10_000


class Agent:
    def __init__(self,n_obsv:int,n_actions:int,replayMemory:ReplayMemory,eps_start=EPS_START,
                eps_end=EPS_END, eps_decay=EPS_DECAY,batch_size=BATCH_SIZE):

        # This takes a vector of szie 11 and outputs a vector of size 4
        self.model = DQN(n_obsv,n_actions)
        self.batch_size = batch_size

        self.trainer = QTrainer(self.model,replayMemory,n_actions,n_obsv,batch_size=self.batch_size,lr=0.5)

        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0


        pass

    def getGameState(self,game:SnakeEnv):

        

        if game.done:

            return [0]*12
    
        snake_pos = game.snake_pos
        grid_size = game.grid_size

        point_left  = snake_pos + np.array([grid_size,0])
        point_right = snake_pos - np.array([grid_size,0])
        point_down  = snake_pos + np.array([0,grid_size])
        point_up    = snake_pos - np.array([0,grid_size])

        dir_left  = game.dir == 'LEFT'
        dir_right = game.dir == 'RIGHT'
        dir_up    = game.dir == 'UP'
        dir_down  = game.dir == 'DOWN'

        food_pos = game.food_pos

        state = [
            # Danger Right  
            game.isCollision(point_right),

            # Danger Left
            game.isCollision(point_left),

            # Danger Up
            game.isCollision(point_up),

            # Danger Down
            game.isCollision(point_down),


            # Move dir 
            dir_left,
            dir_right,
            dir_up,
            dir_down,



            # Food location
            food_pos[0] >= snake_pos[0],
            food_pos[0] < snake_pos[0],
            food_pos[1] >= snake_pos[1],
            food_pos[1] <  snake_pos[1]
            

        ]
        state_torch = torch.Tensor(state)
        return state_torch
    def SelectAction(self,state):


        eps_threshold = self.eps_end + (self.eps_start - self.eps_end)*(    math.exp(-1*self.steps_done/self.eps_decay)   )

        # state = self.getGameState(game)

        eps  = random.random()
        self.steps_done += 1 
        if  eps > eps_threshold:

            with torch.no_grad():
                # Note the shape of self.model(state) should be (4,)

                # NOTE: THIS RETURNS A LongTensor (This might shoot us in the foot later on) that is a 2D array with
                #       DIMS 1 by 1
                return self.model(state).max(1).indices.view(1,1)

        else:
            return torch.tensor([[random.randint(0,3)]],dtype=torch.long)
        



    def optimize(self):

        self.trainer.optimize()

    def pushRPM(self,state,action,next_state,reward):


        self.trainer.push(state,action,next_state,reward)
        pass

  