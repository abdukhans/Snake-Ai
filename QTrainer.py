import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from DQN import DQN
from RepalyMemory import ReplayMemory

class QTrainer:

    def __init__(self, model:DQN, memory:ReplayMemory,n_actions:int,n_obs:int,
     lr=1e-3,batch_size=10_000,discount=0.999) -> None:
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr,amsgrad=True)
        self.targetModel = DQN(n_obs,n_actions)
        self.targetModel.load_state_dict(self.model.state_dict())


        self.batch_size = batch_size
        self.memory = memory
        self.discount = discount 

        self.loss = None


        pass


    def optimize(self):
        if len(self.memory) < self.batch_size:
            return -1


        # print("OPTIM")


        state,action,next_state,reward =  zip(*self.memory.memory )

        # print(state)
        # print(action)
        # print(next_state)
        # print(reward)

        # This is the hides all final next_states 
        mask_non_final_next = torch.Tensor([s is not None for s in next_state ]).bool()


        non_final_next_states = torch.cat([s for s in next_state if s is not None])


        state = torch.cat(state)
        action = torch.cat(action)
        reward = torch.cat(reward)

        # print(action)
        # print(reward)


    

        # This variable captures all the Q(s,a) from the taken  state, action variables  
        qsa_current:torch.Tensor = self.model(state).gather(1,action)

        expected_qsa = torch.zeros(self.batch_size)
        with torch.no_grad():


            #print(len(non_final_next_states),len(reward))


            expected_qsa[mask_non_final_next]:torch.Tensor= reward +  self.discount*(self.targetModel(non_final_next_states).max(1).values)
 




        cirterion = nn.SmoothL1Loss()
        
        
        loss = cirterion(qsa_current,expected_qsa)


        loss_val = loss.item()

        #print("LOSS: " , loss.item())
        

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()



        return loss_val
    
    def push(self,state,action,next_state,reward):

        self.memory.push(state,action,next_state,reward)