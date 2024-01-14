import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from DQN import DQN
from RepalyMemory import ReplayMemory


def allZeros(iter):

    for i in iter[0]:
        #print(i)
        if i != 0 :
            return False

    return True

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

        
        sample = self.memory.sample(self.batch_size)

        # print("OPTIM")


        state,action,next_state,reward =  zip(*sample )

        # print(state)
        # print(action)
        # print(next_state)
        # print(reward)

        # This is the hides all final next_states 
        mask_non_final_next = torch.tensor([not(allZeros(s)) for s in next_state ],dtype = torch.bool)


        non_final_next_states = torch.cat([s for s in next_state if s is not None])


        state = torch.cat(state)
        action = torch.cat(action)
        reward = torch.cat(reward)

        # print(action)
        # print(reward)


    

        # This variable captures all the Q(s,a) from the taken  state, action variables  
        qsa_current:torch.Tensor = self.model(state).gather(1,action)




        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():


            #print(len(non_final_next_states),len(reward))


            next_state_values[mask_non_final_next]:torch.Tensor = self.targetModel(non_final_next_states).max(1).values
 




        cirterion = nn.MSELoss()
        
        expected_qsa = (next_state_values* self.discount) + reward

        
        loss = cirterion(qsa_current,expected_qsa.unsqueeze(1))


        loss_val = loss.item()

        #print("LOSS: " , loss.item())
        

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()



        return loss_val
    
    def push(self,state,action,next_state,reward):

        self.memory.push(state,action,next_state,reward)