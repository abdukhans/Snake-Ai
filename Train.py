from SnakeEnv import SnakeEnv
from Agent import Agent
from RepalyMemory import ReplayMemory
import matplotlib.pyplot as plt
import torch

GAMMA = 0.99
NUM_EPSIODES = 50
TAU = 0.005
BATCH_SIZE = 100
memory = ReplayMemory(BATCH_SIZE)
AI = Agent(12,4,memory,batch_size=BATCH_SIZE,lr=0.5)
env = SnakeEnv(render=True)
loss_val = [0]*NUM_EPSIODES

action_dict = {0:"UP",1:"DOWN",2:"RIGHT",3:"LEFT"}

eps_dur = []
exp_rew = []


def plot_dur_anim(avg_pool=100):
    plt.figure(1)



    plt.plot(eps_dur)


    if len(eps_dur) >= avg_pool:


        means =  torch.Tensor(eps_dur).unfold(0,avg_pool,1).mean(1)

        #print(means)

        means = torch.cat((torch.Tensor(avg_pool - 1), means ))

        plt.plot(means.numpy())



    plt.show(block=False)

    plt.pause(0.0001)
    plt.clf()



    pass

def plot_dur(avg_pool=100):
    plt.figure(1)



    plt.plot(eps_dur)


    if len(eps_dur) >= avg_pool:


        means =  torch.Tensor(eps_dur).unfold(0,avg_pool,1).mean(1)

        #print(means)

        means = torch.cat((torch.Tensor(avg_pool - 1), means ))

        plt.plot(means.numpy())



    plt.show()


def plot_loss():
    plt.figure(2)

    plt.title("Avg Loss vs Episode")


    plt.xlabel("Epsiode #")
    plt.ylabel("Avg  loss")
    

    plt.plot([i for i in loss_val if i != -1])

    plt.show()






def plot_rewards():
    plt.figure(3)

    plt.title("Discounted rewards vs Episode")


    plt.xlabel("Epsiode #")
    plt.ylabel("Discounted rewards")
    

    plt.plot(exp_rew)

    plt.show()





for i in range(NUM_EPSIODES): 


    
    done = False



    sum_loss = 0 

    tot_time = 0

    discounted_tot_reward = 0 

    while not(done):
        tot_time += 1 

        t = env.frame_iteration 

        state = AI.getGameState(env).unsqueeze(0) # [[_,_,....,_]] <--- state tensor is a 2D array with containing
                                                  # one 1D array that has 11 entries  

        action_idx = AI.SelectAction(state)       # ACTION <-- tensor which is going to be 1 by 1 NOTE: THis is a long tensor
        action = action_dict[action_idx.item()]
        
        reward, done = env.playMove(action)

        discounted_tot_reward  = reward + GAMMA*discounted_tot_reward
        


       


        #print(env.frame_iteration)

        # if done:
        #     next_state = None
        # else:
        next_state = AI.getGameState(env).unsqueeze(0)      # [[_,_,....,_]] <--- next_state tensor is a 2D array with containing
                                                                # one 1D array that has 11 entries  

        memory.push(state,action_idx,next_state,reward)    


        loss  = AI.optimize() 
         
        
        if loss == -1:
            tot_time -=1 
        else:
            sum_loss += loss

        


        target_net_state = AI.trainer.targetModel.state_dict()
        policy_net_state = AI.model.state_dict()

        #print(target_net_state.keys())

        for key in policy_net_state:
            target_net_state[key] =  policy_net_state[key]*TAU + target_net_state[key]*(1-TAU)


        AI.trainer.targetModel.load_state_dict(target_net_state)



        if done:



            eps_dur.append(t+1) 



    if tot_time > 0 :
        loss_val[i] = sum_loss/tot_time
        exp_rew.append(discounted_tot_reward)
    else:
        loss_val[i] = -1
        


    if loss_val[i] != -1:
        print(f"Epsiode #:{i}  Avg Loss: {loss_val[i]}")





    memory.push(state,action_idx,next_state,reward)    





    
    


    pass



plot_rewards()
# plot_loss()
# plot_dur()
