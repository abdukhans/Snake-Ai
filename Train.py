from SnakeEnv import SnakeEnv
from os.path import exists


from Agent import Agent
from RepalyMemory import ReplayMemory
import matplotlib.pyplot as plt
import torch

GAMMA = 0.99
NUM_EPSIODES = 200
TAU = 0.005
BATCH_SIZE = 1000
MAX_MEMORY = 10_000
memory = ReplayMemory(MAX_MEMORY)
N_OBSV = 12
N_ACTIONS = 4
AI = Agent(N_OBSV,N_ACTIONS,memory,batch_size=BATCH_SIZE,lr=0.001)
PATH = "Weights.txt"


def isLoadable(state_dict_load,model_state_dict):
    for key in state_dict_load:

        if not(state_dict_load[key].size() == model_state_dict[key].size()):
            return False
        
    return True
    

if exists(PATH):

    dict_ = torch.load(PATH)


    if isLoadable(dict_,AI.model.state_dict()):
        AI.model.load_state_dict(dict_)
    else:
        print(f"Not correct size from {PATH} going to build a new model and save to {PATH}")

env = SnakeEnv(render=True,grid_size=20,epsiodeLen=100)
loss_val = [0]*NUM_EPSIODES



action_dict = {0:"UP",1:"DOWN",2:"RIGHT",3:"LEFT"}

eps_dur = []
exp_rew = []
score_list = []


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



def plot_score(avg_pool=10):
    plt.figure(4)

    plt.title("Score vs Episode")


    plt.xlabel("Epsiode")
    plt.ylabel("Score")
    

    plt.plot(score_list)

    if len(score_list) >= avg_pool:

        means = torch.Tensor(score_list).unfold(0,avg_pool,1).mean(1)

        plt.plot(list(range(avg_pool-1,NUM_EPSIODES)),means.numpy() ) 


    plt.show()


def plot_score_anim(avg_pool = 10):
    plt.figure(5)


    plt.plot(score_list)

    if len(score_list) >= avg_pool:

        means = torch.Tensor(score_list).unfold(0,avg_pool,1).mean(1)

        plt.plot(list(range(avg_pool-1,len(score_list))),means.numpy() ) 



    plt.show(block=False)
    plt.pause(0.0001)
    plt.clf()



for i in range(NUM_EPSIODES): 
    done = False
    sum_loss = 0 

    tot_time = 0

    discounted_tot_reward = 0 

    score = 0 
    while not(done):
        tot_time += 1 

        t = env.frame_iteration 

        state = AI.getGameState(env).unsqueeze(0) # [[_,_,....,_]] <--- state tensor is a 2D array with containing
                                                  # one 1D array that has 12 entries  

        action_idx = AI.SelectAction(state)       # ACTION <-- tensor which is going to be 1 by 1 NOTE: THis is a long tensor
        action = action_dict[action_idx.item()]
        
        reward, done = env.playMove(action)

        discounted_tot_reward  = reward.item() + GAMMA*discounted_tot_reward

        if reward.item() == 10:
            score += 1







       


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
            score_list.append(score)

        env.render(AI.model.state_dict())


    if tot_time > 0 :
        loss_val[i] = sum_loss/tot_time
        exp_rew.append(discounted_tot_reward)
    else:
        loss_val[i] = -1
        


    if loss_val[i] != -1:
        print(f"Epsiode #:{i}  Avg Loss: {loss_val[i]:.4f} Exp Rewards: {discounted_tot_reward:.4f}")





    memory.push(state,action_idx,next_state,reward)    


torch.save(AI.model.state_dict(), PATH)
plot_score()

# plot_rewards()
# plot_loss()
# plot_dur()
