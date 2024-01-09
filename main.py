from SnakeEnv import SnakeEnv
from Agent import Agent
from RepalyMemory import ReplayMemory
NUM_EPSIODES = 500
TAU = 0.005
BATCH_SIZE = 100
memory = ReplayMemory(BATCH_SIZE)
AI = Agent(12,4,memory,batch_size=BATCH_SIZE)
env = SnakeEnv(render=True)

action_dict = {0:"UP",1:"DOWN",2:"RIGHT",3:"LEFT"}

for i in range(NUM_EPSIODES): 


    
    done = False

    print(i)
    while not(done):

        state = AI.getGameState(env).unsqueeze(0) # [[_,_,....,_]] <--- state tensor is a 2D array with containing
                                                  # one 1D array that has 11 entries  

        action_idx = AI.SelectAction(state)       # ACTION <-- tensor which is going to be 1 by 1 NOTE: THis is a long tensor
        action = action_dict[action_idx.item()]
        
        reward, done = env.playMove(action)

        # if done:
        #     next_state = None
        # else:
        next_state = AI.getGameState(env).unsqueeze(0)      # [[_,_,....,_]] <--- next_state tensor is a 2D array with containing
                                                                # one 1D array that has 11 entries  

        memory.push(state,action_idx,next_state,reward)    

        AI.optimize()


        


        target_net_state = AI.trainer.targetModel.state_dict()
        policy_net_state = AI.model.state_dict()

        #print(target_net_state.keys())

        for key in policy_net_state:
            target_net_state[key] =  policy_net_state[key]*TAU + target_net_state[key]*(1-TAU)


        AI.trainer.targetModel.load_state_dict(target_net_state)


        env.render() 

    memory.push(state,action_idx,next_state,reward)    





    
    


    pass
