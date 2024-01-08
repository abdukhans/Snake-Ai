from SnakeEnv import SnakeEnv
from Agent import Agent
from RepalyMemory import ReplayMemory
NUM_EPSIODES = 50

BATCH_SIZE = 2
memory = ReplayMemory(BATCH_SIZE)
AI = Agent(12,4,memory,batch_size=BATCH_SIZE)
env = SnakeEnv(render=True)

action_dict = {0:"UP",1:"DOWN",2:"RIGHT",3:"LEFT"}

for i in range(NUM_EPSIODES): 


    print(i)
    
    done = False
    while not(done):

        state = AI.getGameState(env)
        action_idx = AI.SelectAction(state).int()
        action = action_dict[action_idx.item()]
        
        reward, done = env.playMove(action)
        next_state = AI.getGameState(env)

        memory.push(state,action_idx,next_state,reward)    

        AI.optimize()


        env.render() 

    memory.push(state,action_idx,next_state,reward)    





    
    


    pass
