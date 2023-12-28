"""
Snake Eater
Made with PyGame
"""

import pygame, sys, time, random
import numpy as np


BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED  = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

FPS_CONT = pygame.time.Clock()

class SnakeEnv():    

    def __init__(self,render=False,grid_size=10,frame_x=720,
    frame_y=480,difficulty = 10,epsiodeLen=100):

        self.frame_size_x = frame_x
        self.frame_size_y = frame_y
        self.render_      = render
        # self.difficulty =  difficulty

        self.frame_iteration = 0
        
        self.grid_size = grid_size



        
        self.fps_controller = lambda : FPS_CONT.tick(difficulty)
        self.epsiodeLen = epsiodeLen
        self.snake = np.array([self.get_pos(10,5), self.get_pos(10-1,5),self.get_pos(10-2,5)])
        self.snake_pos = self.get_pos(10,5)
        self.food_spawn = False
        self.food_pos = self.genFood()
        
        self.score = 0
        self.isInitRender = False
        self.gotFood = False
        
        self.dir = 'DOWN'
        
    


        pass

    def render(self):

        if self.render_:
            pass
        else:
            raise Exception("You need to make sure you turn on rendering in init mehtod")
        

        if not self.isInitRender:
            self.isInitRender = True 
            check_errors = pygame.init()
            # pygame.init() example output -> (6, 0)
            # second number in tuple gives number of errors
            if check_errors[1] > 0:
                print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
                sys.exit(-1)
            else:
                print('[+] Game successfully initialised')

            pygame.display.set_caption('Snake Eater')
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            
        self.game_window.fill(BLACK)

        # Draw snake bod
        for pos in self.snake:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1],self.grid_size, self.grid_size))

        # Draw food
        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], \
            self.grid_size, self.grid_size))

        pygame.display.update()
        # Refresh rate
        self.fps_controller()    

            

        

        
        pass

    """
        Decr:
            This function will take in an action i.e "UP", "DOWN", "LEFT", "RIGHT".
            Then it will perform that action on the game.

        Ret  :
            This will retrun a tuple (reward:Int,terminated:Bool,episodeMax:Bool).
            The "reward" variable will return the reward gained.
            The "terminated" variable will say weather or not the game is in a terminal state.
            The "episodeMax" variable will tell weather or not the game ended due to the agent taking to long 
    
    """
    def playMove(self, action):

        self.frame_iteration += 1
        # change_to = self.dir
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()
        #     # Whenever a key is pressed down
        #     elif event.type == pygame.KEYDOWN:
        #         # W -> Up; S -> Down; A -> Left; D -> Right
        #         if event.key == pygame.K_UP or event.key == ord('w'):
        #             change_to = 'UP'
        #         if event.key == pygame.K_DOWN or event.key == ord('s'):
        #             change_to = 'DOWN'
        #         if event.key == pygame.K_LEFT or event.key == ord('a'):
        #             change_to = 'LEFT'
        #         if event.key == pygame.K_RIGHT or event.key == ord('d'):
        #             change_to = 'RIGHT'
        #         # Esc -> Create event to quit the game
        #         if event.key == pygame.K_ESCAPE:
        #             pygame.event.post(pygame.event.Event(pygame.QUIT))

        # change_to = AI.pred(GAME_STATE)
        change_to = action

        # Making sure the snake cannot move in the opposite direction instantaneously
        if change_to == 'UP' and self.dir != 'DOWN':
            self.dir = 'UP'
        if change_to == 'DOWN' and self.dir != 'UP':
            self.dir = 'DOWN'
        if change_to == 'LEFT' and self.dir != 'RIGHT':
            self.dir = 'LEFT'
        if change_to == 'RIGHT' and self.dir != 'LEFT':
            self.dir = 'RIGHT'

        

        # Moving the snake
        if self.dir == 'UP':
            self.snake_pos[1] -= self.grid_size
        if self.dir == 'DOWN':
            self.snake_pos[1] += self.grid_size
        if self.dir == 'LEFT':
            self.snake_pos[0] -= self.grid_size
        if self.dir == 'RIGHT':
            self.snake_pos[0] += self.grid_size

        # Snake body growing mechanism
        self.snake = np.insert(self.snake,0, np.array(self.snake_pos,dtype=int),axis=0)

        reward = 0
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            reward = 10

            # NOTE THis is kinda stupid, You dont need the food spawn  variable
            self.food_spawn = False

        else:
            self.snake = self.snake[0:len(self.snake)-1]

        # Spawning food on the screen

        # This will generate a new food postion if and only if the snake got the food
        # Other wise it will just return the original food postion

        
        game_Over = False




        if self.checkIsOver():

            reward = -10
            game_Over = True
            self.reset()
            
            
        elif not(self.food_spawn):

            reward = 10
            self.food_pos = self.genFood()
            
            

        else:
            reward = 1
            

        return (reward,game_Over)



    def checkIsOver(self):
        
        if self.frame_iteration > self.epsiodeLen*(len(self.snake)):
            return True
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            return True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            return True

        # if self.frame_iteration == self.epsiodeLen:
        #     return True
        # Touching the snake body
        for block in self.snake[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True
            

        return False 

    def reset(self):
        self.__init__(render=self.render_,epsiodeLen=self.epsiodeLen)
        
    def get_pos(self,x,y):

        return np.array([x*self.grid_size,y*self.grid_size],dtype = int)

    def genFood(self):

        if not self.food_spawn:
            food_pos = np.array([random.randrange(1, (self.frame_size_x//self.grid_size)) * self.grid_size, 
                                 random.randrange(1, (self.frame_size_y//self.grid_size)) * self.grid_size])
            while food_pos in self.snake:
                food_pos =  np.array([random.randrange(1, (self.frame_size_x//self.grid_size)) * self.grid_size, 
                            random.randrange(1, (self.frame_size_y//self.grid_size)) * self.grid_size])


        self.food_spawn  = True

        return food_pos
