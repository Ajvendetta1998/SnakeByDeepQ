
import pygame 
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from copy import deepcopy
import numpy as np 
import os 
import keras 
import sys 
import matplotlib.pyplot as plt

import random
from collections import deque
import numpy as np

class DQL:
    def __init__(self, model, actions, discount_factor=0.99, exploration_rate=0.9, memory_size=1000000, batch_size=500,base_decay_rate = 0.99995, decay_rate=0.95, base_exploration_rate = 1.0,validation_batch_size = 100): # 0.1 de base explo
        #NN
        self.model = model
        self.actions = actions
        #gamma
        self.discount_factor = discount_factor
        #for epsilon-greedy
        self.exploration_rate = exploration_rate
        #buffer
        self.memory = deque(maxlen=memory_size)
        self.evalmemory  = deque(maxlen = memory_size)
        self.batch_size = batch_size
        #diminish exploration 
        self.base_decay_rate = base_decay_rate
        self.decay_rate = decay_rate
        self.validation_batch_size = validation_batch_size
        self.base_exploration_rate = base_exploration_rate

        #ligne 1 = scores, ligne 2 = reward episode, ligne 3 = explo rates

    def get_action(self, state, direction, snake_list, block_size, width, height):
        action = np.random.randint(len(self.actions))
        possible_moves = list(range(0, len(self.actions)))
        # eliminate all impossible moves
        acts = list(self.actions)
        poss_copy = possible_moves.copy()
        for p in poss_copy:
            (u, v) = (snake_list[-1][0] + acts[p][1] * block_size, snake_list[-1][1] + acts[p][0] * block_size)
            if (u < 0 or u >= width or v < 0 or v >= height or [u, v] in snake_list):
                possible_moves.remove(p)
        if (len(possible_moves) > 0):
            if np.random.rand() < self.base_exploration_rate + self.exploration_rate:
                # Choose a random action
                action = possible_moves[np.random.randint(len(possible_moves))]
            else:
                # Choose the best action according to the model
                q_values = self.model.predict(np.array([state]), verbose=0)

                sorted = q_values[0].argsort()[::-1]
                for s in sorted:
                    if (s in possible_moves):
                        action = s
                        break

        return action

    def add_memory(self, state, action, reward, next_state, done,episode_reward):
        x = np.random.rand()

        if(x<0.9):
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.evalmemory.append((state, action, reward, next_state, done))
        episode_reward=episode_reward*self.discount_factor+reward
        return episode_reward

    def train(self,batch_size):
        if len(self.memory) < batch_size:
            # Not enough memories to train the model
            return
        # Randomly sample memories from the replay buffer
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = np.array(states)
        next_states = np.array(next_states)

        # Decrease the exploration rate
        self.exploration_rate *= self.decay_rate
        self.base_exploration_rate*= self.base_decay_rate

        # Calculate the target Q-values
        #qw(st+1,a[0],a[1],a[2].. )
        next_q_values = self.model.predict(next_states,verbose = 0)

        target_q_values = np.zeros((batch_size,len(self.actions)))

        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]

            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])

        # Train the model with the target Q-values
        # we want for qw(St,a) to become target_q[a]
        self.model.fit(states,target_q_values,epochs=1, verbose = 0)


    def evaluate(self):

        batch = random.sample(self.evalmemory, min(len(self.evalmemory),self.validation_batch_size))
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = np.array(states)
        next_states = np.array(next_states)
        # Calculate the target Q-values
        next_q_values = self.model.predict(next_states,verbose = 0)
        target_q_values = np.zeros((len(batch),len(self.actions)))

        for i in range(len(batch)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]

            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * max(next_q_values[i])
        j = np.random.randint(len(target_q_values))

        print(target_q_values[j],self.model.predict(np.array([states[j]]))[0])

        self.model.evaluate(states, target_q_values  )

        print("Exploration rate " , self.exploration_rate)
        
# Snake block size
block_size = 25


# Set display width and height
width = 500 
height = 500


pygame.init()   


# Create display surface
screen = pygame.display.set_mode((width, height))

# Set title for the display window
pygame.display.set_caption("Snake Game CNN")

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
grey = (100,100,100)
green = (0, 255, 0)
dark_green = (0, 100, 0)
# Set clock to control FPS
clock = pygame.time.Clock()

# Font for displaying score
font = pygame.font.Font(None, 30)

# FPS
fps = 2

def game_over():
    # Display Game Over message
    text = font.render("Game Over!", True, red)
    screen.blit(text, [width/2 - text.get_width()/2, height/2 - text.get_height()/2])

def display_score(score,gen,s,maxscore):
    # Display current score
    text = font.render("Gen: " + str(gen) + " Length : " + str(score)+ " Score: " + str(s) + " Max_score : "+str(maxscore), True, black)
    screen.blit(text, [0,0])

def draw_snake(snake_list):
    # Draw the snake
    for block in snake_list[:-1]:
        pygame.draw.rect(screen, green, [block[0], block[1], block_size, block_size])
        pygame.draw.rect(screen, black, [block[0], block[1], block_size, block_size], 1)
    pygame.draw.rect(screen, dark_green, [snake_list[-1][0], snake_list[-1][1], block_size, block_size])
    pygame.draw.rect(screen, black, [snake_list[-1][0], snake_list[-1][1], block_size, block_size],1)


def generate_food(snake_list):
    # Generate food for the snake where there is no snake
    food_x, food_y = None, None
    
    while food_x is None or food_y is None or [food_x, food_y] in snake_list:
        food_x = round(random.randrange(0, width - block_size) / block_size) * block_size
        food_y = round(random.randrange(0, height - block_size) / block_size) * block_size
    return food_x, food_y


#dictionary of possible actions 
actions = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}

#(x,y) apple + snake body (which has at most width*height parts) 
input_size = 2*width*height//block_size**2+2


def initNNmodel():

    # create a CNN
    model = Sequential()
    model.add(Conv2D(80, kernel_size = 3 , input_shape=(3,width//block_size, height//block_size), activation='ReLU'))
    model.add(Flatten())
    model.add(Dense(1024 , activation = 'ReLU'))
    model.add(Dense(512 , activation = 'ReLU'))
    model.add(Dense(256 , activation = 'ReLU'))
    model.add(Dense(128,activation = 'ReLU'))
    model.add(Dense(len(actions), activation='linear'))

    # Compile the model using mean squared error loss and the Adam optimizer
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def state(snake_list,apple):
    layer_head = np.zeros((height//block_size,width//block_size))
    layer_tail = np.zeros((height//block_size,width//block_size))
    layer_apple = np.zeros((height//block_size,width//block_size))
    layer_head[snake_list[-1][0]//block_size-1, snake_list[-1][1]//block_size-1] =1 
    layer_apple[apple[0]//block_size-1, apple[1]//block_size-1] =1 
    for s in snake_list[:-1]:
        layer_tail[s[0]//block_size-1, s[1]//block_size-1] =1 
    input = np.zeros( (3,height//block_size, width//block_size))
    input[0,:,:] = layer_head
    input[1,:,:] = layer_tail
    input[2,:,:] = layer_apple

    return(input)

def normalized_distance(u,v,food_x,food_y):
    return np.sqrt((((u-food_x)/width)**2+((v-food_y)/height)**2)/2)

def inBounds(u,v):
    if(u>=0 and v>=0):
        if(u<width and v<height):
            return True
    return False

def gaussian_aroundone(x,alpha):
    return(np.exp(-alpha*(x-1)**2))
def danger_distance(direction, snake_list):
    dis =0  
    acts = list(actions.values())
    (u,v) = (snake_list[-1][0],snake_list[-1][1])
    while (inBounds(u,v)):
        u +=block_size*acts[direction][1]
        v+=block_size*acts[direction][0]
        if([u,v] in snake_list[1:]):
            return (-1+1.0*dis*block_size/max(width,height))
        dis+=1
    return(0)
#reward function for each state and action
def reward(action, snake_list,episode_length):
    copy = deepcopy(snake_list)
    p = copy[-1]
    a = list(actions.values())
    (u,v)=(a[action][1]*block_size+p[0],a[action][0]*block_size+p[1])
    penalty_touch_self = 0 
    if [u,v] in snake_list:
        penalty_touch_self=-1 # return a negative reward if the snake collides with itself
    copy.append([u,v])
    del copy[0]
    
    
    global food_x, food_y

    # reward the agent for getting closer to the food
    reward_distance = 1-normalized_distance(u,v,food_x,food_y)

    #if too far then the reward is very close to 0 
    gass_reward =gaussian_aroundone(reward_distance,20)

    # reward the agent for eating the food
    reward_eat = 1 if u == food_x and v == food_y else 0

    # penalize the agent for moving away from the food
    penalty_distance = -2 if normalized_distance(u,v,food_x,food_y) > normalized_distance(p[0], p[1], food_x, food_y) else 1

    # penalize he agent for hitting a wall
    penalty_wall = -1 if not (inBounds(u,v)) else 0

    #penalize the agent for getting closer to danger
    penalty_danger = danger_distance(action,snake_list)

    #print(penalty_danger)
    compacity_value = 1/compacity(snake_list)

    #accessible points 
    accessible_points_proportion = find_accessible_points(snake_list)
    episode_length_penalty = -episode_length/(width*height//block_size**2+2)/5
    penalties = np.array([accessible_points_proportion,penalty_distance,penalty_touch_self,penalty_distance*gass_reward,reward_eat,penalty_wall,penalty_danger,compacity_value,episode_length_penalty])
    penalty_names  = ['accessible_points_proportion','penalty_distance','penalty_touch_self','penalty_distance*gass_reward','reward_eat','penalty_wall','penalty_danger','compacity','episode_len_penalty']
    c = np.array([0.1863077196643149,0.8176132430371873,0.560514500321722,0.2359458810412274,0.5130391965064794,0.4729984548003503,0.02492140605359113,0.5531607672522686,0.4327075132071214])

    total_reward = penalties@c/c.sum()

    return total_reward

def compacity(snake_list):
    snake_list = np.array(snake_list)
    min_x = snake_list[:,0].min()
    min_y = snake_list[:,1].min()
    max_x = snake_list[:,0].max()
    max_y = snake_list[:,1].max()
    return((max_y-min_y+block_size)*(max_x-min_x+block_size)/(len(snake_list)*block_size**2))   


#if all cells are accessible 
def find_accessible_points(snake_list):
    accessible_points= np.zeros((height//block_size,width//block_size))
    head_position = snake_list[-1]
    explore = [head_position]

    while(len(explore)>0):
        p = explore.pop()
        accessible_points[p[1]//block_size,p[0]//block_size]=1
        for m in actions.values():
            (u,v)=(m[1]*block_size+p[0],m[0]*block_size+p[1])
            if(inBounds(u,v)):
                if(accessible_points[v//block_size,u//block_size]==0):
                    if(not [u,v] in snake_list):
                        explore.append((u,v))

    return((np.sum(accessible_points)+len(snake_list)-1)*block_size**2/(height*width))

#initialize NN
filename = str(width)+" " + str(height)+" CNNDeepQ.h5"
if(os.path.exists("./"+filename)):
    print("model already exists ")
    model = keras.models.load_model("./"+filename)

else: 

    model = initNNmodel()

#initialize deepQ
dql = DQL(model,actions.values())
# Initialize pygame

food_x, food_y = generate_food([])   

def main(gen,length,maxlen):
    episode_reward = 0
    # Initial snake position and food
    snake_x = (width//block_size)//2     * block_size   
    snake_y = (height//block_size)//2     * block_size   

    snake_list = [[snake_x,snake_y]]
    
    global food_x,food_y
    #food_x, food_y = generate_food([])   
    episode_length =0 
    # Initial snake direction and length
    direction = "right"
    snake_length =length
    

    prev_direction = "right"
    #direction of the snake [0,1,2,3] each corresponding to one of up down left right
    a=0
    # = True if snake hits wall or itself
    done = False
    # number of collected fruit
    score = 0
    acts = list(actions.values())
    # Game loop
    check = 0 
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Start a new game
                main()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        
        St1 = state(snake_list,[food_x,food_y]) 
        a= dql.get_action(St1,a,snake_list,block_size,width,height)
        if (gen<nb_episodes_model):
            r = reward(a,snake_list,episode_length-check)

        #move snake
        snake_x+=acts[a][1]*block_size
        snake_y+=acts[a][0]*block_size
        # Add new block of snake to the list
        snake_list.append([snake_x, snake_y])
        # Keep the length of snake same as snake_length
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check if snake hits the boundaries
        if snake_x >= width or snake_x < 0 or snake_y >= height or snake_y < 0:
            done = True
        # Check if snake hits itself
        for block in snake_list[:-1]:
            if block[0] == snake_x and block[1] == snake_y:
                done = True
                break

        # Fill the screen with white color
        screen.fill(white)

        # Display food
        pygame.draw.rect(screen, red, [food_x, food_y, block_size, block_size])

        # Draw the snake
        draw_snake(snake_list)

        if (gen<nb_episodes_model):
            St2 = state(snake_list,[food_x,food_y])
            #add to Buffer
            episode_reward= dql.add_memory(St1,a,r,St2,done,episode_reward)

        # Display score and other metrics
        display_score(snake_length-1,gen,score,maxlen)

        # Update the display
        pygame.display.update()

        # Check if snake hits the food
        if snake_x == food_x and snake_y == food_y:
            food_x, food_y = generate_food(snake_list)
            snake_length += 1
            score+=1
            check = episode_length
        episode_length+=1

        if((episode_length-check)%((width*height//block_size**2)*5) ==0 ):
  
            #dql.exploration_rate/= dql.decay_rate
            done = True
        #if snake has hit something quit
        if(done):

            if (gen<nb_episodes_model):
                return [snake_length,episode_length,score,episode_reward]
            else :
                return [snake_length,episode_length,score,None]
        
        # Set the FPS
        #clock.tick(fps)


#number of episodes
num_episodes =150
nb_episodes_model = 150
#maximum score reached
m =0 
#initial max length for the snake at birth
max_length = 1
#maximum allowed length for a snake 
max_max_length = width*height//(block_size**2)                                            #LA
#max_length = width*height//block_size**2//20



#generation of episodes 
for i in range(num_episodes):

    #do a generation and see the outcome
    a= main(i,5,m)                                                                  #LA
    #update maximum score 
    m = max(a[2],m)
    #generate a new food position every 20 generations
    if(i%20 ==0):
        food_x, food_y = generate_food([])   
    #increase maximum birth length every 1000 generation 
    if((i+1)%20==0):
        model.save(str(width)+" " + str(height)+" DeepQ.h5")
    print("episode reward : ", a[-1], "Exploration Rate : ", dql.exploration_rate+dql.base_exploration_rate)


    dql.train(a[1])

    


model.save(str(width)+" " + str(height)+" DeepQ.h5")