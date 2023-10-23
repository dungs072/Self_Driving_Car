
from threading import Thread
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from Communication import ConnectionNetwork
from Communication import UnRealConnectionNetwork


from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(len(gpus))
# tf.config.experimental.set_memory_growth(gpus[0], True)
### The experience replay memory ###
class ReplayBuffer:
    def __init__(self, state_dim, act_dim, size):
        self.states1_buff = np.zeros([size,state_dim],dtype=np.float32)
        self.states2_buff = np.zeros([size, state_dim], dtype = np.float32)
        self.acts_buff = np.zeros(size,dtype=np.uint8)
        self.rewards_buff = np.zeros(size,dtype = np.float32)
        self.done_buff = np.zeros(size,dtype = np.uint8)
        self.pointer, self.size, self.max_size = 0,0,size
    
    def store(self, state, act, reward, next_state, done):
        self.states1_buff[self.pointer] = state
        self.states2_buff[self.pointer] = next_state
        self.acts_buff[self.pointer] = act
        self.rewards_buff[self.pointer] = reward
        self.done_buff[self.pointer] = done
        self.pointer = (self.pointer+1)%self.max_size
        self.size = min(self.size+1,self.max_size)
        
    def sample_batch(self, batch_size=32):
        indexes = np.random.randint(0,self.size,size = batch_size)
        return dict(s = self.states1_buff[indexes],
                    next_s = self.states2_buff[indexes],
                    a = self.acts_buff[indexes],
                    r = self.rewards_buff[indexes],
                    d = self.done_buff[indexes]
                    )
#######
### Deep q neuron agent
class DQNAgent(object):
    def __init__(self, state_size, action_size,typeModel:str):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size,action_size, size = 100000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay =  0.99998
        if typeModel in ('train'):
            self.model = create_mlp(state_size,action_size,3,32)
    
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
    
    def get_action(self, state):
        if np.random.rand()<=self.epsilon:
            return np.random.choice(self.action_size)
        print(state)
        input_data = np.array(state)
        input_data = input_data.reshape(1, -1)
        act_values = self.model.predict(input_data)
        # print("this is action value")
        # print(act_values)
        return np.argmax(act_values[0]) #return action
    
    def replay(self, batch_size = 32):
        # first check if replay buffer contains enough data
        if self.memory.size<batch_size:
            return
        
        #sample a batch of data from the replay memory
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['next_s']
        done = minibatch['d']
        
        # Calculate the tentative target: Q(s', a)
        target = rewards + (1-done) * self.gamma * np.amax(self.model.predict(next_states),axis=1)
        
        # With the keras API, the target (usually) must  have the same
        # shape as the predictions.
        # however, we only need to update the network for the actions
        # which were actually taken.
        # we can accomplish this by setting the target to be equal to 
        # the prediction for all values
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size),actions] = target
        
        # Run one training step
        self.model.train_on_batch(states,target_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self,name):
        #self.model.load_weights(name)
        self.model = tf.keras.models.load_model(name)
    
    def save(self, name):
        self.model.save(name,save_format='h5')
    
#######
### Create ENV
class RealWorldEnv:
    """
        State: 
        - distance between car and obstacle only
        Action: 4 
        - Turn left, right, go forward, stop
    """
    def __init__(self,f_standard_dist = 10,side_standard_dist =12,min_dist = 5, step_frames =0.02):
        self.action_list = [0,1,2]
        self.state_dim = 5
        self.standard_distance = f_standard_dist
        self.side_standard_distance = side_standard_dist
        self.min_dist = min_dist
        self.step_frames = step_frames
        self.connNetwork = UnRealConnectionNetwork()
        self.pre_action = None

    def reset(self):
        self.pre_action=None
    def get_current_state(self):
        return self.connNetwork.get_data_dist()
        
    def step(self, action):
        #assert action in self.action_space
        # get three datas f, l, r distance from car
        self.is_terminal = False
        
        ### implement action here 
        reward = 0
        state = None
        f_dist = 0
        t0 = datetime.now()
        dt = 0
        while dt<=self.step_frames:
            dt = (datetime.now()-t0).total_seconds()
            # print(dt)
            # print(action)
            Rm,Rp,Rex = 0,0,0
            distances = self.connNetwork.get_data_dist()
            (f_dist, l_dist, r_dist,ml_dist,mr_dist) = distances
            min_distance = min(distances)
            max_distance = max(distances)
            if(min_distance<self.min_dist): # front car pumped a obstacle
                reward = -1000
                self.is_terminal = True
                state = distances
                self.connNetwork.send_action('F_E')
                self.pre_action = action
                return reward, np.asarray(state), self.is_terminal
            if action==0: # move forward
                self.connNetwork.send_action('F')
                Rm = 0.5
            elif action==1: # move left
                self.connNetwork.send_action('L')
                Rm = -0.1
                if self.pre_action==2:
                    Rp-=1
            elif action==2: # move right
                self.connNetwork.send_action('R')
                Rm = -0.1
                if self.pre_action==1:
                    Rp-=1
            elif action==3: # stop
                self.connNetwork.send_action('S')
                Rm = 0
            else:
                raise ValueError('`action` should be between 0 and 3.')
            if(min_distance==r_dist or min_distance==mr_dist):
                if max_distance == l_dist or max_distance==ml_dist:
                    if action==1:
                        Rex = 0.7
            if(min_distance==l_dist or min_distance==ml_dist):
                if max_distance == r_dist or max_distance==mr_dist:
                    if action==2:
                        Rex = 0.7
            if(min_distance==f_dist):
                if max_distance == l_dist or max_distance==ml_dist:
                    if action==1:
                        Rex = 0.7
                if max_distance == r_dist or max_distance==mr_dist:
                    if action==2:
                        Rex = 0.7
        reward = Rm+Rp+Rex            
        state = distances#(f_dist,l_dist,r_dist)
        self.pre_action = action
        return reward, np.asarray(state), self.is_terminal  
            
                

def create_mlp(input_dim, n_action, n_hidden_layers = 1, hidden_dim =32):
    ## A multi-layer perceptron
    
    #input layer
    i = Input(shape=(input_dim,))
    x = i
    
    #hidden layers
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)
    
    #final layer
    x = Dense(n_action)(x)
    
    #make the model
    model = Model(i,x)
    
    model.compile(loss='mse',optimizer='adam')
    print((model.summary()))
    return model

def play_one_episode(agent:DQNAgent,env:RealWorldEnv,is_train:str, batch_size:int):
    # note: after transforming states are already 1xD
    states = env.get_current_state()
    state = np.asarray(states)
    done = False
    
    while not done:
        
        action = agent.get_action(state)
        reward, next_state, done = env.step(action)
        if is_train == 'train' or is_train =='retrain':
            agent.update_replay_memory(state,action,reward, next_state, done)
            agent.replay(batch_size)
        state = next_state

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# for input handler
stop = False

def stop_handler():
  global stop
  while not stop:
    user_input = input()
    if user_input == 'q':
      print("Stopping...")
      stop = True
      agent.save(f'{models_folder}/dqn3.h5')

process = Thread(target=stop_handler)
process.start()
##################
if __name__=='__main__':
    #config
    models_folder = 'self_driving_car_models'
    
    maybe_make_dir(models_folder)
    mode = 'train'
    num_episodes = 5001
    env = RealWorldEnv()
    state_size = env.state_dim
    action_size = len(env.action_list)
    agent = DQNAgent(state_size,action_size,mode)
    
    if mode =='test':
        print('testing')
        #no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01
        # load trained weights
        agent.load(f'{models_folder}/dqn3.h5')
    if mode =='retrain':
        print('retraining')
        agent.epsilon = 1
        agent.load(f'{models_folder}/dqn3.h5')
    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        play_one_episode(agent,env,mode,4000)
        env.reset()
        dt = datetime.now()-t0
        print(f"episode: {e + 1}/{num_episodes}, duration: {dt}, epsilon: {agent.epsilon}")
        if e%5000==0 and mode in ('train','retrain'):
            agent.save(f'{models_folder}/dqn3.h5')
        if stop: break
        # save the weight when we are done
    if mode == 'train' or mode =='retrain':
        # save the DQN
        agent.save(f'{models_folder}/dqn3.h5')
        
        
        
    
        
        
    
        
        
    
