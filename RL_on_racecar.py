# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:18:57 2019

@author: joshua.touati
"""
#size state space
print(env.observation_space)
#size action space
print(env.action_space)
#first element state space
print(env.observation_space.low)
#last element state space
print(env.observation_space.high)
#initial state of the environment
print(env.reset())
#take action gas
print(env.step([0,1,0]))


# In[Set working directory]

# Import required package
import os # analysis:ignore

# Set working directory, get paths where datasets are stored
if os.getlogin() == "joshua.touati":
    os.chdir('C:/Users/joshua.touati/Documents/VU/Research Paper')

# In[Imports]
import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import SGD, RMSprop, Adam, Adamax
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import cv2
import bisect



# In[Environment functions]

vector_size = 10 * 10 + 7 + 4

class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # one feedforward nn for all actions.

    def predict(self, s):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    def update(self, s, G):
        self.model.fit(s.reshape(-1, vector_size), np.array(G).reshape(-1, 11), nb_epoch=1, verbose=0)

    def sample_action(self, s, epsilon, method, tau):
        Q_values = self.predict(s)
        if method == 'epsilon-greedy':
            if np.random.random() < epsilon:
                return random.randint(0, 10), Q_values
            else:
                return np.argmax(Q_values), Q_values
            
        elif method == 'softmax':
            Prob = np.exp(Q_values/tau) #tau is the temperature
            Prob = Prob / Prob.sum()
            
            # Determine cumulative distribution        
            cdf = []
            Total = 0
            for a in range(len(Q_values) - 1):
                Total += Prob[a]
                cdf.append(Total)
            
            # Force last entry to be 1
            cdf.append(1)
            
            # Select action based on cdf
            x = random.random()
            idx = bisect.bisect(cdf, x)
            
            return idx, Q_values
        
        elif method == 'VDBE-Softmax':
            if np.random.random() < epsilon:
                Prob = np.exp(Q_values/0.25) #1 is tau here
                Prob = Prob / Prob.sum()
                
                # Determine cumulative distribution        
                cdf = []
                Total = 0
                for a in range(len(Q_values) - 1):
                    Total += Prob[a]
                    cdf.append(Total)
                
                # Force last entry to be 1
                cdf.append(1)
                
                # Select action based on cdf
                x = random.random()
                idx = bisect.bisect(cdf, x)
                
                return idx, Q_values
            else:
                return np.argmax(Q_values), Q_values


def create_nn():
    if os.path.exists('race-car.h5'):
        return load_model('race-car.h5')
        
    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))

 #   model.add(Dense(32, init='lecun_uniform'))
 #   model.add(Activation('relu'))
 #   model.add(Dropout(0.3))

    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

#     rms = RMSprop(lr=0.005)
#     sgd = SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
#     try "adam"
#     adam = Adam(lr=0.0005)
    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss= 'mse', optimizer=adamax)
    model.summary()
    
    return model


def transform(s):
#     cv2.imshow('original', s)
#     cv2.waitKey(1)
    
    # crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    # bottom_black_bar is the section of the screen with steering, speed, abs and gyro information.
    # we crop off the digits on the right as they are illigible, even for ml.
    # since color is irrelavent, we grayscale it.
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)
    
    # upper_field = observation[:84, :96] # this is the section of the screen that contains the track.
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
#     cv2.imshow('video', upper_field_bw)
#     cv2.waitKey(1)
    upper_field_bw = upper_field_bw.astype('float')/255
        
    car_field = s[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]

#     print(car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255)
    car_field_t = [car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255]
    
#     rotated_image = rotateImage(car_field_bw, 45)
#     cv2.imshow('video rotated', rotated_image)
#     cv2.waitKey(1)

    return bottom_black_bar_bw, upper_field_bw, car_field_t


# this function uses the bottom black bar of the screen and extracts steering setting, speed and gyro data
def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean()/255
    left_steering = a[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2
    
    left_gyro = a[6, 46:60].mean()/255
    right_gyro = a[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2
    
    speed = a[:, 0][:-2].mean()/255
    abs1 = a[:, 6][:-2].mean()/255
    abs2 = a[:, 8][:-2].mean()/255
    abs3 = a[:, 10][:-2].mean()/255
    abs4 = a[:, 12][:-2].mean()/255
    
#     white = np.ones((round(speed * 100), 10))
#     black = np.zeros((round(100 - speed * 100), 10))
#     speed_display = np.concatenate((black, white))*255
        
#     cv2.imshow('sensors', speed_display)
#     cv2.waitKey(1)

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]


def convert_argmax_Q_values_to_env_action(output_value):
    # we reduce the action space to 11 values.  9 for steering, 2 for gaz/brake.
    # to reduce the action space, gaz and brake cannot be applied at the same time.
    # as well, steering input and gaz/brake cannot be applied at the same time.
    # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.
    
    gaz = 0.0
    brake = 0.0
    steering = 0.0
    
    # output value ranges from 0 to 10
    
    if output_value <= 8:
        # steering. brake and gaz are zero.
        output_value -= 4
        steering = float(output_value)/4
    elif output_value >= 9 and output_value <= 9:
        output_value -= 8
        gaz = float(output_value)/3 # 33% 
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value)/2 # 50% brakes
    else:
        print("error")
        
    white = np.ones((round(brake * 100), 10))
    black = np.zeros((round(100 - brake * 100), 10))
    brake_display = np.concatenate((black, white))*255  
    
    white = np.ones((round(gaz * 100), 10))
    black = np.zeros((round(100 - gaz * 100), 10))
    gaz_display = np.concatenate((black, white))*255
        
    control_display = np.concatenate((brake_display, gaz_display), axis=1)

    cv2.imshow('controls', control_display)
    cv2.waitKey(1)
    
    return [steering, gaz, brake]


# In[Deep_Q_learning]
    
def Deep_Q_learning(env, model, epsilon, gamma, method, tau):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        action_Q_values, Q_values = model.sample_action(state, epsilon, method, tau)
        prev_state = state
        action = convert_argmax_Q_values_to_env_action(action_Q_values)
        observation, reward, done, info = env.step(action)

        a, b, c = transform(observation)        
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      

        # update the model
        # standard Q learning TD(0)
        next_Q_values = model.predict(state)
        G = reward + gamma*np.max(next_Q_values)
        y = Q_values[:]
        y[action_Q_values] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1
            
        if iters > 1500:
            print("This episode is stuck")
            break
        
    return totalreward, iters
  

# In[Deep_SARSA]
    
def Deep_SARSA(env, model, epsilon, gamma, method, tau):
    #initialize session
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        action_Q_values, Q_values = model.sample_action(state, epsilon, method, tau)
        prev_state = state
        action = convert_argmax_Q_values_to_env_action(action_Q_values)
        observation, reward, done, info = env.step(action)

        a, b, c = transform(observation)        
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        next_action_Q_values, next_Q_values = model.sample_action(state, epsilon, method, tau)

        # update the model
        # standard SARSA TD(0)
        G = reward + (gamma * next_Q_values[next_action_Q_values])
        y = Q_values[:]
        y[action_Q_values] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1
            
        if iters > 1500:
            print("This episode is stuck")
            break
        
    return totalreward, iters

# In[Deep_Q_learning_Experience_Replay]
    
def Deep_Q_learning_ER(env, model, epsilon, gamma, method, T_ER, N_ER, D, D_Capacity, k, l, tau):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        action_Q_values, Q_values = model.sample_action(state, epsilon, method, tau)
        prev_state = state
        action = convert_argmax_Q_values_to_env_action(action_Q_values)
        observation, reward, done, info = env.step(action)

        a, b, c = transform(observation)        
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        
        # When capacity of database is reached, delete the first trajectory
        if len(D) == D_Capacity:
            for i in range(T_ER):
                del D[i]
        
        # Add sample to database
        D.append((prev_state, action_Q_values, state, reward))
        k += 1
        
        # Update model if trajectory complete
        if k == l * T_ER: 
            Q_learning_trajectories(D, gamma, T_ER, N_ER, l)
            #Q_learning_sampling(D, gamma, T_ER, N_ER, l)
            l += 1
        
        totalreward += reward
        iters += 1
            
        if iters > 1500:
            print("This episode is stuck")
            break
        
    return totalreward, iters

def Q_learning_sampling(D, gamma, T_ER, N_ER, l):
    
    # Get number of observations / experiences in D
    N_obs = len(D)
    
    # Update Q values
    for i in range(N_ER * l * T_ER):
        
        # Retrieve random sample
        idx = np.floor(random.random() * N_obs)
        D_temp = D[int(idx)]
        
        # Update model based on sample
        Q_values = model.predict(D_temp[0])
        next_Q_values = model.predict(D_temp[2])
        G = D_temp[3] + (gamma * np.max(next_Q_values))
        y = Q_values[:]
        y[D_temp[1]] = G
        model.update(D_temp[0], y)


def Q_learning_trajectories(D, gamma, T_ER, N_ER, l):
       
    # Get number of Trajectories in D
    N_trajectories = int(round((len(D)/T_ER)))
    
    # Update the model
    for i in range(N_ER * l):
        
        # Retrieve random trajectory
        idx = random.randint(1, N_trajectories)
        
        # Retrieve actual trajectory
        D_temp1 = D[(idx * T_ER) - T_ER:(idx * T_ER)]
        
        for j in range(len(D_temp1)):
                    
            #Use sample of trajectory to update model
            D_temp2 = D_temp1[j]
            
            # Update model based on sample
            Q_values = model.predict(D_temp2[0])
            next_Q_values = model.predict(D_temp2[2])
            G = D_temp2[3] + (gamma * np.max(next_Q_values))
            y = Q_values[:]
            y[D_temp2[1]] = G
            model.update(D_temp2[0], y)
        


# In[Plot Functions]
    
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    
  plt.legend(loc = 'lower right')  
  plt.plot(running_avg)
  
  plt.title("Running Average")
  plt.ylabel("Average reward over past 100 episodes")
  plt.xlabel("Episode")
  plt.show()
  
  
#def plot_reward_per_episode(totalrewards):
#    plt.plot(totalrewards)
#    plt.title("Reward Per Episode")
#    plt.xlabel("Episode")
#    plt.ylabel("Reward")
#    plt.show()
    
def plot_reward_per_episode1(list_data, list_names):
    
    for i in range(len(list_data)):
        plt.plot(list_data[i], label = list_names[i])

    plt.legend(loc = 'lower right')
    plt.title("Reward Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    
    return

def plot_running_avg1(list_data, list_names):
    
    for i in range(len(list_data)):
        plt.plot(list_data[i], label = list_names[i])

    plt.legend(loc = 'upper left')
    plt.xlabel("Episode")
    plt.ylabel("Average reward over past 100 episodes")
    plt.title("Running Average")
    plt.ylim(-100, 600)
    plt.show()
    
    return


# In[Main]
#offline performance meten door het model niet te updaten en wel episodes te doen!!!

##### Create environment #####
# =============================================================================
# gym.envs.register(
#      id='CarRacing-v1', # CHANGED
#     entry_point='gym.envs.box2d:CarRacing',
#     max_episode_steps=1500, # CHANGED
#     reward_threshold=900,
# )
# env = gym.make('CarRacing-v1')
# =============================================================================
env = gym.make('CarRacing-v0')
env = wrappers.Monitor(env, './Recordings_car/', video_callable = lambda episode_id: episode_id%25==0, force=True)
model = Model(env)


##### Global Constants #####
gamma = 0.99
episodes = 1000


#####save & load stuff #####
np.save("totalrewards_Q_eps", totalrewards_Q_eps)
np.save("totalrewards_SARSA_eps", totalrewards_SARSA_eps)
np.save("running_avg_Q_eps", running_avg_Q_eps)
np.save("running_avg_SARSA_eps", running_avg_SARSA_eps)
np.save("totalrewards_SARSA_softmax", totalrewards_SARSA_softmax)
np.save("running_avg_SARSA_softmax", running_avg_SARSA_softmax)
np.save("totalrewards_Q_softmax", totalrewards_Q_softmax)
np.save("running_avg_Q_softmax", running_avg_Q_softmax)
np.save("totalrewards_Qlearning_ER_Samples", totalrewards_Qlearning_ER_Samples)
np.save("running_avg_Qlearning_ER_Samples", running_avg_Qlearning_ER_Samples)
np.save("totalrewards_Qlearning_ER_Trajectories", totalrewards_Qlearning_ER_Trajectories)
np.save("running_avg_Qlearning_ER_Trajectories", running_avg_Qlearning_ER_Trajectories)


totalrewards_Q_eps = np.load("totalrewards_Q_eps.npy")
totalrewards_SARSA_eps = np.load("totalrewards_SARSA_eps.npy")
totalrewards_SARSA_softmax = np.load("totalrewards_SARSA_softmax.npy")
totalrewards_Q_softmax = np.load("totalrewards_Q_softmax.npy")
totalrewards_Qlearning_ER_Samples = np.load("totalrewards_Qlearning_ER_Samples.npy")
totalrewards_Qlearning_ER_Trajectories = np.load("totalrewards_Qlearning_ER_Trajectories.npy")

running_avg_Q_eps = np.load("running_avg_Q_eps.npy")
running_avg_SARSA_eps = np.load("running_avg_SARSA_eps.npy")
running_avg_SARSA_softmax = np.load("running_avg_SARSA_softmax.npy")
running_avg_Q_softmax = np.load("running_avg_Q_softmax.npy")
running_avg_Qlearning_ER_Samples = np.load("running_avg_Qlearning_ER_Samples.npy")
running_avg_Qlearning_ER_Trajectories = np.load("running_avg_Qlearning_ER_Trajectories.npy")


plot_running_avg1([running_avg_Q_eps, running_avg_SARSA_eps, running_avg_Q_softmax, 
                   running_avg_SARSA_softmax], ["Q-learning-epsilon", "SARSA-epsilon", 
                                            "Q-learning-softmax", "SARSA-softmax"])
    
plot_running_avg1([running_avg_Q_eps, running_avg_Qlearning_ER_Samples, running_avg_Qlearning_ER_Trajectories],
                  ["Q-learning", "Q-learning-ER-Samples", "Q-learning-ER-Trajectories"])


# In[Deep Q-learning - Epsilon greedy]

totalrewards = np.empty(episodes)
method = 'epsilon-greedy'
random.seed(42)
for n in range(episodes):
    epsilon = 0.5/np.sqrt(n+1 + 900) 
    totalreward, iters = Deep_Q_learning(env, model, epsilon, gamma, method)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n + 1, "iters", iters, "total reward:", totalreward, "epsilon:", epsilon, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
    if n % 50 == 0:
        model.model.save('race-car.h5')

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

#plot_reward_per_episode(totalrewards)
#plot_running_avg(totalrewards)

totalrewards_Q_eps = totalrewards

N = len(totalrewards)
running_avg_Q_eps = np.empty(N)
  
for t in range(N):
    running_avg_Q_eps[t] = totalrewards[max(0, t-100):(t+1)].mean()

import numpy
numpy.where(running_avg_Q_eps == max(running_avg_Q_eps))


# In[Deep SARSA - Epsilon greedy]
totalrewards = np.empty(episodes)
method = 'epsilon-greedy'
random.seed(42)
for n in range(episodes):
    epsilon = 0.5/np.sqrt(n+1 + 900) 
    totalreward, iters = Deep_SARSA(env, model, epsilon, gamma, method, tau)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n + 1, "iters", iters, "total reward:", totalreward, "epsilon:", epsilon, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
    if n % 50 == 0:
        model.model.save('race-car.h5')

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

#plot_reward_per_episode(totalrewards)
plot_running_avg(totalrewards)

totalrewards_SARSA_eps = totalrewards

N = len(totalrewards)
running_avg_SARSA_eps = np.empty(N)
  
for t in range(N):
    running_avg_SARSA_eps[t] = totalrewards[max(0, t-100):(t+1)].mean()
    
import numpy
numpy.where(running_avg_SARSA_eps == max(running_avg_SARSA_eps))
    
plot_running_avg1([running_avg_Q_eps, running_avg_SARSA_eps], ["Q-learning", "SARSA"])

plot_reward_per_episode1([totalrewards_Q_eps, totalrewards_SARSA_eps], ["Q-learning", "SARSA"])


# In[Deep Q-learning - Softmax]
totalrewards = np.empty(episodes)
method = 'softmax'
random.seed(42)
for n in range(episodes):
    epsilon = 0.5/np.sqrt(n+1 + 900) 
    if n < 250:
        tau = 1.5
    elif n < 500:
        tau = 1
    elif n < 750:
        tau = 0.5
    elif n < 1000:
        tau = 0.25
    totalreward, iters = Deep_Q_learning(env, model, epsilon, gamma, method, tau)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n + 1, "iters", iters, "total reward:", totalreward, "tau:", tau, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
    if n % 50 == 0:
        model.model.save('race-car.h5')

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

#plot_reward_per_episode(totalrewards)
plot_running_avg(totalrewards)

totalrewards_Q_softmax = totalrewards

N = len(totalrewards)
running_avg_Q_softmax = np.empty(N)
  
for t in range(N):
    running_avg_Q_softmax[t] = totalrewards[max(0, t-100):(t+1)].mean()

import numpy
numpy.where(running_avg_Q_softmax == max(running_avg_Q_softmax))


# In[Deep SARSA - Softmax]
totalrewards = np.empty(episodes)
method = 'softmax'
tau = 0.25
random.seed(42)
for n in range(episodes):
    epsilon = 0.5/np.sqrt(n+1 + 900)
    totalreward, iters = Deep_SARSA(env, model, epsilon, gamma, method, tau)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n + 1, "iters", iters, "total reward:", totalreward, "tau:", tau, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
    if n % 50 == 0:
        model.model.save('race-car.h5')

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

#plot_reward_per_episode(totalrewards)
plot_running_avg(totalrewards)

totalrewards_SARSA_softmax = totalrewards

N = len(totalrewards_SARSA_softmax)
running_avg_SARSA_softmax = np.empty(N)
  
for t in range(N):
    running_avg_SARSA_softmax[t] = totalrewards[max(0, t-100):(t+1)].mean()

import numpy
numpy.where(running_avg_SARSA_softmax == max(running_avg_SARSA_softmax))


# In[Deep Q-learning Experience Replay - Epsilon greedy]

totalrewards = np.empty(episodes)
method = 'epsilon-greedy'

# Experience replay parameters
D = [] # Database
D_Capacity = 10000
k = 0 # Counter
l = 1
T_ER = 50 # Number of steps before updating
N_ER = 1 # Number of trajectory updates
tau = 0.25

for n in range(episodes):
    epsilon = 0.15 - (n * 0.00015) 
    totalreward, iters = Deep_Q_learning_ER(env, model, epsilon, gamma, method, T_ER, N_ER, D, D_Capacity, k, l, tau)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n + 1, "iters", iters, "total reward:", totalreward, "epsilon:", epsilon, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
    if n % 50 == 0:
        model.model.save('race-car.h5')

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

#plot_reward_per_episode(totalrewards)
plot_running_avg(totalrewards)

totalrewards_Qlearning_ER_Samples = totalrewards

N = len(totalrewards_Qlearning_ER_Samples)
running_avg_Qlearning_ER_Samples = np.empty(N)
  
for t in range(N):
    running_avg_Qlearning_ER_Samples[t] = totalrewards_Qlearning_ER_Samples[max(0, t-100):(t+1)].mean()

import numpy
numpy.where(running_avg_Qlearning_ER_Trajectories == max(running_avg_Qlearning_ER_Trajectories))



