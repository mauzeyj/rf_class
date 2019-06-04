# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/

# %%
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from simple_21.environment import twenty_one
import pandas as pd
import matplotlib.pyplot as plt
import gym

env = gym.make('Blackjack-v0')
# todo put in a class

# env = twenty_one()

#todo add memory replay
# todo add query environment for observation and action space
env.observation_space
# create network
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(3,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


num_episodes = 50000
y = 0.95  # discount rate
eps = 0.5  # egreedy
decay_factor = 0.999
r_avg_list = []
target_list = []
rewards = []
for i in range(num_episodes):

    # reset environment to start episode
    s = env.reset()

    # calulate current exploit vs explore rate
    eps *= decay_factor

    # print every 100 episode results
    if i % 100 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))

    # set done to false because episode just started
    done = False

    r_sum = 0
    target_sum = 0

    # episode loop
    while not done:

        # do we explore randomly or use the models prediction, save action in a
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(np.array(s).reshape(-1, 3)))

        # pass action a to environment and get new state, reward, done, info
        new_s, r, done, info = env.step(a)

        # pass new state to model to get maximum predicted value multiply by discount and add reward
        target = r + y * np.max(model.predict(np.array(new_s).reshape(-1, 3)))

        # get the prediction for the initial state to pass to model with action values of that state
        # plus update to value of action actually taken
        target_vec = model.predict(np.array(s).reshape(-1, 3))[0]

        # reward for taking the action
        target_vec[a] = target

        # pass for one epoc training to model, state, and the updated rewards vector
        model.fit(np.array(s).reshape(-1, 3), np.array(target_vec).reshape(-1, 2), epochs=1, verbose=0)

        # move new state to old state for next loop
        s = new_s

        # keeps track of rewards over episodes but doesn't seem to get used
        # use this in a visualization   e.g. table of rewards over periods?
        r_sum += r

        # tracks how high the action rewards are for an episode
        target_sum += target

    # adds the target sum to list for visualization.
    # I expect that this is to show stabalization or learning?
    target_list.append(target_sum)

    # Works for adding a reward at the end of an episode if that is the only place you recieve one
    # todo when moving into class keep track of each episodes total rewards
    rewards.append(r)


plt.plot(target_list)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.title('{} episodes'.format(num_episodes))
plt.show()
