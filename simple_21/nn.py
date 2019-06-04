# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/

# %%
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from simple_21.environment import twenty_one
import pandas as pd
import matplotlib.pyplot as plt

env = twenty_one()
#todo add memory replay

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(2,)))  # todo change to relu
model.add(Dense(100, activation='relu'))  # todo change to relu
model.add(Dense(2, activation='softmax'))  # todo change to softmax
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

num_episodes = 50000
y = 0.95  # discount rate
eps = 0.5  # egreedy
decay_factor = 0.999
r_avg_list = []
target_list = []
rewards = []
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    if i % 100 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    target_sum = 0
    while not done:
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(s.reshape(-1, 2)))
        new_s, r, done, info = env.step(a)
        target = r + y * np.max(model.predict(new_s.reshape(-1, 2)))
        target_vec = model.predict(s.reshape(-1, 2))[0]
        target_vec[a] = target
        model.fit(s.reshape(-1, 2), target_vec.reshape(-1, 2), epochs=1, verbose=0)
        s = new_s
        r_sum += r
        target_sum += target
    target_list.append(target_sum)
    r_avg_list.append(r_sum / 1000)
    rewards.append(r)

plt.plot(target_list)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.title('{} episodes'.format(num_episodes))
plt.show()
