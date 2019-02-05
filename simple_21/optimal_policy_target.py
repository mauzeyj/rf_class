import numpy as np
import pandas as pd

from simple_21.environment import twenty_one

data = pd.read_csv('monte_carlo.csv')
data.drop("Unnamed: 0", axis=1, inplace=True)

episode_reward = []
episode_steps = []
env = twenty_one()
for x in range(10000):
    env.reset()
    step = 0
    done = False
    while done != True:
        step += 1
        state = env.state
        dealer, player = state
        if list(data[['stick', 'hit']].where((data['dealer'] == dealer) & (data['player'] == player)).dropna().idxmax(
                axis=1))[0] == 'hit':
            action = 1
        if list(data[['stick', 'hit']].where((data['dealer'] == dealer) & (data['player'] == player)).dropna().idxmax(
                axis=1))[0] == 'stick':
            action = 0
        result = env.step(action)
        state, reward, done, meta = result
    episode_reward.append(reward)
    episode_steps.append(step)

print(pd.crosstab(np.array(episode_reward), columns='count', normalize=True))
