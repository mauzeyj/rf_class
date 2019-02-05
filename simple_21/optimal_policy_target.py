import pandas as pd

from simple_21.environment import twenty_one

data = pd.read_csv('./simple_21/monte_carlo.csv')
data.drop("Unnamed: 0", axis=1, inplace=True)

episode_reward = []
episode_steps = []

for x in range(10000):
    env = twenty_one.reset()
    step = 0
    while done != False:
        step += 1
        state = env.state
        dealer, player = state
        if data[['stick', 'hit']].where((data['dealer'] == dealer) & (data['player'] == 11)).dropna().idxmax(
                axis=1) == 'hit':
            action = 1
        if data[['stick', 'hit']].where((data['dealer'] == dealer) & (data['player'] == 11)).dropna().idxmax(
                axis=1) == 'stick':
            action = 0
        result = env.step(action)
        state, reward, done, meta = result
    episode_reward.append(reward)
    episode_steps.append(step)
