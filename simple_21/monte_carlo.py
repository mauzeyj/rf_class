# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from simple_21.environment import twenty_one

Axes3D
def incremental_mean(x, u, k):
    """
    calculates incremental mean
    :param x: current reward
    :param u: mean of existing rewards
    :param k: number of times state action pair has been seen
    :return: new state action value
    """
    return u + ((1 / k) * (x - u))


def egreedy_exploration(k, k_fixed=100):
    return k_fixed / (k_fixed + k)


def random_action():
    if np.random.random() > .5:
        return 1
    else:
        return 0


states = []
action_rewards = []

# TODO why are my numbers shifted lower.  Expect 21 to have argmax of around 1
env = twenty_one()

for x in range(5000000):
    state = env.reset()
        current_states_index = []
        current_actions = []
    done = False
    while done != True:
        state = list(env.state)
        if state in states:
            index = states.index(state)
                current_states_index.append(index)
                actions = action_rewards[index]
                previous_rewards = action_rewards[index]
                if egreedy_exploration(previous_rewards[2]) < np.random.random():
                    action = random_action()
                    current_actions.append(action)
                elif previous_rewards[0] == previous_rewards[1]:
                    action = random_action()
                    current_actions.append(action)
                else:
                    if previous_rewards[0] > previous_rewards[1]:
                        action = 0

                        current_actions.append(action)
                    else:
                        action = 1
                        current_actions.append(action)
            state, reward, done, meta = env.step(action)
        else:
            states.append(state[:2])
            action_rewards.append([0, 0, 1])

        for cur_state in range(len(current_states_index)):
            if action == 0:
                action_rewards[current_states_index[cur_state]][0] = incremental_mean(reward,
                                                                                      action_rewards[
                                                                                          current_states_index[
                                                                                              cur_state]][
                                                                                          0],
                                                                                      action_rewards[
                                                                                          current_states_index[
                                                                                              cur_state]][
                                                                                          2])
            elif action == 1:
                action_rewards[current_states_index[cur_state]][1] = incremental_mean(reward,
                                                                                      action_rewards[
                                                                                          current_states_index[
                                                                                              cur_state]][
                                                                                          1],
                                                                                      action_rewards[
                                                                                          current_states_index[
                                                                                              cur_state]][
                                                                                          2])
            action_rewards[current_states_index[cur_state]][2] = action_rewards[index][2] + 1


# %%


action_df = pd.DataFrame(action_rewards, columns=['stick', 'hit', 'visits'])
state_df = pd.DataFrame(states, columns=['dealer', 'player'])
df = pd.concat([action_df, state_df], axis=1)
df['max'] = df[['hit', 'stick']].max(axis=1)
df.to_csv('monte_carlo.csv')

# %%


plt.style.use('fivethirtyeight')

fig = plt.figure(figsize=(14, 8))

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('dealer')
ax.set_ylabel('player')
ax.set_zlabel('value')
ax.set_ylim(12, 21)
ax.view_init(25, -35)
surf = ax.plot_trisurf(df['dealer'], df['player'], df['max'], cmap=cm.coolwarm)
fig.colorbar(surf)
plt.show()
