import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

Axes3D
def discount(rate, t):
    return rate ** (t - 1)


def lambdo(lam, t):
    return lam ** t


def gamma(gam, t):
    return gam ** t


def diff(R, gamma, q_prime, q):
    return R + (gamma * (q_prime - q))


def hist_loop(s_hist, a_hist):


"""
loop

gamma g = discount rate 0-1
alpha a = learning rate 0-1
lambda l = decay backward pass (discount?) 0-1
epsilon e = exploration/exploitation
elegibility trade E

Build list of lists with action value pairs.
When reward is earned back propagate the reward 
based on lambda steps with the discount. Then take 
those rewards and update the primary rewards data.

diff = R + g *Q(S',A') - Q(S,A)
add one to count
loop over current history
    Q(s,a) <- Q(s,a) + a * diff * e(s,a)
    E(s,a) <- g * l * e(s,a)
S <- S'; A <- A'
until S is terminal

This is basically monte carlo but lambda and discount are applied when calculating values 
"""

# g =
# d =
# l =


states = []
action_rewards = []

# for x in range(5):
#         state = [0, 0, 'other', np.nan]
#         state = step(state)
#         current_states_index = []
#         current_actions = []
#         while state[2] != 'done':
#
#             while state[1] < 10:
#                 state[2] = 'hit'
#                 state = step(state)
#             if state[:2] in states:
#                 # previous_state = state[2:]
#                 index = states.index(state[:2])
#                 current_states_index.append(index)
#                 actions = action_rewards[index]
#                 previous_rewards = action_rewards[index]
#                 if egreedy_exploration(previous_rewards[2]) < np.random.random():
#                     previous_action = random_action()
#                     state[2] = previous_action
#                     current_actions.append(previous_action)
#                 elif previous_rewards[0] == previous_rewards[1]:
#                     previous_action = random_action()
#                     state[2] = previous_action
#                     current_actions.append(previous_action)
#                 else:
#                     if previous_rewards[0] > previous_rewards[1]:
#                         previous_action = 'stick'
#                         state[2] = previous_action
#                         current_actions.append(previous_action)
#                     else:
#                         previous_action = 'hit'
#                         state[2] = previous_action
#                         current_actions.append(previous_action)
#                 state = step(state)
#                 # if state[2] == 'done': # this is the point to start incrementing mean
#                 for cur_state in range(len(current_states_index)):
#                     if previous_action == 'stick':
#                         action_rewards[current_states_index[cur_state]][0] = incremental_mean(state[3],
#                                                                                               action_rewards[
#                                                                                                   current_states_index[
#                                                                                                       cur_state]][
#                                                                                                   0],
#                                                                                               action_rewards[
#                                                                                                   current_states_index[
#                                                                                                       cur_state]][
#                                                                                                   2])
#                     elif previous_action == 'hit':
#                         action_rewards[current_states_index[cur_state]][1] = incremental_mean(state[3],
#                                                                                               action_rewards[
#                                                                                                   current_states_index[
#                                                                                                       cur_state]][
#                                                                                                   1],
#                                                                                               action_rewards[
#                                                                                                   current_states_index[
#                                                                                                       cur_state]][
#                                                                                                   2])
#                     action_rewards[current_states_index[cur_state]][2] = action_rewards[index][2] + 1
#             else:
#                 states.append(state[:2])
#                 action_rewards.append([0, 0, 1])

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
