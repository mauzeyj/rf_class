import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from simple_21.environment import step


Axes3D
"""
loop

gamma g = discount rate 0-1
alpha a = learning rate 0-1
lambda l = decay backward pass (discount?) 0-1
epsilon e = exploration/exploitation
elegibility trace E

Build list of lists with action value pairs.
When reward is earned back propagate the reward 
based on lambda steps with the discount. Then take 
those rewards and update the primary rewards data.

Initialize Q(s,a) arbitrarily, for all s in S, a in A(s)   -
Repeat (for each episode)                                  -
    E(s,a) = 0, for all s in S, a in A(s)                  -
    Initialize S, A                                        -
    Repeat (for each step of episode):
        Take action A, observe R, S'                       -
        Choose A' from S' using policy derived from Q (e.g., e-greedy)  -
        diff = R + g * Q(S',A') - Q(S,A)                   -
        E(S,A) <- E(S,A) + 1                               -
        For all s in S, a in A(s):
            Q(s,a) <- Q(s,a) + a * diff * E(s,a)           -
            E(s,a) <- g * l * E(s,a)                       
S <- S'; A <- A'
until S is terminal

This is basically monte carlo but lambda and discount are applied when calculating values 
"""


def diff(R, gamma, q_prime, q):
    return R + (gamma * q_prime) - q


def initialize():
    state = [np.random.randint(1, 10), np.random.randint(10), 0, np.nan]
    return state


def egreedy_exploration(k, k_fixed=100):
    return k_fixed / (k_fixed + k)


def e_greed(state, previous_state):
    if egreedy_exploration(
            e_greedy[previous_state[0] - 1, previous_state[1] - 1, previous_state[2]]) < np.random.random():
        action = random_action()
    else:
        action = np.argmax([previous_state[0] - 1, previous_state[1] - 1])
    e_greedy[previous_state[0] - 1, previous_state[1] - 1, previous_state[2]] += 1
    return action


def random_action():
    if np.random.random() > .5:
        return 1
    else:
        return 0


def get_difference(state, previous_state):
    # diff = R + g * Q(S',A') - Q(S,A)
    if state[3] in [-1, 0, 1] and (state[1] < 0 or state[1] > 21):
        d = diff(state[3], g, np.max(Q[previous_state[0] - 1, previous_state[1] - 1]),
                 Q[previous_state[0] - 1, previous_state[1] - 1, previous_state[2]])
    elif state[3] in [-1, 0, 1]:
        d = diff(state[3], g, np.max(Q[state[0] - 1, state[1] - 1]),
                 Q[previous_state[0] - 1, previous_state[1] - 1, previous_state[2]])
    else:
        d = diff(np.max(Q[state[0] - 1, state[1] - 1]), g, np.max(Q[state[0] - 1, state[1] - 1]),
                 Q[previous_state[0] - 1, previous_state[1] - 1, previous_state[2]])
    E[previous_state[0] - 1, previous_state[0] - 1, previous_state[2]] += 1
    return d


g = .9
d = .9
l = .9
a = .9

Q = np.zeros([10, 21, 2])  # dealer, player, (hit, stick)
e_greedy = np.zeros([10, 21, 2])
for x in range(1000):
    state = initialize()
    states = []
    E = np.zeros([10, 21, 2])  # dealer, player, (hit, stick)
    while state[2] != 2:
        previous_state = state
        states.append(previous_state)
        state = step(state)
        # E-greedy
        current_action = e_greed(state, previous_state)
        state[2] == current_action
        state = step(state)
        # Get difference for back prop
        d = get_difference(state, previous_state)
        # Back prop
        # Q(s,a) <- Q(s,a) + a * diff * E(s,a)
        for x in states:
            Q[x[0] - 1, x[1] - 1, x[2]] = Q[x[0] - 1, x[1] - 1, x[2]] + ((a * d) * E[x[0] - 1, x[0] - 1, x[2]])
            E[x[0] - 1, x[0] - 1, x[2]] = (g * l * E[x[0] - 1, x[0] - 1, x[2]])
    if state[2] == 2:
        # st = state
        # st[2] = previous_state[2]
        # states.append(st)
        d = diff(state[3], g, np.max(Q[previous_state[0] - 1, previous_state[1] - 1]),
                 Q[previous_state[0] - 1, previous_state[1] - 1, previous_state[2]])
        for x in states:
            Q[x[0] - 1, x[1] - 1, x[2]] = Q[x[0] - 1, x[1] - 1, x[2]] + ((a * d) * E[x[0] - 1, x[0] - 1, x[2]])


# %%
# TODO visualize the Q max surface

# action_df = pd.DataFrame(action_rewards, columns=['stick', 'hit', 'visits'])
# state_df = pd.DataFrame(states, columns=['dealer', 'player'])
# df = pd.concat([action_df, state_df], axis=1)
# df['max'] = df[['hit', 'stick']].max(axis=1)
# df.to_csv('monte_carlo.csv')

# %%


# plt.style.use('fivethirtyeight')
#
# fig = plt.figure(figsize=(14, 8))
#
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('dealer')
# ax.set_ylabel('player')
# ax.set_zlabel('value')
# ax.set_ylim(12, 21)
# ax.view_init(25, -35)
# surf = ax.plot_trisurf(df['dealer'], df['player'], df['max'], cmap=cm.coolwarm)
# fig.colorbar(surf)
# plt.show()
