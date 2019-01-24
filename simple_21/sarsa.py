# TODO Video 5 45:00  has psuedo code for sarsa and 1:05:10 for sarsa lambda

import numpy as np

from simple_21.monte_carlo import step, incremental_mean, egreedy_exploration, random_action


def discount(rate, t):
    return rate ** (t - 1)


def lambdo(lam, t):
    return lam ** t


def gamma(gam=.9):
    return gam


def diff(R, gamma, q_prime, q):
    return R + (gamma * (q_prime - q))


"""
loop

gamma g = discount rate 0-1
alpha a = learning rate 0-1
lambda l = decay backward pass (discount?) 0-1
epsilon e = exploration/exploitation

Build list of lists with action value pairs.
When reward is earned back propagate the reward 
based on lambda steps with the discount. Then take 
those rewards and update the primary rewards data.

diff = R + g *Q(S',A') - Q(S,A)
add one to count
loop over current hitory
    Q(s,a) <- Q(s,a) + a * diff * e(s,a)
    E(s,a) <- g * l * e(s,a)
S <- S'; A <- A'
until S is terminal

This is basically monte carlo but lambda and discount are applied when calculating values 
"""

states = []
action_rewards = []

for x in range(10):
    current_states = []
    current_action_rewards = []
    state = [0, 0, 'other', np.nan]
    state = step(state)
    while state[2] != 'done':
        while state[1] < 10:
            state[2] = 'hit'
            state = step(state)

            if state[:2] in states:  # this finishes out the episode before updating -- sarsa should update at each flip
                # previous_state = state[2:]
                index = states.index(state[:2])
                actions = action_rewards[index]
                previous_rewards = action_rewards[index]
                if egreedy_exploration(previous_rewards[2]) < np.random.random():
                    previous_action = random_action()
                    state[2] = previous_action
                elif previous_rewards[0] == previous_rewards[1]:
                    previous_action = random_action()
                    state[2] = previous_action
                else:
                    if previous_rewards[0] > previous_rewards[1]:
                        previous_action = 'stick'
                        state[2] = previous_action
                    else:
                        previous_action = 'hit'
                        state[2] = previous_action
                state = step(state)
                if state[2] == 'done':
                    # if state[:2] in states:
                    #     index = states.index(state[:2])
                    if previous_action == 'stick':
                        action_rewards[index][0] = incremental_mean(state[3],
                                                                    previous_rewards[0],
                                                                    previous_rewards[2])
                    elif previous_action == 'hit':
                        action_rewards[index][1] = incremental_mean(state[3],
                                                                    previous_rewards[1],
                                                                    previous_rewards[2])
                    action_rewards[index][2] = action_rewards[index][2] + 1
            else:
                states.append(state[:2])
                action_rewards.append([0, 0, 1])
