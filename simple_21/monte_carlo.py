# %%
import numpy as np

from simple_21.environment import step


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
        return 'hit'
    else:
        return 'stick'


states = []  # should be state_actions?
action_rewards = []  # should be rewards?

# does state exist
# actions/rewards list should have [stick_reward, hit_reward, evaluations]
for x in range(50000):
        state = [0, 0, 'other', np.nan]
        state = step(state)
        while state[2] != 'done':
            while state[1] < 10:
                state[2] = 'hit'
                state = step(state)
            if state[:2] in states:
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

# %%

import pandas as pd

action_df = pd.DataFrame(action_rewards, columns=['stick', 'hit', 'visits'])
state_df = pd.DataFrame(states, columns=['dealer', 'player'])
df = pd.concat([action_df, state_df], axis=1)
