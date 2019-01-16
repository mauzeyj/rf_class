# %%
import numpy as np

from simple_21.environment import step

states = []
actions = []


def incremental_mean(x, u, k):
    """
    calculates incremental mean
    :param x: current reward
    :param u: mean of existing rewards
    :param k: number of times state action pair has been seen
    :return: new state action value
    """
    return u + ((1 / k) * (x - u))


# %%
def egreedy_exploration(k, k_fixed=100):
    return k_fixed / (k_fixed + k)


# %%

# does state exist
# actions list should have (stick, reward, evaluations), (hit, reward, evaluations)
for x in range(20):
        state = [0, 0, 'other', np.nan]
        state = step(state)
        while state[2] != 'done':
                if np.random.random() > .5:
                        action = 'hit'
                else:
                        action = 'stick'
                state[2] = action
                state = step(state)
                states.append(state)
                actions.append(action)
                print(state)
