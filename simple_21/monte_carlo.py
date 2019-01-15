# %%
import numpy as np

from simple_21.environment import step

states = []


def step_size(states, actions):
        return 1 / states


for x in range(20):
        state = [0, 0, 'other', np.nan]
        state = step(state)
        while state[2] != 'done':
                if np.random.random() > .5:
                        action = 'hit'
                else:
                        action = 'stick'
                print(action)
                state[2] = action
                state = step(state)
                states.append(state.append(action))
                print(state)
