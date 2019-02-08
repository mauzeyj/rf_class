import numpy as np
from gym import spaces
from gym.utils import seeding


class matching_data():
    def __init__(self):
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=0, high=334, dtype=np.int, shape=(20, 2))
        self.state = None
        self.counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = matching_data.create_table(self)
        # self.state = matching_data.score_table(self)
        self.counter = 0
        return self.state

    def create_table(self):
        a_one = np.random.randint(111, 333, 17)
        a_two = np.copy(a_one)
        np.random.shuffle(a_two)
        b_one = np.random.randint(111, 333, 3)
        b_two = np.copy(b_one)
        table_a = np.array([a_one, a_two]).T
        table_b = np.array([b_one, b_two]).T
        table = np.concatenate([table_a, table_b])
        np.random.shuffle(table)
        self.state = table
        return self.state

    def score_table(self):
        table = self.state[:, 0:2]
        score = table[:, 0] == table[:, 1]
        score = np.reshape(score, (20, 1))
        # scored_table = np.concatenate([table, score], axis = 1)
        # self.state = scored_table
        return score

    def move_value(self, target):
        table = self.state
        to_move = np.copy(table[target, 1])
        top = np.copy(table[0, 1])
        table[target, 1] = top
        table[0, 1] = to_move
        self.state = table
        return self.state

    def shift_table(self):
        self.state = np.roll(self.state, 1, axis=0)
        return self.state

    def step(self, action):  # TODO try not shifting table.  just swaping two values
        self.counter += 1
        score = np.sum(matching_data.score_table(self))
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if action == 0:
            self.state = matching_data.shift_table(self)
            reward = 0
            done = False
        else:
            self.state = matching_data.move_value(self, action - 1)
            # self.state = matching_data.score_table(self)
            reward = np.diff([score, np.sum(matching_data.score_table(self))])[0]
            if np.sum(matching_data.score_table(self)) == 7:
                done = True
                reward = 10
            else:
                done = False
        if self.counter > 1000:
            done = True

        return np.array(self.state), reward, done, {}
