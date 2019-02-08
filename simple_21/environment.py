import numpy as np
from gym import spaces
from gym.utils import seeding


class twenty_one():
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0, -10]), np.array([31, 31]), dtype=np.float32)
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array(
            [twenty_one.card(self), twenty_one.card(self)])  # dealer player reward done info
        return self.state

    def card(self):
        return np.random.randint(1, 11)

    def color(self):
        n = np.random.rand()
        if n <= .333333:
            return 'r'
        else:
            return 'b'

    def draw(self):
        if twenty_one.color(self) == 'r':
            c = twenty_one.card(self) * -1
        else:
            c = twenty_one.card(self)
        return c

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        dealer, player = state
        reward = 0
        done = False
        if action == 1:
            player = player + twenty_one.draw(self)
            if player > 21:
                reward = -1
                done = True
            if player < 1:
                reward = -1
                done = True
        if action == 0:
            while dealer < 17:
                dealer = dealer + twenty_one.draw(self)  # TODO should be card
            if dealer > 21:
                reward = 1
            elif dealer == player:
                reward = 0
            elif dealer > player:
                reward = -1
            elif player > dealer:
                reward = 1
            done = True
        self.state = np.array([dealer, player])
        return np.array(self.state), reward, done, {}
