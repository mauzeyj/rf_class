import numpy as np


class twenty_one():
    def __init__(self):
        self.action_space = [1, 0]
        self.state = None

    def reset(self):
        self.state = np.array(
            [twenty_one.card(self), twenty_one.card(self)]), np.nan, False, {}  # dealer player reward done info

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
        state = self.state
        dealer, player = state[0]
        reward = None
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
                dealer = dealer + twenty_one.draw(self)
            if dealer > 21:
                reward = 1
            elif dealer == player:
                reward = 0
            elif dealer > player:
                reward = -1
            elif player > dealer:
                reward = 1
            done = True
        self.state = np.array([dealer, player]), reward, done, {}
        return np.array([dealer, player]), reward, done, {}
