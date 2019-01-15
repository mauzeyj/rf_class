import numpy as np


def card():
    return np.random.randint(1, 11)


def color():
    n = np.random.rand()
    if n <= .333333:
        return 'r'
    else:
        return 'b'


def draw():
    if color() == 'r':
        c = card() * -1
    else:
        c = card()
    return c


def step(state):
    dealer = state[0]
    player = state[1]
    action = state[2]
    score = state[3]
    if action == 'other':
        dealer = card()
        player = card()
        action = 'start'
    if action == 'hit':
        player = player + draw()
        if player > 21:
            score = -1
            action = 'done'
    if action == 'stick':
        while dealer < 17:
            dealer = dealer + draw()
        if dealer > 21:
            score = 1
        if dealer == player:
            score = 0
        if dealer > player:
            score = -1
        if player > dealer:
            score = 1
        action = 'done'
    return [dealer, player, action, score]
