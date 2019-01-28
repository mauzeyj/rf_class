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
    """
    states: 0 = hit, 1 = stick, 2 = done, 3 = start
    :param state:
    :return:
    """
    dealer = state[0]
    player = state[1]
    action = state[2]
    score = state[3]
    if action == 3:
        dealer = card()
        player = card()
        action = np.nan
    if action == 0:
        player = player + draw()
        if player > 21:
            score = -1
            action = 2
        if player < 0:
            score = -1
            action = 2
    if action == 1:
        while dealer < 17:
            dealer = dealer + draw()
        if dealer == player:
            score = 0
            action = 2
        elif dealer > 21:
            score = 1
            action = 2
        elif dealer > player:
            score = -1
            action = 2
        elif player > dealer:
            score = 1
            action = 2

    return [dealer, player, action, score]
