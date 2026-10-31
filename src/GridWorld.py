import numpy as np

class GridWorld:
    def __init__(self, rows, columns, probability):
        self.initial_state = (0, 0)
        self.rows = rows
        self.cols = columns
        self.prob = probability
        self.actions = [0, 1, 2, 3]  # 0: North, 1: East, 2: West, 3: South
        self.time = 0

    def move(self, start, action):
        x, y = start

        if action == 0:
            new_x, new_y = x, y + 1
        elif action == 1:
            new_x, new_y = x + 1, y
        elif action == 2:
            new_x, new_y = x - 1, y
        elif action == 3:
            new_x, new_y = x, y - 1
        else:
            return start

        if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
            return (new_x, new_y)
        else:
            return start

    def step(self, state, action, rate):
        x, y = state
        tau = np.random.exponential(scale=1 / rate) # Sampled time when it is not specified

        if np.random.rand() >= (1 - self.prob):
            x1, y1 = self.move((x, y), action)
            return x1, y1, tau
        else:
            return x, y, tau

    def next_states(self, state):
        next_states = []
        for action in self.actions:
            next_states.append(self.move(state, action))
        return next_states

    def reset(self):
        self.time = 0
        return self.initial_state

    def next_state(self, state, action):
        return self.move(state, action)