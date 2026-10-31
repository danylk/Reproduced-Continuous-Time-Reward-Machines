import numpy as np
from GridWorld import GridWorld

class CopCarEnv:
    def __init__(self, rows, cols, probability):
        self.initstate = (0, 0, 0, 0, 0)  # Starting at (0,0) with atomic propositions siren and theif set to false
        self.state = self.initstate
        self.actions = [0, 1, 2, 3]  # 0:north, 1:east, 2: west, 3:south
        # self.actionmapping = {'north': 0, 'east': 1, 'west': 2, 'south': 3}
        self.sirenpos = (0, rows - 1)  # position of siren
        self.theifpos = (cols - 1, rows - 1)  # position of theif
        self.startpos = (0, 0)
        self.time = 0
        self.env = GridWorld(rows, cols, probability)
        self.statesize = 4
        self.actionsize = 4
        self.states = rows * cols  # total number of states
        self.probability = probability

    def next_states(self, s):  # provides the set of next states for value iteration
        next_states = self.env.next_states((s[0], s[1]))
        nstates = []
        for nstate in next_states:
            x, y = nstate
            if (x, y) == self.sirenpos:
                siren = 1
            else:
                siren = 0
            if (x, y) == self.theifpos:
                theif = 1
            else:
                theif = 0
            if (x, y) == self.startpos:
                start = 1
            else:
                start = 0
            state = (x, y, siren, theif, start)
            nstates.append(state)

        return nstates

    def step(self, action, rate):
        x, y, siren, thief, start = self.state
        x1, y1, time = self.env.step((x, y), action, rate)
        if (x1, y1) == self.sirenpos:
            siren = 1
        else:
            siren = 0
        if (x1, y1) == self.theifpos:
            theif = 1
        else:
            theif = 0
        if (x1, y1) == self.startpos:
            start = 1
        else:
            start = 0
        self.state = (x1, y1, siren, theif, start)
        self.time = self.time + time
        return self.state, time

    def next_state(self, state, action):
        next_state = {}
        next_state[state] = 1 - self.probability
        x, y, siren, thief, start = state
        x1, y1 = self.env.next_state((x, y), action)
        if (x1, y1) == self.sirenpos:
            siren = 1
        else:
            siren = 0
        if (x1, y1) == self.theifpos:
            theif = 1
        else:
            theif = 0
        if (x1, y1) == self.startpos:
            start = 1
        else:
            start = 0
        state = (x1, y1, siren, theif, start)
        next_state[state] = self.probability
        return next_state

    def reset(self):
        self.state = self.initstate  # Reset to initial state
        self.time = 0
        return self.state