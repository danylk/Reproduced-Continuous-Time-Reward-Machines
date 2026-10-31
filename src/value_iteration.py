import numpy as np
import math
from collections import deque


class ValueIteration:
    def __init__(self, gamma=0.001, environment=None, ctrm=None):
        self.V = {}
        self.gamma = gamma
        self.env = environment
        self.ctrm = ctrm
        self.states = None
        self.ctrm_states = tuple(self.ctrm.states)

    # This function does the value iteration and returns the value of the initial state
    def doVI(self):
        self.states = self.fill_states()  # Gets the set of states from the environment
        self.fill_vtable()  # Fills the v-table with the states of the product
        value = self.value_iteration()
        return value

    # This function returns the set of states of the environment
    # Requirements: - Environment: A function next_states(state) that would provide the next states of the
    # input state
    def fill_states(self):
        initial_state = self.env.initstate
        state_set = {initial_state}  # This set stores all the states seen so far
        # Initialize the queue with the initial state
        queue = deque([initial_state])  # Initializes the queue with the initial state
        while queue:  # Runs until the queue is empty
            current_state = queue.popleft()  # pops the state from the queue
            for next_state in self.env.next_states(current_state):  # Gets the next states of the current state
                if next_state not in state_set:  # Checks if the state is seen or not
                    state_set.add(next_state)  # Adds the unseen state to the state set
                    queue.append(next_state)  # Adds to the queue to check for unseen next states
        return state_set

    # The function fills the V table with the states of the product and gives them initial value 0
    # Requirements: ctrm: A variable states with the set of states
    def fill_vtable(self):
        for state in self.states:
            for ctrm_state in self.ctrm.states:
                s = state + (ctrm_state,)
                self.V[s] = 0

            # Returns the value of the initial state

    def value_iteration(self):
        enable = True
        initstate = self.env.initstate + (self.ctrm.initstate,)  # Initial state of the product
        while enable:
            delta = 0
            for state in self.V:  # Each state is encoded as (environment state, ctrm state)
                v = self.V[state]  # Previous value
                max_value = float('-inf')
                for a in self.env.actions:
                    ctrm_state = state[-1]  # Get the CTRM state
                    env_state = state[:-1]  # Get the environment state
                    rate = self.ctrm.get_rate_counterfactual(ctrm_state, env_state, a)  # Gets the rate
                    next_states = self.env.next_state(env_state, a)  # Gets the next state of the environment

                    for next_state, probability in next_states.items():
                        action_value = 0
                        reward, ctrm_next = self.ctrm.transition_function_counterfactual(ctrm_state, next_state)
                        next_state1 = next_state + (ctrm_next,)
                        if reward is not None and rate is not None:
                            time = 1 / rate
                            action_value += (probability * (reward + rate / (self.gamma + rate) * self.V[next_state1]))
                    if action_value >= max_value:
                        max_value = action_value
                self.V[state] = max_value
                delta = max(delta, abs(v - self.V[state]))
            if delta == 0:
                enable = False
        return self.V[initstate]

    def print_table(self, indent=0, separator='=>'):
        indent_str = ' ' * indent
        for key, value in self.V.items():
            print(f"{indent_str}{key}{separator} {value}")







