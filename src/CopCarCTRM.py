import numpy as np

class CopCarCTRM:
    def __init__(self):
        self.states = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3]
        self.totalstates = 3
        self.initstate = 0  # initial state
        self.state = self.initstate
        self.function1 = self.generate_rates()
        self.function1 = {position: round(value * 0.03, 3) for position, value in self.function1.items()}
        self.function2 = {position: round(value * 10, 2) for position, value in self.function1.items()}
        # print("New reward machine")

    def generate_rates(self):
        function1 = {
            (0, 0): 0.06, (0, 1): 0.2, (0, 2): 0.03, (0, 3): 0.2, (0, 4): 0.06, (0, 5): 0.06,
            (0, 6): 0.03, (0, 7): 0.06, (0, 8): 0.03, (0, 9): 0.06, (0, 10): 0.06, (0, 11): 0.06, (0, 12): 0.06,
            (1, 0): 0.03, (1, 1): 0.03, (1, 2): 0.03,
            (1, 3): 0.06, (1, 4): 0.2, (1, 5): 0.2, (1, 6): 0.06, (1, 7): 0.03, (1, 8): 0.2, (1, 9): 0.03,
            (1, 10): 0.06, (1, 11): 0.2, (1, 12): 0.2,
            (2, 0): 0.2, (2, 1): 0.06, (2, 2): 0.03, (2, 3): 0.03, (2, 4): 0.06, (2, 5): 0.06, (2, 6): 0.03,
            (2, 7): 0.06, (2, 8): 0.06, (2, 9): 0.03,
            (2, 10): 0.2, (2, 11): 0.06, (2, 12): 0.06, (3, 0): 0.03, (3, 1): 0.03, (3, 2): 0.03, (3, 3): 0.03,
            (3, 4): 0.06, (3, 5): 0.2,
            (3, 6): 0.2, (3, 7): 0.03, (3, 8): 0.06, (3, 9): 0.03, (3, 10): 0.03, (3, 11): 0.2, (3, 12): 0.06,
            (4, 0): 0.2, (4, 1): 0.06,
            (4, 2): 0.2, (4, 3): 0.03, (4, 4): 0.06, (4, 5): 0.03, (4, 6): 0.03, (4, 7): 0.03, (4, 8): 0.06,
            (4, 9): 0.2, (4, 10): 0.2, (4, 11): 0.03,
            (4, 12): 0.2, (5, 0): 0.06, (5, 1): 0.03, (5, 2): 0.06, (5, 3): 0.03, (5, 4): 0.2, (5, 5): 0.03,
            (5, 6): 0.06, (5, 7): 0.03, (5, 8): 0.03,
            (5, 9): 0.2, (5, 10): 0.03, (5, 11): 0.03, (5, 12): 0.06, (6, 0): 0.03, (6, 1): 0.06, (6, 2): 0.2,
            (6, 3): 0.06, (6, 4): 0.06, (6, 5): 0.06, (6, 6): 0.03,
            (6, 7): 0.03, (6, 8): 0.06, (6, 9): 0.03, (6, 10): 0.03, (6, 11): 0.06, (6, 12): 0.2, (7, 0): 0.2,
            (7, 1): 0.06, (7, 2): 0.06, (7, 3): 0.06, (7, 4): 0.2,
            (7, 5): 0.2, (7, 6): 0.06, (7, 7): 0.2, (7, 8): 0.03, (7, 9): 0.06, (7, 10): 0.06, (7, 11): 0.2,
            (7, 12): 0.03, (8, 0): 0.06, (8, 1): 0.2, (8, 2): 0.03,
            (8, 3): 0.2, (8, 4): 0.06, (8, 5): 0.06, (8, 6): 0.2, (8, 7): 0.06, (8, 8): 0.03, (8, 9): 0.2,
            (8, 10): 0.03, (8, 11): 0.06, (8, 12): 0.2, (9, 0): 0.06,
            (9, 1): 0.2, (9, 2): 0.2, (9, 3): 0.2, (9, 4): 0.03, (9, 5): 0.2, (9, 6): 0.03, (9, 7): 0.2, (9, 8): 0.06,
            (9, 9): 0.06, (9, 10): 0.06, (9, 11): 0.03,
            (9, 12): 0.03, (10, 0): 0.2, (10, 1): 0.2, (10, 2): 0.2, (10, 3): 0.06, (10, 4): 0.06, (10, 5): 0.2,
            (10, 6): 0.2, (10, 7): 0.2, (10, 8): 0.03, (10, 9): 0.06,
            (10, 10): 0.2, (10, 11): 0.2, (10, 12): 0.2, (11, 0): 0.06, (11, 1): 0.06, (11, 2): 0.06, (11, 3): 0.2,
            (11, 4): 0.2, (11, 5): 0.06, (11, 6): 0.2, (11, 7): 0.06, (11, 8): 0.2, (11, 9): 0.03, (11, 10): 0.06,
            (11, 11): 0.2,
            (11, 12): 0.03, (12, 0): 0.03, (12, 1): 0.06, (12, 2): 0.2, (12, 3): 0.06, (12, 4): 0.2, (12, 5): 0.06,
            (12, 6): 0.03, (12, 7): 0.06, (12, 8): 0.2, (12, 9): 0.03, (12, 10): 0.03, (12, 11): 0.06, (12, 12): 0.03}
        rates_with_actions = {}
        for (state1, state2), base_value in function1.items():
            for action in self.actions:
                rates_with_actions[(state1, state2, action)] = self.deterministic_transformation(base_value, action)

        return rates_with_actions

    def deterministic_transformation(self, base_value, action):
        return round(base_value * (1 + 0.1 * action), 2)

    def transitionfunction(self, input_state):  # Takes the transition in the reward machine and gives the reward
        if self.state == 0:
            if input_state[2] > 0:
                self.state = 1
                return 0
            else:
                self.state = 0
                return 0
        elif self.state == 1:
            if input_state[3] > 0:
                self.state = 2
                return 0
            else:
                self.state = 1
                return 0

        elif self.state == 2:
            if input_state[4] > 0:
                self.state = 3
                return 1
            else:
                self.state = 2
                return 0
        elif self.state == 3:
            return None

    def transition_function_counterfactual(self, ctrmstate, input_state):
        if ctrmstate == 0:
            if input_state[2] > 0:
                next_state = 1
                return 0, next_state
            else:
                next_state = 0
                return 0, next_state
        elif ctrmstate == 1:
            if input_state[3] > 0:
                next_state = 2
                return 0, next_state
            else:
                next_state = 1
                return 0, next_state

        elif ctrmstate == 2:
            if input_state[4] > 0:
                next_state = 3
                return 1, next_state
            else:
                next_state = 2
                return 0, next_state
        elif ctrmstate == 3:
            return None, None

    def get_rate_counterfactual(self, ctrmstate, input_state, action):
        state = (input_state[0], input_state[1], action)
        if ctrmstate == 0 or ctrmstate == 2:
            return self.function1[state]
        elif ctrmstate == 1:
            return self.function2[state]

    def get_rate(self, input_state, action):
        # print("Input state is:", input_state)
        if self.state == 0 or self.state == 2:
            return self.function1[(input_state[0], input_state[1], action)]
        elif self.state == 1:
            return self.function2[(input_state[0], input_state[1], action)]
        else:
            return None

    def transition_VI(self, ctrmstate, next_state):
        if ctrmstate == 0:
            if next_state == 1:
                return 0
            else:
                return 0
        elif ctrmstate == 1:
            if next_state == 2:
                return 0
            else:
                return 0
        elif ctrmstate == 2:
            if next_state == 3:
                return 1
            else:
                return 0
        elif ctrmstate == 3:
            return None

    def next_states(self, ctrmstate):
        next_states = []
        if ctrmstate == 0:
            next_states.extend([0, 1])
        elif ctrmstate == 1:
            next_states.extend([1, 2])
        elif ctrmstate == 2:
            next_states.extend([2, 3])
        elif ctrmstate == 3:
            next_states = None
        return next_states

    def reset(self):
        self.state = self.initstate  # Reset to initial state
        return self.state

# # Example usage:
# example = RewardMachineCopCarUT()
# print("function1 at (0,3):", example.function2[(0, 3)])
# print("function2 at (0,3):", example.function2[(0, 3)])