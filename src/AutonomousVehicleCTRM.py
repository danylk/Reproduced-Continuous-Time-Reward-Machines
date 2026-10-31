import numpy as np


class AutonomousVehicleCTRM:
    def __init__(self, grid_size=11):
        self.states = [0, 1, 2]  # q0, q1, q2
        self.actions = [0, 1, 2, 3]  # 0:north, 1:east, 2:west, 3:south
        self.totalstates = 3
        self.initstate = 0  # Start in q0 (no package)
        self.state = self.initstate
        self.grid_size = grid_size

        # Generate base rates for the gridworld
        self.base_rates = self.generate_base_rates()

        # f1: Normal traffic rates (for q0 and q2)
        self.function1 = self.create_rate_function(self.base_rates, multiplier=1.0)

        # f2: Emergency delivery rates (for q1 - with package, faster traversal)
        # Emergency vehicles with sirens can traverse faster (10x multiplier)
        self.function2 = self.create_rate_function(self.base_rates, multiplier=10.0)

    def generate_base_rates(self):
        base_rates = {}
        np.random.seed(42)  # For reproducibility

        # Define special locations
        package_location = (10, 0)  # Bottom-right
        hospital_location = (10, 10)  # Top-right

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Calculate distance to package and hospital
                dist_to_package = abs(row - package_location[0]) + abs(col - package_location[1])
                dist_to_hospital = abs(row - hospital_location[0]) + abs(col - hospital_location[1])

                # Base rate depends on proximity to congested areas
                # Closer to package or hospital = more congestion = lower rate
                min_dist = min(dist_to_package, dist_to_hospital)

                if min_dist <= 2:
                    # Heavy traffic near package depot and hospital
                    base_rate = np.random.uniform(0.03, 0.06)
                elif min_dist <= 4:
                    # Moderate traffic
                    base_rate = np.random.uniform(0.06, 0.12)
                else:
                    # Light traffic in outer areas
                    base_rate = np.random.uniform(0.12, 0.20)

                base_rates[(row, col)] = round(base_rate, 3)

        return base_rates

    def create_rate_function(self, base_rates, multiplier=1.0):
        rate_function = {}

        for (row, col), base_rate in base_rates.items():
            for action in self.actions:
                # Action-dependent rate variation
                # Some actions might be slightly faster/slower depending on traffic patterns
                action_factor = self.get_action_factor(action)
                final_rate = base_rate * multiplier * action_factor
                rate_function[(row, col, action)] = round(final_rate, 3)

        return rate_function

    def get_action_factor(self, action):
        # 0:north, 1:east, 2:west, 3:south
        factors = {
            0: 1.0,  # North - neutral
            1: 1.1,  # East - slightly faster (main traffic flow)
            2: 0.9,  # West - slightly slower (against main flow)
            3: 1.0  # South - neutral
        }
        return factors.get(action, 1.0)

    def transitionfunction(self, input_state):
        x, y, has_package, at_hospital = input_state

        if self.state == 0:  # q0: Without package
            if has_package:
                self.state = 1  # Picked up package
                return 0
            else:
                self.state = 0  # Still searching for package
                return 0

        elif self.state == 1:  # q1: With package
            if at_hospital:
                self.state = 2  # Delivered to hospital
                return 1  # Positive reward for task completion
            else:
                self.state = 1  # Still delivering
                return 0

        elif self.state == 2:  # q2: Task completed (accepting state)
            return None  # Terminal state

    def transition_function_counterfactual(self, ctrmstate, input_state):
        x, y, has_package, at_hospital = input_state

        if ctrmstate == 0:  # q0: Without package
            if has_package:
                next_state = 1
                return 0, next_state
            else:
                next_state = 0
                return 0, next_state

        elif ctrmstate == 1:  # q1: With package
            if at_hospital:
                next_state = 2
                return 1, next_state  # Reward for completion
            else:
                next_state = 1
                return 0, next_state

        elif ctrmstate == 2:  # q2: Terminal state
            return None, None

    def get_rate_counterfactual(self, ctrmstate, input_state, action):
        x, y, has_package, at_hospital = input_state
        state_key = (x, y, action)

        if ctrmstate == 0 or ctrmstate == 2:
            # q0 (without package) and q2 (completed) use normal rates (f1)
            return self.function1.get(state_key, 0.1)  # Default if not found
        elif ctrmstate == 1:
            # q1 (with package - emergency delivery) uses faster rates (f2)
            return self.function2.get(state_key, 1.0)  # Default if not found

    def get_rate(self, input_state, action):
        x, y, has_package, at_hospital = input_state
        state_key = (x, y, action)

        if self.state == 0 or self.state == 2:
            # Normal traffic rates
            return self.function1.get(state_key, 0.1)
        elif self.state == 1:
            # Emergency delivery rates (faster)
            return self.function2.get(state_key, 1.0)
        else:
            return None

    def transition_VI(self, ctrmstate, next_state):
        if ctrmstate == 0:  # q0
            return 0  # No reward for picking up package or staying
        elif ctrmstate == 1:  # q1
            if next_state == 2:
                return 1  # Reward for completing delivery
            else:
                return 0
        elif ctrmstate == 2:  # q2
            return None  # Terminal state
        else:
            return 0

    def next_states(self, ctrmstate):
        if ctrmstate == 0:  # q0
            return [0, 1]  # Can stay in q0 or move to q1 (pick up package)
        elif ctrmstate == 1:  # q1
            return [1, 2]  # Can stay in q1 or move to q2 (deliver)
        elif ctrmstate == 2:  # q2
            return None  # Terminal state
        else:
            return None

    def reset(self):
        """Reset CTRM to initial state."""
        self.state = self.initstate
        return self.state

    def get_rate_description(self, x, y):
        avg_rate = np.mean([self.function1[(x, y, a)] for a in self.actions])

        if avg_rate < 0.06:
            return 'heavy'
        elif avg_rate < 0.12:
            return 'moderate'
        else:
            return 'light'