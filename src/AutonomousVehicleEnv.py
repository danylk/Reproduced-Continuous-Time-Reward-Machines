import numpy as np
from GridWorld import GridWorld


class AutonomousVehicleEnv:
    def __init__(self, rows=11, cols=11, probability=0.9):
        self.rows = rows
        self.cols = cols
        self.probability = probability

        # Special locations
        self.start_pos = (0, 0)  # Distribution center (bottom-left)
        self.package_pos = (10, 0)  # Medical supply depot (bottom-right corner)
        self.hospital_pos = (10, 10)  # Hospital (top-right corner)

        assert 0 <= self.package_pos[0] < cols and 0 <= self.package_pos[1] < rows, \
            f"Package position {self.package_pos} out of bounds!"
        assert 0 <= self.hospital_pos[0] < cols and 0 <= self.hospital_pos[1] < rows, \
            f"Hospital position {self.hospital_pos} out of bounds!"

        self.initstate = (0, 0, 0, 0)
        self.state = self.initstate

        # Actions: 0:north, 1:east, 2:west, 3:south
        self.actions = [0, 1, 2, 3]

        self.env = GridWorld(rows, cols, probability)

        self.statesize = 4  # (x, y, has_package, at_hospital)
        self.actionsize = 4
        self.states = rows * cols  # Total grid positions

        self.time = 0

    def step(self, action, rate):
        x, y, has_package, at_hospital = self.state

        # Execute movement in gridworld
        x1, y1, sampled_time = self.env.step((x, y), action, rate)

        # Update package status
        if (x1, y1) == self.package_pos and has_package == 0:
            has_package = 1  # Pick up package

        # Update hospital status
        if (x1, y1) == self.hospital_pos:
            at_hospital = 1
        else:
            at_hospital = 0

        # Update state
        self.state = (x1, y1, has_package, at_hospital)
        self.time += sampled_time

        return self.state, sampled_time

    def next_state(self, state, action):
        next_states = {}
        x, y, has_package, at_hospital = state

        # Probability of staying in same position (action fails)
        next_states[state] = 1 - self.probability

        # Get next position from gridworld
        x1, y1 = self.env.next_state((x, y), action)

        # Update package status
        new_has_package = has_package
        if (x1, y1) == self.package_pos and has_package == 0:
            new_has_package = 1  # Pick up package

        # Update hospital status
        if (x1, y1) == self.hospital_pos:
            new_at_hospital = 1
        else:
            new_at_hospital = 0

        # Add successful transition
        next_state = (x1, y1, new_has_package, new_at_hospital)
        next_states[next_state] = self.probability

        return next_states

    def next_states(self, state):
        x, y, has_package, at_hospital = state
        possible_next_states = []

        # Get possible next positions
        next_positions = self.env.next_states((x, y))

        for x1, y1 in next_positions:
            # Update package status
            new_has_package = has_package
            if (x1, y1) == self.package_pos and has_package == 0:
                new_has_package = 1

            # Update hospital status
            if (x1, y1) == self.hospital_pos:
                new_at_hospital = 1
            else:
                new_at_hospital = 0

            next_state = (x1, y1, new_has_package, new_at_hospital)
            possible_next_states.append(next_state)

        return possible_next_states

    def reset(self):
        self.state = self.initstate
        self.time = 0
        return self.state

    def render_grid(self):
        x, y, has_package, at_hospital = self.state

        print("\n" + "=" * 50)
        print("Autonomous Vehicle Delivery - Grid State")
        print("=" * 50)

        for row in range(self.rows - 1, -1, -1):  # Print top to bottom
            line = ""
            for col in range(self.cols):
                if (col, row) == (x, y):
                    if has_package:
                        line += " A* "  # Agent with package
                    else:
                        line += " A  "  # Agent without package
                elif (col, row) == self.start_pos:
                    line += " S  "
                elif (col, row) == self.package_pos:
                    if has_package:
                        line += " .  "  # Package taken
                    else:
                        line += " P  "
                elif (col, row) == self.hospital_pos:
                    line += " H  "
                else:
                    line += " .  "
            print(line)

        print("=" * 50)
        print(f"Position: ({x}, {y})")
        print(f"Has Package: {'Yes' if has_package else 'No'}")
        print(f"At Hospital: {'Yes' if at_hospital else 'No'}")
        print(f"Total Time: {self.time:.2f}")
        print("=" * 50 + "\n")

    def get_state_description(self):
        x, y, has_package, at_hospital = self.state

        location_desc = f"Position ({x}, {y})"

        if (x, y) == self.start_pos:
            location_desc += " [Distribution Center]"
        elif (x, y) == self.package_pos:
            location_desc += " [Package Depot]"
        elif (x, y) == self.hospital_pos:
            location_desc += " [Hospital]"

        package_desc = "Has package" if has_package else "No package"

        return f"{location_desc} | {package_desc}"

    def is_terminal(self):
        x, y, has_package, at_hospital = self.state
        return has_package == 1 and at_hospital == 1