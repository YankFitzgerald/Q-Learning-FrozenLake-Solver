"""
Actions: LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3

"""

import numpy as np


# Define the FrozenLake environment
class FrozenLake_4x4:
    def __init__(self):
        self.grid = np.array([
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ])
        self.state = (0, 0)  # Start at the top-left corner
        self.done = False
        self.reward = 0

    def reset(self):
        self.state = (0, 0)
        self.done = False
        self.reward = 0
        return self.state

    def step(self, action):
        # Define action effects
        actions = {
            0: (0, -1),  # LEFT
            1: (1, 0),  # DOWN
            2: (0, 1),  # RIGHT
            3: (-1, 0)  # UP
        }

        # Calculate new state
        new_state = (self.state[0] + actions[action][0], self.state[1] + actions[action][1])

        # Check if the new state is within the grid boundaries
        if 0 <= new_state[0] < 4 and 0 <= new_state[1] < 4:
            self.state = new_state

        # Check for holes or goal
        cell = self.grid[self.state]
        if cell == 'H':
            self.done = True
            self.reward = 0
        elif cell == 'G':
            self.done = True
            self.reward = 1
        else:
            self.reward = 0

        return self.state, self.reward, self.done


# Define the FrozenLake environment
class FrozenLake_8x8:
    def __init__(self):
        self.grid = np.array([
            ['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'H'],
            ['F', 'F', 'F', 'F', 'H', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'F', 'F', 'F', 'F', 'G']
        ])
        self.state = (0, 0)  # Start at the top-left corner
        self.done = False
        self.reward = 0

    def reset(self):
        self.state = (0, 0)
        self.done = False
        self.reward = 0
        return self.state

    def step(self, action):
        # Define action effects
        actions = {
            0: (0, -1),  # LEFT
            1: (1, 0),  # DOWN
            2: (0, 1),  # RIGHT
            3: (-1, 0)  # UP
        }

        # Calculate new state
        new_state = (self.state[0] + actions[action][0], self.state[1] + actions[action][1])

        # Check if the new state is within the grid boundaries
        if 0 <= new_state[0] < 8 and 0 <= new_state[1] < 8:
            self.state = new_state

        # Check for holes or goal
        cell = self.grid[self.state]
        if cell == 'H':
            self.done = True
            self.reward = 0
        elif cell == 'G':
            self.done = True
            self.reward = 1
        else:
            self.reward = 0

        return self.state, self.reward, self.done
