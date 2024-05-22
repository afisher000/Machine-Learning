# %%
import gym
from gym import spaces
import numpy as np

class Game2048Env(gym.Env):
    def __init__(self, size=4):
        super(Game2048Env, self).__init__()

        # Define the size of the 2048 grid
        self.size = size

        # Define the action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=2048, shape=(size, size), dtype=int)

        # Initialize the game state
        self.state = np.zeros((size, size), dtype=int)

        # Start the game with two random tiles
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        # Add a random tile (2 or 4) to an empty spot on the board
        empty_cells = np.argwhere(self.state == 0)
        if len(empty_cells) > 0:
            location = empty_cells[np.random.choice(len(empty_cells))]
            self.state[location[0], location[1]] = 2 if np.random.random()<0.9 else 4

    def slide_row(self, row):
        """
        Slide the non-zero elements of a row to the left and merge identical adjacent tiles.
        """
        new_row = np.zeros_like(row)
        merged = False
        index = 0

        for value in row:
            if value != 0:
                if not merged and index > 0 and new_row[index - 1] == value:
                    # Merge identical adjacent tiles
                    new_row[index - 1] *= 2
                    merged = True
                else:
                    new_row[index] = value
                    index += 1
                    merged = False

        return new_row


    def slide(self, action):
        new_state = np.copy(self.state)

        # Perform the specified action (0: up, 1: down, 2: left, 3: right)
        if action == 0:  # Up
            new_state = np.transpose(new_state)
            new_state = np.array([self.slide_row(row) for row in new_state])
            new_state = np.transpose(new_state)
        elif action == 1:  # Down
            new_state = np.transpose(new_state)
            new_state = np.array([np.flip(self.slide_row(np.flip(row))) for row in new_state])
            new_state = np.transpose(new_state)
        elif action == 2:  # Left
            new_state = np.array([self.slide_row(row) for row in new_state])
        elif action == 3:  # Right
            new_state = np.array([np.flip(self.slide_row(np.flip(row))) for row in new_state])
        return new_state

    def step(self, action):

        # Save the current state for later comparison
        new_state = self.slide(action)

        # Define reward
        reward = np.sum(self.state!=0) - np.sum(new_state!=0)
        self.state = new_state

        # Add a random tile to the board
        self.add_random_tile()

        # Check for termination conditions
        done = self.is_game_over()

        return self.state, reward, done, {}

    def is_game_over(self):
        # Check if the game is over (no more valid moves)
        for action in range(4):
            if not np.array_equal(self.slide(action),self.state):
                return False
        return True

    def reset(self):
        # Reset the game state to the initial configuration
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        return self.state

    def render(self, mode='human'):
        # Render the current state of the game (optional)
        pass

# %%
