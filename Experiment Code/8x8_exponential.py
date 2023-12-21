"""
Frozen Lake (8x8, exponential decay)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from environment import FrozenLake_8x8
from tqdm import tqdm

# Initialize the FrozenLake environment
environment = FrozenLake_8x8()

# Initialize Q-table
num_states = 8 * 8  # 64 states in an 8x8 grid
num_actions = 4     # 4 actions (LEFT, DOWN, RIGHT, UP)
qtable = np.zeros((num_states, num_actions))  # Initialize with zeros

# Hyperparameters
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
episodes = 250000      # Total number of episodes
epsilon = 1            # Initial exploration rate
decay = np.power(0.001, (1 / episodes))  # Exponential Decay

# Function to convert state coordinates to a single number
def state_to_number(state):
    return state[0] * 8 + state[1]

# Training loop
outcome = []  # To record the outcome (success or failure) of each episode
epsilon_total = []  # To track the decay of epsilon

for episode in tqdm(range(episodes), desc="Training Progress"):
    state = environment.reset()
    state_number = state_to_number(state)
    done = False
    epsilon_total.append(epsilon)
    outcome.append("Failure")

    while not done:
        # Exploration or Exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice([0, 1, 2, 3])  # Exploration: choose a random action
        else:
            action = np.argmax(qtable[state_number])  # Exploitation: choose the best action from Q-table

        # Take the action
        new_state, reward, done = environment.step(action)
        new_state_number = state_to_number(new_state)

        # Q-table update
        qtable[state_number, action] = qtable[state_number, action] + alpha * (
            reward + gamma * np.max(qtable[new_state_number]) - qtable[state_number, action])

        state_number = new_state_number  # Update the state number

        if reward == 1:  # Check for success
            outcome[-1] = "Success"

    # Decay epsilon
    epsilon = max(0.001, epsilon * decay)


# Plotting the results
print('===========================================')
print('Q-table after training:')
print(qtable)


# Draw Plot
# Q-table after training
plt.figure(figsize=(5, 32))
sns.heatmap(qtable,  cmap="YlGnBu", annot=True, cbar=False)
plt.xlabel("Action")
plt.ylabel("State")
plt.show()

# Outcome of each episode

# Aggregating outcome data for efficient plotting
chunk_size = 1000  # Aggregate every 1000 episodes
chunks = episodes // chunk_size
success_count = [np.sum(np.array(outcome[i*chunk_size:(i+1)*chunk_size]) == "Success") for i in range(chunks)]
failure_count = [chunk_size - success for success in success_count]

# Plotting aggregated outcomes
plt.figure(figsize=(12, 5))
plt.bar(range(chunks), success_count, color="green", label="Success")
plt.bar(range(chunks), failure_count, bottom=success_count, color="red", label="Failure")
plt.xlabel("Chunk of 1000 Episodes")
plt.ylabel("Count")
plt.title("Aggregated Outcome of Episodes")
plt.legend()
plt.show()

# Epsilon decay along episode
plt.figure(figsize=(12, 5))
plt.plot(epsilon_total, color="#1f2f87")
plt.title("ε Decay")
plt.xlabel("Episode")
plt.ylabel("ε")
plt.show()

# Function to print the Frozen Lake grid with the current position of the agent
def print_grid(state, grid):
    grid_with_agent = np.copy(grid)
    grid_with_agent[state] = 'A'  # Mark the agent's current position with 'A'
    print("\n".join(["".join(row) for row in grid_with_agent]))
    print()

# Simulate one episode
state = environment.reset()
done = False
action_list = []  # To store the sequence of actions

print("Starting Episode...")
print_grid(state, environment.grid)

while not done:
    state_number = state_to_number(state)
    # Choose the best action from Q-table (exploitation)
    action = np.argmax(qtable[state_number])
    new_state, reward, done = environment.step(action)

    # Record the action
    action_list.append(action)

    # Print the updated grid
    print_grid(new_state, environment.grid)

    # Update the state
    state = new_state

action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
action_list_named = [action_names[a] for a in action_list]

print(action_list_named)
