# Frozen Lake (8x8, linear decay)

# Actions: LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3

# There are two basic steps.
# The first one is action. We use a random number to decide exploration or exploitation.
#       If random number < epsilon, exploration.
#       Else, if random number > epsilon, exportation.
# The second step is updating the Q-table by this equation.
# We keep repeating these steps until the agent falls into the hole or reaches the goal.
#       When it happens, we just jump to the next episode.

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time

# The environment (playground) is pre-defined "FrozenLake-v1" in gym
environment = gym.make("FrozenLake8x8-v1", is_slippery=False)

# Initialize Q-table
num_states = environment.observation_space.n  # 64
num_actions = environment.action_space.n  # 4
qtable = np.zeros((num_states, num_actions))  # all zero
print('Q-table before training:')
print(qtable)

# Hyper parameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
episodes = 250000
epsilon = 1
# Calculate decay rate by episodes
epsilon_decay = 1/episodes  # Linear Decay

# List of outcomes to plot
outcomes = []
plt.rcParams['figure.dpi'] = 300
epsilon_total = []

# Training
for e in range(episodes):
    # print("episode: {}".format(e))

    state = environment.reset()
    done = False
    epsilon_total.append(epsilon)

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Exploration-Exploitation Tradeoff
        # Generate a random number [0.0, 1.0)
        rand = np.random.random()
        # If random number < epsilon, take a random action
        if rand < epsilon:
            action = environment.action_space.sample()
        # Else, choose max Q(s,a)
        else:
            action = np.argmax(qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = environment.step(action)

        # Update new Q(s,a)
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"

    # Update epsilon
    epsilon -= epsilon_decay

print('===========================================')
print('Q-table after training:')
print(qtable)


# Draw Plot
# Q-table after training
plt.figure(figsize=(5, 32))
sns.heatmap(qtable,  cmap="YlGnBu", annot=True, cbar=False);
plt.xlabel("Action")
plt.ylabel("State")
plt.show()

# Outcome of each episode
plt.figure(figsize=(12, 5))
plt.xlabel("Episode")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#1f2f87", width=1.0)
plt.show()

# Epsilon decay along episode
plt.figure(figsize=(12, 5))
plt.title("ε Decay")
plt.xlabel("Episode")
plt.ylabel("ε")
plt.bar(range(len(epsilon_total)), epsilon_total, color="#1f2f87", width=1.0)
plt.show()


# Visualization
state = environment.reset()
done = False
action_list = []
environment.render()

while not done:
    # Choose the action with max Q(s,a)
    if np.max(qtable[state]) > 0:
        action = np.argmax(qtable[state])
    # If there's no best action (only zeros), take a random one
    else:
        action = environment.action_space.sample()

    # Add the action to the sequence
    action_list.append(action)

    # Implement this action and move the agent in the desired direction
    new_state, reward, done, info = environment.step(action)

    # Update our current state
    state = new_state

    # Update the render
    clear_output(wait=True)
    environment.render()
    time.sleep(0.5)

print(f"Action List = {action_list}")
