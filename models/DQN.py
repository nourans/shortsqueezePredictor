import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the DQN Model
class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
INPUT_SIZE = 2  # time and sentiment
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3  # actions: predict up, down, or no change
GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 1000
EPISODES = 1000

# Replay Memory
memory = deque(maxlen=MEMORY_SIZE)

# Initialize model, optimizer, and loss function
model = DQNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Function to choose an action (epsilon-greedy)
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, OUTPUT_SIZE - 1)  # Explore
    with torch.no_grad():
        q_values = model(torch.tensor(state, dtype=torch.float32))
    return torch.argmax(q_values).item()  # Exploit

# Function to train the model
def train():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Current Q-values
    q_values = model(states).gather(1, actions).squeeze()

    # Target Q-values
    with torch.no_grad():
        max_next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * GAMMA * max_next_q_values

    # Compute loss and optimize
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Simulated environment
def simulate_environment(sentiment, price):
    time_steps = len(sentiment)
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(EPISODES):
        state = [0, sentiment[0]]  # Start state (time=0, sentiment=first value)
        total_reward = 0

        for t in range(1, time_steps):
            action = choose_action(state, epsilon)
            next_state = [t, sentiment[t]]

            # Calculate reward
            actual_change = price[t] - price[t - 1]
            predicted_change = [-1, 0, 1][action]  # Map action to change
            reward = -abs(actual_change - predicted_change)  # Negative for large errors

            # Check if the episode is done
            done = (t == time_steps - 1)

            # Store transition in memory
            memory.append((state, action, reward, next_state, done))

            # Update state
            state = next_state
            total_reward += reward

            # Train the model
            train()

        # Decay exploration rate
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")

# Example Data
time = np.arange(0, 100)
sentiment = np.sin(time / 10) + np.random.normal(scale=0.1, size=100)  # Simulated sentiment
price = np.cumsum(np.random.choice([-1, 1], size=100) * (1 + 0.1 * sentiment))  # Simulated price

simulate_environment(sentiment, price)
