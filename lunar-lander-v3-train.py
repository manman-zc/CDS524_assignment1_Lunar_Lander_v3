import os
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt  # For plotting reward curves
import time  # For tracking training time


# Neural Network Definition for Deep Q-Learning
class Network(nn.Module):
    """Q-Network architecture for Deep Q-Learning"""

    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        # Initialize network layers
        self.seed = torch.manual_seed(seed)  # For reproducibility
        self.fc1 = nn.Linear(state_size, 64)  # Input layer to hidden layer
        self.fc2 = nn.Linear(64, 64)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(64, action_size)  # Hidden layer to output layer

    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))  # First hidden layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second hidden layer with ReLU activation
        return self.fc3(x)  # Output layer (action values)


# Environment Initialization
env = gym.make('LunarLander-v3')
state_size = env.observation_space.shape[0]  # Size of state space (8 dimensions)
number_actions = env.action_space.n  # Number of possible actions (4 actions)

# Hyperparameters Configuration
learning_rate_alpha = 5e-4  # Learning rate for optimizer
minibatch_size = 100  # Batch size for experience replay
discount_factor_gamma = 0.99  # Discount factor for future rewards
experience_replay_buffer_size = 100000  # Capacity of replay buffer
interpolation_parameter_tau = 1e-3  # Soft update interpolation parameter


# Experience Replay Buffer Implementation
class ReplayMemory(object):
    """Experience replay memory for storing and sampling transitions"""

    def __init__(self, capacity_of_memory):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity_of_memory = capacity_of_memory  # Max memory size
        self.memory = []  # Storage for experiences

    def push(self, event):
        """Add new experience to memory"""
        self.memory.append(event)
        if len(self.memory) > self.capacity_of_memory:
            del self.memory[0]  # Remove oldest experience if over capacity

    def sample(self, batch_size):
        """Sample random batch of experiences for learning"""
        experiences = random.sample(self.memory, k=batch_size)
        # Convert experiences to PyTorch tensors and move to device
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones


# DQN Agent Implementation
class Agent():
    """Deep Q-Learning Agent implementation"""

    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size  # Dimension of state space
        self.action_size = action_size  # Number of possible actions

        # Initialize Q-Networks
        self.local_qnetwork = Network(state_size, action_size).to(self.device)  # Primary network
        self.target_qnetwork = Network(state_size, action_size).to(self.device)  # Target network
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate_alpha)

        # Initialize experience replay
        self.memory = ReplayMemory(experience_replay_buffer_size)
        self.t_step = 0  # Counter for learning interval

    def step(self, state, action, reward, next_state, done):
        """Store experience and perform learning at intervals"""
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4  # Update every 4 steps
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor_gamma)

    def act(self, state, epsilon=0.):
        """Select action using ε-greedy policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()  # Set back to training mode

        # ε-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update Q-network using sampled experiences"""
        states, next_states, actions, rewards, dones = experiences

        # Calculate target Q-values using target network
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_targets * (1 - dones))

        # Calculate expected Q-values using local network
        q_expected = self.local_qnetwork(states).gather(1, actions)

        # Calculate and minimize loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter_tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# Training Process Setup
agent = Agent(state_size, number_actions)
number_episodes = 2000  # Total training episodes
maximum_number_timesteps_per_episode = 1000  # Max steps per episode
epsilon_starting_value = 1.0  # Initial exploration rate
epsilon_ending_value = 0.01  # Minimum exploration rate
epsilon_decay_value = 0.995  # Exploration rate decay
epsilon = epsilon_starting_value  # Current exploration rate
scores_on_100_episodes = deque(maxlen=100)  # Rolling average window

# Training Metrics Tracking
episode_rewards = []  # Store rewards for all episodes

# Training Execution with Timing
start_time = time.time()  # Record start time

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0  # Cumulative reward for current episode

    for t in range(maximum_number_timesteps_per_episode):
        # Agent selects and performs action
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Agent learns from experience
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break  # End episode if termination condition met

    # Update tracking metrics
    scores_on_100_episodes.append(score)
    episode_rewards.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)  # Decay exploration rate

    # Progress reporting
    print("\rEpisode: {}\t Average Score: {:.2f}".format(episode, np.mean(scores_on_100_episodes)), end="")
    if episode % 100 == 0:
        print("\rEpisode: {}\t Average Score: {:.2f}".format(episode, np.mean(scores_on_100_episodes)))
    if np.mean(scores_on_100_episodes) >= 200.0:
        print("\nEnvironment Solved in {:d} episodes!\t Average Score: {:.2f}".format(episode - 100,
                                                                                      np.mean(scores_on_100_episodes)))
        torch.save(agent.local_qnetwork.state_dict(), 'lunar_lander.pth')
        break

# Training Completion Metrics
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Training Performance Visualization
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.legend()
plt.grid(True)
plt.savefig('training_reward_curve.png')  # Save visualization
plt.show()