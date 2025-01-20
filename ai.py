import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

# Neural Network for AI
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)

# AI settings
state_dim = 7  # Bird's normalized state (y, velocity, nearest pipe info, distance to pipes, speed)
action_dim = 2  # Actions: [Do nothing, Jump]
learning_rate = 1e-3
epsilon = 0.5  # Start exploration at 50%
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.99  # Discount factor
batch_size = 64
buffer_capacity = 10000

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_capacity)

# Reward Configuration
def calculate_reward(done, bird_y, bird_size, score, pipes, bird_x, high_score, SCREEN_HEIGHT, pipe_width):
    reward = 1.0 if not done else -100.0  # Negative reward for game over

    # Penalize for hitting the top or bottom of the screen
    if bird_y <= 0 or bird_y >= SCREEN_HEIGHT - bird_size:
        reward -= 20.0  # Increase penalty for collision with top/bottom

    # Reward for surviving a frame
    reward += 0.1

    # Reward for passing pipes (each pipe pair passed)
    for pipe in pipes:
        if pipe.x + pipe_width == bird_x:  # Bird just passed a pipe
            reward += 1.0  # Reward for passing a pipe

    # Reward for staying alive longer (based on score progression)
    reward += score * 0.1  # Increase reward based on score

    # Reward for navigating through the gap successfully
    last_score = score
    for pipe in pipes:
        if pipe.x + pipe_width == bird_x:  # Bird just passed a pipe
            score += 0.5  # Increment score for passing pipes
            if score > last_score:  # Check if score has increased
                reward += 2.0  # Reward for increasing score (extra for successful gap navigation)

    # Encourage survival for longer by rewarding staying alive
    if high_score == 0:
        reward += 0.1  # Small positive reward for staying alive in the beginning

    # Give larger rewards for staying alive and avoiding immediate danger
    if bird_y > 0 and bird_y < SCREEN_HEIGHT - bird_size:
        reward += 0.5  # Reward for staying in the middle portion of the screen

    # Encourage movement towards pipe gaps (avoiding top and bottom)
    for pipe in pipes:
        if pipe.top <= bird_y <= pipe.bottom:
            reward += 0.5  # Reward for being in the safe gap area

    return reward
