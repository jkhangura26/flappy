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

def calculate_reward(done, bird_y, bird_size, score, pipes, bird_x, high_score, SCREEN_HEIGHT, pipe_width, bird_velocity):
    reward = 0.0
    
    # Primary survival reward (dense reward)
    reward += 0.2  # Reduced from 1.0 to prevent reward hacking
    
    # Terminal state handling
    if done:
        # Split penalty into collision types
        if bird_y <= 0 or bird_y >= SCREEN_HEIGHT - bird_size:
            return -150.0  # Boundary collision
        return -200.0  # Pipe collision
    
    # Pipe passage reward (optimized detection)
    next_pipe = next((p for p in sorted(pipes, key=lambda x: x.x) 
                     if p.x + pipe_width > bird_x), None)
    
    if next_pipe:
        vertical_distance = abs(bird_y - (next_pipe.y + next_pipe.height/2))
        reward += max(1.0 - vertical_distance/100, 0)  # [0, 1] reward
        
        # Direct pipe passage bonus
        if next_pipe.x + pipe_width < bird_x:
            reward += 2.0  # Successful passage
    
    # Positional rewards (smooth gradient)
    ideal_y = SCREEN_HEIGHT * 0.4
    y_distance = abs(bird_y - ideal_y)
    reward += max(1.0 - y_distance/200, 0)  # [0, 1] reward
    
    # Velocity-based reward (encourage controlled movement)
    if -8 < bird_velocity < 5:  # Good velocity range
        reward += 0.1
    
    # Progressive difficulty scaling
    difficulty_bonus = min(score / 50, 2.0)  # Max +2 bonus at 100 pipes
    reward += difficulty_bonus
    
    # Exploration bonus (early training)
    if high_score < 10:
        reward += 0.05 * (10 - high_score)
    
    return min(max(reward, -1.0), 5.0)  # Clip rewards to [-1, 5]
