import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ----------------------------
# Neural Network for DQN
# ----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a simple feed-forward neural network.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
# Replay Buffer for Experience Replay
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Save a transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        """
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

# ----------------------------
# AI Hyperparameters
# ----------------------------
state_dim = 7        # Dimensions of the state vector
action_dim = 2       # [Do nothing, Jump]
learning_rate = 1e-3
epsilon = 0.5        # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.99         # Discount factor
batch_size = 64
buffer_capacity = 10000

# Initialize networks and optimizer
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set target network to evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_capacity)

def calculate_reward(done, bird_y, bird_size, score, pipes, bird_x, high_score,
                     screen_height, pipe_width, bird_velocity):
    """
    Compute the reward for a given state transition.
    
    Args:
        done (bool): Whether the episode has terminated.
        bird_y (float): Vertical position of the bird.
        bird_size (int): Size of the bird.
        score (float): Current score.
        pipes (list): List of active pipes.
        bird_x (float): Horizontal position of the bird.
        high_score (float): Highest score achieved.
        screen_height (int): Height of the game screen.
        pipe_width (int): Width of a pipe.
        bird_velocity (float): Vertical velocity of the bird.
    
    Returns:
        float: The computed reward, clipped between -1 and 5.
    """
    reward = 0.0

    if done:
        # Penalize collisions differently
        if bird_y <= 0 or bird_y >= screen_height - bird_size:
            return -150.0  # Boundary collision
        return -200.0      # Pipe collision

    # Reward for approaching the next pipe
    next_pipe = next((p for p in sorted(pipes, key=lambda x: x.x) if p.x + pipe_width > bird_x), None)
    if next_pipe:
        vertical_distance = abs(bird_y - (next_pipe.y + next_pipe.height / 2))
        reward += max(1.0 - vertical_distance / 100, 0)
        if next_pipe.x + pipe_width < bird_x:
            reward += 2.0  # Bonus for passing the pipe

    # Positional reward encouraging optimal altitude
    ideal_y = screen_height * 0.4
    y_distance = abs(bird_y - ideal_y)
    reward += max(1.0 - y_distance / 200, 0)

    # Encourage smooth flight with controlled velocity
    if -8 < bird_velocity < 5:
        reward += 0.1

    # Scaling bonus based on progression
    difficulty_bonus = min(score / 50, 2.0)
    reward += difficulty_bonus

    # Bonus for exploration in early stages
    if high_score < 10:
        reward += 0.05 * (10 - high_score)

    return min(max(reward, -1.0), 5.0)


def save_checkpoint(path):
    """Saves the current state of the policy network, target network, optimizer, and epsilon."""
    checkpoint = {
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_checkpoint(path):
    """Loads a saved checkpoint and restores the state of the policy network, target network, optimizer, and epsilon."""
    global epsilon
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epsilon = checkpoint['epsilon']
    print(f"Model loaded from {path}")
