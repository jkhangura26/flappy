import sys
import random
import numpy as np
import torch
from config import *
from ai import policy_net, target_net, optimizer, replay_buffer, epsilon, epsilon_decay, epsilon_min, gamma, batch_size, ReplayBuffer, DQN
import torch.nn as nn
from ai import calculate_reward

# Define the buffer_capacity if not defined in config.py
buffer_capacity = 1000000  # Add this line to define buffer capacity

# Game-specific variables
bird_size = 20
bird_x = 50
gravity = 0.5
jump_strength = -10

pipe_width = 50
pipe_gap = 150
pipe_velocity = -5
pipe_spawn_time = SPAWN_TIME_PIPE

# Number of agents to run in parallel
num_agents = 10


# Initialize death_count variable outside the game loop
death_count = 0

# List of agents
agents = []

# Initialize agents (Each agent has its own DQN, replay buffer, and variables)
for i in range(num_agents):
    agent = {
        'id': i,
        'bird_y': SCREEN_HEIGHT // 2,
        'bird_velocity': 0,
        'pipes': [],
        'score': 0,
        'high_score': 0,  # Initialize individual high score for each agent
        'last_pipe_time': pygame.time.get_ticks(),
        'epsilon': epsilon,
        'policy_net': DQN(7, 2),  # Each agent has its own DQN for individual experience
        'target_net': DQN(7, 2),
        'replay_buffer': ReplayBuffer(buffer_capacity),
    }
    agent['target_net'].load_state_dict(agent['policy_net'].state_dict())
    agent['target_net'].eval()
    agents.append(agent)

# Function to get the current state for a single agent
def get_state(agent):
    bird_y = agent['bird_y']
    pipes = agent['pipes']
    bird_x = 50  # All agents share same X position
    nearest_pipe = None
    for pipe in pipes:
        if pipe.x + pipe_width > bird_x:
            nearest_pipe = pipe
            break

    if nearest_pipe:
        next_pipe_index = pipes.index(nearest_pipe)
        top_pipe = pipes[next_pipe_index]
        bottom_pipe = pipes[next_pipe_index + 1]
        pipe_distance = top_pipe.x - bird_x
        return np.array([
            bird_y / SCREEN_HEIGHT,
            agent['bird_velocity'] / 10.0,
            top_pipe.x / SCREEN_WIDTH,
            top_pipe.height / SCREEN_HEIGHT,
            bottom_pipe.top / SCREEN_HEIGHT,
            pipe_distance / SCREEN_WIDTH,
            pipe_velocity / 10.0
        ])
    else:
        return np.array([bird_y / SCREEN_HEIGHT, agent['bird_velocity'] / 10.0, 1.0, 0.0, 1.0, 1.0, pipe_velocity / 10.0])

# Function to create new pipes
def create_pipe():
    pipe_height = random.randint(100, SCREEN_HEIGHT - pipe_gap - 100)
    top_pipe = pygame.Rect(SCREEN_WIDTH, 0, pipe_width, pipe_height)
    bottom_pipe = pygame.Rect(SCREEN_WIDTH, pipe_height + pipe_gap, pipe_width, SCREEN_HEIGHT - pipe_height - pipe_gap)
    return top_pipe, bottom_pipe

# Function to check collisions for a single agent
def check_collision(agent, bird_rect):
    for pipe in agent['pipes']:
        if bird_rect.colliderect(pipe):
            return True
    if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
        return True
    return False

# Function to reset the game for a single agent
def reset_game(agent):
    agent['bird_y'] = SCREEN_HEIGHT // 2
    agent['bird_velocity'] = 0
    agent['pipes'] = []
    agent['score'] = 0
    agent['last_pipe_time'] = pygame.time.get_ticks()

# Game loop
running = True
clock = pygame.time.Clock()

# Track last spawn time
last_pipe_time = pygame.time.get_ticks()

while running:
    current_time = pygame.time.get_ticks()

    for agent in agents:
        # AI decision-making for each agent
        state = get_state(agent)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if random.random() < agent['epsilon']:
            action = random.randint(0, 1)  # Exploration
        else:
            with torch.no_grad():
                q_values = agent['policy_net'](state_tensor)
                action = torch.argmax(q_values).item()  # Exploitation

        # Apply action for the agent
        if action == 1:  # Jump
            agent['bird_velocity'] = jump_strength

        # Update bird position for the agent
        agent['bird_velocity'] += gravity
        agent['bird_y'] += agent['bird_velocity']
        bird_rect = pygame.Rect(bird_x, agent['bird_y'], bird_size, bird_size)

        # Spawn pipes based on FPS
        time_since_last_pipe = current_time - agent['last_pipe_time']
        if time_since_last_pipe > pipe_spawn_time:
            agent['pipes'].extend(create_pipe())  # Create new pipes
            agent['last_pipe_time'] = current_time  # Update the last spawn time

        # Move pipes
        for pipe in agent['pipes']:
            pipe.x += pipe_velocity

        # Remove off-screen pipes and update score
        agent['pipes'] = [pipe for pipe in agent['pipes'] if pipe.x + pipe_width > 0]
        for pipe in agent['pipes']:
            if pipe.x + pipe_width == bird_x:
                agent['score'] += 0.5  # Increment by 0.5 for each pipe passed (top and bottom)

        # Check for collisions
        done = check_collision(agent, bird_rect)

        # Next state for agent
        next_state = get_state(agent)

        # Store transition in replay buffer
        reward = calculate_reward(done, agent['bird_y'], bird_size, agent['score'], agent['pipes'], bird_x, agent['high_score'], SCREEN_HEIGHT, pipe_width)
        agent['replay_buffer'].push(state, action, reward, next_state, done)

        # Update Q-network for the agent
        if len(agent['replay_buffer']) > batch_size:
            states, actions, rewards, next_states, dones = agent['replay_buffer'].sample(batch_size)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = agent['policy_net'](states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q_values = agent['target_net'](next_states).max(1)[0]
                target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        if current_time % TARGET_NETWORK_UPDATE_TIME == 0:
            agent['target_net'].load_state_dict(agent['policy_net'].state_dict())

        # Decay epsilon for the agent
        agent['epsilon'] = max(agent['epsilon'] * epsilon_decay, epsilon_min)

        # If done, reset the agent
        if done:
            if agent['score'] > agent['high_score']:
                agent['high_score'] = agent['score']  # Update high score for this agent
            death_count += 1  # Increment death count
            print(f"Agent {agent['id']} Game Over! Score: {int(agent['score'])} | High Score: {int(agent['high_score'])} | Deaths: {death_count}")
            reset_game(agent)

    # Shared Training: After all agents have completed an update, synchronize the policy network
    # This step ensures that all agents are using the same policy net after training.
    for agent in agents:
        agent['policy_net'].load_state_dict(policy_net.state_dict())
