import sys
import random
import numpy as np
import torch
from config import *
from ai import policy_net, target_net, optimizer, replay_buffer, epsilon, epsilon_decay, epsilon_min, gamma, batch_size, ReplayBuffer, DQN
import torch.nn as nn
from ai import calculate_reward

# Game-specific variables
bird_size = 20
bird_x = 50
bird_y = SCREEN_HEIGHT // 2
bird_velocity = 0
gravity = 0.5
jump_strength = -10

pipe_width = 50
pipe_gap = 150
pipe_velocity = -5
pipes = []
pipe_spawn_time = SPAWN_TIME_PIPE
last_pipe_time = pygame.time.get_ticks()

score = 0
high_score = 0
death_count = 0  # Track number of deaths

# Function to get the current state
def get_state():
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
            bird_velocity / 10.0,
            top_pipe.x / SCREEN_WIDTH,
            top_pipe.height / SCREEN_HEIGHT,
            bottom_pipe.top / SCREEN_HEIGHT,
            pipe_distance / SCREEN_WIDTH,
            pipe_velocity / 10.0
        ])
    else:
        return np.array([bird_y / SCREEN_HEIGHT, bird_velocity / 10.0, 1.0, 0.0, 1.0, 1.0, pipe_velocity / 10.0])

# Function to create new pipes
def create_pipe():
    pipe_height = random.randint(100, SCREEN_HEIGHT - pipe_gap - 100)
    top_pipe = pygame.Rect(SCREEN_WIDTH, 0, pipe_width, pipe_height)
    bottom_pipe = pygame.Rect(SCREEN_WIDTH, pipe_height + pipe_gap, pipe_width, SCREEN_HEIGHT - pipe_height - pipe_gap)
    return top_pipe, bottom_pipe

# Function to check collisions
def check_collision(bird_rect, pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return True
    if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
        return True
    return False

# Function to reset the game
def reset_game():
    global bird_y, bird_velocity, pipes, score, last_pipe_time
    bird_y = SCREEN_HEIGHT // 2
    bird_velocity = 0
    pipes = []
    score = 0
    last_pipe_time = pygame.time.get_ticks()

# Game loop
running = True
clock = pygame.time.Clock()

# Track last spawn time
last_pipe_time = pygame.time.get_ticks()

while running:
    # AI decision-making
    state = get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    if random.random() < epsilon:
        action = random.randint(0, 1)  # Exploration
    else:
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()  # Exploitation

    # Apply action
    if action == 1:  # Jump
        bird_velocity = jump_strength

    # Update bird position
    bird_velocity += gravity
    bird_y += bird_velocity
    bird_rect = pygame.Rect(bird_x, bird_y, bird_size, bird_size)

    # Spawn pipes based on FPS
    current_time = pygame.time.get_ticks()

    # Time difference from the last spawn
    time_since_last_pipe = current_time - last_pipe_time

    # Spawn pipes based on the calculated pipe_spawn_time interval
    if time_since_last_pipe > pipe_spawn_time:
        pipes.extend(create_pipe())  # Create new pipes
        last_pipe_time = current_time  # Update the last spawn time

    # Move pipes
    for pipe in pipes:
        pipe.x += pipe_velocity

    # Remove off-screen pipes and update score
    pipes = [pipe for pipe in pipes if pipe.x + pipe_width > 0]
    for pipe in pipes:
        if pipe.x + pipe_width == bird_x:
            score += 0.5  # Increment by 0.5 for each pipe passed (top and bottom)

    # Check for collisions
    done = check_collision(bird_rect, pipes)

    # Next state
    next_state = get_state()

    # Store transition in replay buffer
    reward = calculate_reward(done, bird_y, bird_size, score, pipes, bird_x, high_score, SCREEN_HEIGHT, pipe_width)
    replay_buffer.push(state, action, reward, next_state, done)

    # Update Q-network
    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network periodically
    if current_time % TARGET_NETWORK_UPDATE_TIME == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if done:
        if score > high_score:
            high_score = score
        death_count += 1  # Increment death count
        print(f"Game Over! Score: {int(score)} | High Score: {int(high_score)} | Deaths: {death_count}")
        reset_game()
