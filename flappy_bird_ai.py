import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLUE = (135, 206, 250)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

# Bird settings
bird_size = 20
bird_x = 50
bird_y = SCREEN_HEIGHT // 2
bird_velocity = 0
gravity = 0.5
jump_strength = -10

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -5
pipes = []
pipe_spawn_time = 1500  # Milliseconds
last_pipe_time = pygame.time.get_ticks()

# Scoring
score = 0
high_score = 0
font = pygame.font.SysFont(None, 36)

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

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_capacity)

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

# Function to display the restart screen
def show_restart_screen():
    global high_score, score
    if score > high_score:
        high_score = score
    
    screen.fill(BLUE)
    game_over_text = font.render("Game Over!", True, RED)
    score_text = font.render(f"Score: {int(score)}", True, WHITE)
    high_score_text = font.render(f"High Score: {int(high_score)}", True, WHITE)
    restart_text = font.render("Press R to Restart or Q to Quit", True, WHITE)

    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 70))
    screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2 - 20))
    screen.blit(high_score_text, (SCREEN_WIDTH // 2 - high_score_text.get_width() // 2, SCREEN_HEIGHT // 2 + 20))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 70))

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    waiting = False
                    reset_game()
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # AI decision-making
    state = get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    if random.random() < epsilon:
        action = random.randint(0, action_dim - 1)  # Exploration
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

    # Spawn pipes
    current_time = pygame.time.get_ticks()
    if current_time - last_pipe_time > pipe_spawn_time:
        pipes.extend(create_pipe())
        last_pipe_time = current_time

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
    # Reward adjustments
    reward = 1.0 if not done else -100.0

    # Penalize for hitting the top or bottom of the screen more harshly
    if bird_y <= 0 or bird_y >= SCREEN_HEIGHT - bird_size:
        reward -= 20.0  # Increase penalty for collision with top/bottom

    # Give a small positive reward for surviving a frame
    if high_score == 0:
        reward += 0.1 


    # Reward for passing pipes
    for pipe in pipes:
        if pipe.x + pipe_width == bird_x:  # Bird just passed a pipe
            reward += 1.0  # Reward for passing a pipe

    # Extra reward for navigating through the gap successfully
    # Example: If bird is close to the gap and has a good chance of passing it safely
    # Track when the score increases and reward accordingly
    last_score = score
    for pipe in pipes:
        if pipe.x + pipe_width == bird_x:  # Bird just passed a pipe
            score += 0.5  # Increment score for passing pipes
            if score > last_score:  # Check if score has increased
                reward += 1.0  # Reward for increasing score

    # Use score as a dynamic reward to encourage staying alive longer
    reward += score * 0.1  # Reward based on the score progression

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

        # Log training progress periodically
        if current_time % 1000 == 0:
            print(f"Loss: {loss.item():.4f}, Epsilon: {epsilon:.4f}")

    # Update target network periodically
    if current_time % 5000 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Render
    screen.fill(BLUE)  # Background color
    pygame.draw.rect(screen, RED, bird_rect)  # Bird

    for pipe in pipes:
        pygame.draw.rect(screen, GREEN, pipe)  # Pipes

    # Display score
    score_text = font.render(f"Score: {int(score)}", True, WHITE)
    high_score_text = font.render(f"High Score: {int(high_score)}", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(high_score_text, (10, 40))

    pygame.display.flip()
    clock.tick(FPS)

    if done:
        # show_restart_screen()
        if score > high_score:
            high_score = score
        reset_game()
