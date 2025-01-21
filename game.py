import pygame
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
bird_y = SCREEN_HEIGHT // 2  # Initialize bird_y globally
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

def run_game(agent_id, epsilon_value):
    global bird_y, bird_velocity, pipes, score, high_score  # Ensure score is managed properly

    pygame.init()

    # Create a new display window for each agent
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Flappy Bird - Agent {agent_id}")
    
    clock = pygame.time.Clock()

    # Initialize score and high_score for this agent
    score = 0
    high_score = 0  # Reset high score for each agent

    # Game loop
    running = True
    last_pipe_time = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Run AI logic for the agent
        state = get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Exploration or Exploitation based on epsilon
        if random.random() < epsilon_value:
            action = random.randint(0, 1)  # Exploration
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()  # Exploitation

        if action == 1:  # Jump
            bird_velocity = jump_strength

        bird_velocity += gravity
        bird_y += bird_velocity
        bird_rect = pygame.Rect(bird_x, bird_y, bird_size, bird_size)

        # Manage pipes
        current_time = pygame.time.get_ticks()
        time_since_last_pipe = current_time - last_pipe_time

        if time_since_last_pipe > pipe_spawn_time:
            pipes.extend(create_pipe())  # Create new pipes
            last_pipe_time = current_time

        # Move pipes
        for pipe in pipes:
            pipe.x += pipe_velocity

        pipes = [pipe for pipe in pipes if pipe.x + pipe_width > 0]

        # Check collisions
        done = check_collision(bird_rect, pipes)

        # Calculate reward and store transition in replay buffer
        next_state = get_state()
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
        epsilon_value = max(epsilon_value * epsilon_decay, epsilon_min)

        # Increment score for passing pipes
        for pipe in pipes:
            if pipe.x + pipe_width == bird_x:  # Bird passes the pipe
                score += 0.5  # Increment score (you can adjust the increment as needed)

        # Display the game and update score
        screen.fill(BLUE)
        pygame.draw.rect(screen, RED, bird_rect)
        for pipe in pipes:
            pygame.draw.rect(screen, GREEN, pipe)

        # Update and display score and high_score for the current agent
        score_text = font.render(f"Score: {int(score)}", True, WHITE)
        high_score_text = font.render(f"High Score: {int(high_score)}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(high_score_text, (10, 40))

        pygame.display.flip()
        clock.tick(FPS)

        # If done (game over), reset the game for this agent
        if done:
            if score > high_score:
                high_score = score  # Update high_score if needed
            reset_game()  # This will reset score to 0




if __name__ == "__main__":
    agents = 3  # Define the number of agents
    epsilon_value = epsilon  # Get the epsilon value from ai.py

    # Run each agent in a separate game loop
    for agent_id in range(agents):
        run_game(agent_id, epsilon_value)
