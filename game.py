import pygame
import sys
import random
import torch
import numpy as np
from config import *
from ai import policy_net, target_net, optimizer, replay_buffer, epsilon, epsilon_decay, epsilon_min, gamma, batch_size, ReplayBuffer, DQN, calculate_reward
from utils import get_state, check_collision, create_pipe

# Initialize Pygame once
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Multi-Agent")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Global variables
high_score = 0
pipe_spawn_time = SPAWN_TIME_PIPE
pipe_width = PIPE_WIDTH
pipe_velocity = PIPE_VELOCITY
gravity = GRAVITY
jump_strength = JUMP_STRENGTH
bird_size = BIRD_SIZE

# Initialize networks
target_net.load_state_dict(policy_net.state_dict())

def reset_game(agents):
    """Reset all agents to initial state"""
    for agent in agents:
        agent['bird_y'] = SCREEN_HEIGHT // 2
        agent['bird_velocity'] = 0
        agent['pipes'] = []
        agent['score'] = 0
        agent['last_pipe_time'] = pygame.time.get_ticks()
        agent['done'] = False

def show_restart_screen():
    """Display game over screen and handle input"""
    global high_score
    current_score = max(agent['score'] for agent in agents)
    
    if current_score > high_score:
        high_score = current_score

    screen.fill(BLUE)
    texts = [
        ("Game Over!", RED, -70),
        (f"Score: {int(current_score)}", WHITE, -20),
        (f"High Score: {int(high_score)}", WHITE, 20),
        ("Press R to Restart or Q to Quit", WHITE, 70)
    ]

    for text, color, offset in texts:
        rendered = font.render(text, True, color)
        screen.blit(rendered, (SCREEN_WIDTH//2 - rendered.get_width()//2, 
                              SCREEN_HEIGHT//2 + offset))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

def run_game(agents):
    """Main game loop with multiple agents"""
    global high_score, epsilon

    while True:
        # Reset all agents when starting
        reset_game(agents)
        
        # Game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Check if all agents are done
            if all(agent['done'] for agent in agents):
                running = False
                continue

            # Update each agent
            for agent in agents:
                if agent['done']:
                    continue

                # Get current state
                state = get_state(agent['bird_x'], agent['bird_y'], agent['pipes'], 
                                pipe_width, agent['bird_velocity'], pipe_velocity)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # Choose action
                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        action = policy_net(state_tensor).argmax().item()

                # Update bird physics
                if action == 1:
                    agent['bird_velocity'] = jump_strength
                agent['bird_velocity'] += gravity
                agent['bird_y'] += agent['bird_velocity']

                # Pipe management
                current_time = pygame.time.get_ticks()
                if current_time - agent['last_pipe_time'] > pipe_spawn_time:
                    agent['pipes'].extend(create_pipe())
                    agent['last_pipe_time'] = current_time

                # Move pipes
                agent['pipes'] = [pipe for pipe in agent['pipes'] if pipe.x + pipe_width > 0]
                for pipe in agent['pipes']:
                    pipe.x += pipe_velocity

                # Check collisions
                bird_rect = pygame.Rect(agent['bird_x'], agent['bird_y'], bird_size, bird_size)
                agent['done'] = check_collision(bird_rect, agent['pipes'])

                # Store experience
                next_state = get_state(agent['bird_x'], agent['bird_y'], agent['pipes'],
                                      pipe_width, agent['bird_velocity'], pipe_velocity)
                reward = calculate_reward(agent['done'], agent['bird_y'], bird_size,
                                         agent['score'], agent['pipes'], agent['bird_x'],
                                         high_score, SCREEN_HEIGHT, pipe_width)
                replay_buffer.push(state, action, reward, next_state, agent['done'])

                # Update score
                for pipe in agent['pipes']:
                    if pipe.x + pipe_width == agent['bird_x']:
                        agent['score'] += 0.5

                # Update high score
                if agent['score'] > high_score:
                    high_score = agent['score']

            # Train the network
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                next_states = torch.FloatTensor(next_states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = target_net(next_states).max(1)[0]
                target = rewards + (1 - dones) * gamma * next_q

                loss = torch.nn.MSELoss()(current_q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if pygame.time.get_ticks() % TARGET_NETWORK_UPDATE_TIME == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            # Draw everything
            screen.fill(BLUE)
            for agent in agents:
                # Draw bird
                pygame.draw.rect(screen, RED, (agent['bird_x'], agent['bird_y'], bird_size, bird_size))
                
                # Draw pipes
                for pipe in agent['pipes']:
                    pygame.draw.rect(screen, GREEN, pipe)
                
                # Draw score
                score_text = font.render(f"Agent {agent['id']}: {int(agent['score'])}", True, WHITE)
                screen.blit(score_text, (10 + agent['id'] * 200, 10))

            # Draw high score
            high_score_text = font.render(f"High Score: {int(high_score)}", True, WHITE)
            screen.blit(high_score_text, (SCREEN_WIDTH - 200, 10))

            pygame.display.flip()
            clock.tick(FPS)

        # Show restart screen
        show_restart_screen()

if __name__ == "__main__":
    # Initialize agents with horizontal spacing
    agents = [{
        'id': i,
        'bird_x': 100 + (i * 200),  # Space agents horizontally
        'bird_y': SCREEN_HEIGHT // 2,
        'bird_velocity': 0,
        'pipes': [],
        'score': 0,
        'last_pipe_time': pygame.time.get_ticks(),
        'done': False
    } for i in range(AGENTS)]

    run_game(agents)