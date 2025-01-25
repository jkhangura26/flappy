import pygame
import sys
import random
import torch
from config import *
from ai import policy_net, target_net, optimizer, replay_buffer, epsilon, epsilon_decay, epsilon_min, gamma, batch_size, ReplayBuffer, DQN, calculate_reward
from utils import get_state, check_collision, create_pipe

# Initialize Pygame
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
pipes = []  # Shared across all agents
last_pipe_time = pygame.time.get_ticks()

# Initialize networks
target_net.load_state_dict(policy_net.state_dict())

def get_agent_color(agent_id):
    """Generate unique color using HSL color space"""
    hue = (agent_id * 360 / AGENTS) % 360
    color = pygame.Color(0)
    color.hsla = (hue, 100, 50, 100)
    return color

def reset_game(agents):
    """Reset all agents to initial state"""
    global pipes, last_pipe_time
    pipes = []
    last_pipe_time = pygame.time.get_ticks()
    
    for agent in agents:
        agent['bird_y'] = SCREEN_HEIGHT // 2
        agent['bird_velocity'] = 0
        agent['score'] = 0
        agent['done'] = False

def run_game(agents):
    """Main game loop with multiple agents"""
    global high_score, epsilon, pipes, last_pipe_time

    # Generate color for each agent
    agent_colors = [get_agent_color(i) for i in range(AGENTS)]

    while True:
        reset_game(agents)
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

            # Update pipes (shared across all agents)
            current_time = pygame.time.get_ticks()
            if current_time - last_pipe_time > pipe_spawn_time:
                top_pipe, bottom_pipe = create_pipe()
                pipes.extend([top_pipe, bottom_pipe])  # Add both pipes as Rect objects
                last_pipe_time = current_time

            # Move pipes
            pipes = [pipe for pipe in pipes if pipe.x + PIPE_WIDTH > 0]
            for pipe in pipes:
                pipe.x += pipe_velocity

            # Update each agent
            for agent, color in zip(agents, agent_colors):
                if agent['done']:
                    continue

                # Get current state
                state = get_state(agent['bird_x'], agent['bird_y'], pipes, 
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

                # Check collisions
                bird_rect = pygame.Rect(agent['bird_x'], agent['bird_y'], bird_size, bird_size)
                agent['done'] = check_collision(bird_rect, pipes)

                # Store experience
                next_state = get_state(agent['bird_x'], agent['bird_y'], pipes,
                                      pipe_width, agent['bird_velocity'], pipe_velocity)
                reward = calculate_reward(agent['done'], agent['bird_y'], bird_size,
                                         agent['score'], pipes, agent['bird_x'],
                                         high_score, SCREEN_HEIGHT, pipe_width, agent['bird_velocity'])
                replay_buffer.push(state, action, reward, next_state, agent['done'])

                # Update score
                for pipe in pipes:
                    if abs((pipe.x + pipe_width) - agent['bird_x']) <= 1:
                        # Use pipe position as unique identifier
                        pipe_id = (pipe.x, pipe.height)
                        if pipe_id not in agent['passed_pipes']:
                            agent['score'] += 0.5
                            agent['passed_pipes'][pipe_id] = True

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
            
            # Draw shared pipes
            for pipe in pipes:
                pygame.draw.rect(screen, GREEN, pipe)
            
            # Draw agents
            for agent, color in zip(agents, agent_colors):
                # Draw bird
                pygame.draw.rect(screen, color, (agent['bird_x'], agent['bird_y'], bird_size, bird_size))
                
                # Draw score
                score_text = font.render(f"Score: {int(agent['score'])}", True, color)
                screen.blit(score_text, (10 + agent['id'] * 200, 10))

            # Draw high score
            high_score_text = font.render(f"High Score: {int(high_score)}", True, WHITE)
            screen.blit(high_score_text, (SCREEN_WIDTH - 200, 10))

            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    # Initialize agents with identical starting positions
    agents = [{
        'id': i,
        'bird_x': SCREEN_WIDTH // 4,
        'bird_y': SCREEN_HEIGHT // 2,
        'bird_velocity': 0,
        'score': 0,
        'passed_pipes': dict(),
        'done': False
    } for i in range(AGENTS)]

    run_game(agents)