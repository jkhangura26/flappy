import pygame
import sys
import time
import random
import torch
import logging
import argparse
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, AGENTS, BIRD_SIZE, GRAVITY, JUMP_STRENGTH, PIPE_WIDTH, PIPE_VELOCITY, SPAWN_TIME_PIPE, TARGET_NETWORK_UPDATE_TIME, BLUE, GREEN, WHITE, clock, font
from ai import policy_net, target_net, optimizer, replay_buffer, epsilon, epsilon_decay, epsilon_min, gamma, batch_size, save_checkpoint, load_checkpoint
from ai import calculate_reward
from utils import get_state, check_collision, create_pipe

# ----------------------------
# Parse command-line arguments for headless mode
# ----------------------------
parser = argparse.ArgumentParser(description="Flappy Bird AI Championships")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
parser.add_argument("--load_model", type=str, default=None, help="Path to a saved model checkpoint to load")
parser.add_argument("--save_model", type=str, default="model_checkpoint.pt", help="Path to save the model checkpoint")
args = parser.parse_args()

HEADLESS_MODE = args.headless
MODEL_SAVE_PATH = args.save_model
MODEL_LOAD_PATH = args.load_model

# ----------------------------
# Pygame Initialization (only if GUI is needed)
# ----------------------------
pygame.init()
if not HEADLESS_MODE:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Multi-Agent AI Championships")
else:
    # In headless mode, we do not create a visible display.
    screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------
# Agent Class
# ----------------------------
class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.reset()
    
    def reset(self):
        self.bird_x = SCREEN_WIDTH // 4
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.score = 0
        self.done = False
        self.death_reported = False
        self.passed_pipes = {}

    def get_color(self):
        # Unique color for this agent using HSL color space.
        hue = (self.id * 360 / AGENTS) % 360
        color = pygame.Color(0)
        color.hsla = (hue, 100, 50, 100)
        return color

    def update(self, pipes):
        if self.done:
            return

        # Get state and choose an action
        state = get_state(self.bird_x, self.bird_y, pipes, PIPE_WIDTH, self.bird_velocity, PIPE_VELOCITY)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        global epsilon
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

        # Apply physics
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Check collisions
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, BIRD_SIZE, BIRD_SIZE)
        self.done = check_collision(bird_rect, pipes)

        # Update score when passing pipes
        for pipe in pipes:
            if abs((pipe.x + PIPE_WIDTH) - self.bird_x) <= 1:
                pipe_id = (pipe.x, pipe.height)
                if pipe_id not in self.passed_pipes:
                    self.score += 0.5
                    self.passed_pipes[pipe_id] = True

        # Store experience for training
        next_state = get_state(self.bird_x, self.bird_y, pipes, PIPE_WIDTH, self.bird_velocity, PIPE_VELOCITY)
        reward = calculate_reward(self.done, self.bird_y, BIRD_SIZE, self.score, pipes, self.bird_x,
                                    Game.high_score, SCREEN_HEIGHT, PIPE_WIDTH, self.bird_velocity)
        replay_buffer.push(state, action, reward, next_state, self.done)

        # Update global high score
        if self.score > Game.high_score:
            Game.high_score = self.score

    def report_death(self):
        if not self.death_reported:
            logging.info(f"Agent {self.id} died with score: {self.score:.1f}, High Score: {int(Game.high_score)}")
            self.death_reported = True

    def draw(self, surface):
        color = self.get_color()
        pygame.draw.rect(surface, color, (self.bird_x, self.bird_y, BIRD_SIZE, BIRD_SIZE))
        score_text = font.render(f"Score: {int(self.score)}", True, color)
        surface.blit(score_text, (10 + self.id * 200, 10))

# ----------------------------
# PipeManager Class
# ----------------------------
class PipeManager:
    def __init__(self):
        self.pipes = []
        self.last_pipe_time = pygame.time.get_ticks()

    def update(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe_time > SPAWN_TIME_PIPE:
            top_pipe, bottom_pipe = create_pipe()
            self.pipes.extend([top_pipe, bottom_pipe])
            self.last_pipe_time = current_time

        # Move pipes and remove those off-screen
        self.pipes = [pipe for pipe in self.pipes if pipe.x + PIPE_WIDTH > 0]
        for pipe in self.pipes:
            pipe.x += PIPE_VELOCITY

    def draw(self, surface):
        for pipe in self.pipes:
            pygame.draw.rect(surface, GREEN, pipe)

# ----------------------------
# Game Class with Fixed Time-Step and Headless Mode
# ----------------------------
class Game:
    high_score = 0  # Shared global high score

    def __init__(self):
        self.agents = [Agent(i) for i in range(AGENTS)]
        self.pipe_manager = PipeManager()

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.pipe_manager = PipeManager()
        Game.high_score = 0

    def train_network(self):
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

    def update(self):
        self.pipe_manager.update()
        pipes = self.pipe_manager.pipes
        for agent in self.agents:
            if not agent.done:
                agent.update(pipes)
            else:
                agent.report_death()

        self.train_network()

        global epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    def draw(self):
        # Only called when GUI is enabled
        screen.fill(BLUE)
        self.pipe_manager.draw(screen)
        for agent in self.agents:
            agent.draw(screen)
        high_score_text = font.render(f"High Score: {int(Game.high_score)}", True, WHITE)
        screen.blit(high_score_text, (SCREEN_WIDTH - 200, 10))
        pygame.display.flip()

    def save_model(self):
        save_checkpoint(MODEL_SAVE_PATH)

    def load_model(self):
        if MODEL_LOAD_PATH:
            load_checkpoint(MODEL_LOAD_PATH)

    def run(self):
        self.reset()
        running = True

        # Fixed time-step settings
        simulation_dt = 1.0 / 60.0  # 60 updates per second
        accumulator = 0.0
        last_time = time.time()

        # Timer for target network update
        last_target_update = time.time()

        # Load model if specified
        self.load_model()

        while running:
            current_time = time.time()
            frame_time = current_time - last_time
            last_time = current_time
            accumulator += frame_time

            # Process Pygame events (even in headless mode, to catch QUIT)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update simulation with fixed time-step
            while accumulator >= simulation_dt:
                self.update()
                accumulator -= simulation_dt

                # Update target network at fixed intervals (convert ms to seconds)
                if current_time - last_target_update >= TARGET_NETWORK_UPDATE_TIME / 1000.0:
                    target_net.load_state_dict(policy_net.state_dict())
                    last_target_update = current_time

            # Render only if not in headless mode
            if not HEADLESS_MODE:
                self.draw()

            clock.tick(FPS)

            # If all agents are done, reset the game
            if all(agent.done for agent in self.agents):
                logging.info("All agents are dead. Resetting episode.")
                self.reset()

        # Save the model at the end of the run
        self.save_model()

        pygame.quit()
        sys.exit()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    game = Game()
    game.run()
