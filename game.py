import pygame
import random
import numpy as np

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
BIRD_SIZE = 20
PIPE_WIDTH = 50
PIPE_GAP = 150
GRAVITY = 0.5
JUMP_STRENGTH = -10
PIPE_SPEED = 5
FPS = 60  

class FlappyBirdGame:
    def __init__(self, agent_id):
        pygame.init()
        self.agent_id = agent_id
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Flappy Bird - Agent {agent_id}")

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.frames_since_pipe = 0

    def get_state(self):
        nearest_pipe = next((p for p in self.pipes if p.x + PIPE_WIDTH > 50), None)
        if nearest_pipe:
            return np.array([
                self.bird_y / SCREEN_HEIGHT,
                self.bird_velocity / 10.0,
                nearest_pipe.x / SCREEN_WIDTH,
                nearest_pipe.height / SCREEN_HEIGHT,
                (nearest_pipe.height + PIPE_GAP) / SCREEN_HEIGHT
            ])
        return np.array([self.bird_y / SCREEN_HEIGHT, self.bird_velocity / 10.0, 1.0, 0.0, 1.0])

    def step(self, action):
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        if self.frames_since_pipe >= FPS * 1.5:
            self.spawn_pipe()
            self.frames_since_pipe = 0
        self.frames_since_pipe += 1

        for pipe in self.pipes:
            pipe.x -= PIPE_SPEED

        self.pipes = [pipe for pipe in self.pipes if pipe.x + PIPE_WIDTH > 0]

        if any(self.check_collision(pipe) for pipe in self.pipes) or self.bird_y <= 0 or self.bird_y >= SCREEN_HEIGHT:
            self.game_over = True
            return self.get_state(), -100, True

        return self.get_state(), 1, False  # Reward of 1 for surviving

    def spawn_pipe(self):
        pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
        self.pipes.append(Pipe(SCREEN_WIDTH, pipe_height))

    def check_collision(self, pipe):
        return (pipe.x < 50 + BIRD_SIZE < pipe.x + PIPE_WIDTH and 
                not (pipe.height < self.bird_y < pipe.height + PIPE_GAP))

    def render(self):
        self.screen.fill((135, 206, 250))  # Light blue background

        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe.x, 0, PIPE_WIDTH, pipe.height))
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe.x, pipe.height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT))

        # Draw bird
        pygame.draw.circle(self.screen, (255, 255, 0), (50, int(self.bird_y)), BIRD_SIZE)

        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

class Pipe:
    def __init__(self, x, height):
        self.x = x
        self.height = height
