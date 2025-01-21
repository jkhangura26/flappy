import pygame

# Initialize Pygame
pygame.init()

# Number of agents
AGENTS = 1000

# Screen settings: Modify the width based on the number of agents
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 600
TARGET_NETWORK_UPDATE_TIME = 150
SPAWN_TIME_PIPE = 1000

# Colors
WHITE = (255, 255, 255)
BLUE = (135, 206, 250)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

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

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Multiple Agents")
clock = pygame.time.Clock()

# Font for rendering text
font = pygame.font.SysFont(None, 36)
