import pygame

# Initialize Pygame
pygame.init()

# ----------------------------
# Screen Settings
# ----------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
FPS = 60  # Use a fixed FPS for consistent physics updates

# ----------------------------
# Game Constants
# ----------------------------
AGENTS = 100          # Number of simultaneous AI agents
BIRD_SIZE = 20        # Bird dimensions (square)
GRAVITY = 0.5         # Downward acceleration
JUMP_STRENGTH = -8    # Upward velocity when jumping

# ----------------------------
# Pipe Settings
# ----------------------------
PIPE_WIDTH = 50       # Width of each pipe
PIPE_VELOCITY = -3    # Speed at which pipes move to the left
SPAWN_TIME_PIPE = 1500  # Time in milliseconds between pipe spawns
TARGET_NETWORK_UPDATE_TIME = 1000  # Update interval for target network in ms

# ----------------------------
# Colors (RGB)
# ----------------------------
WHITE = (255, 255, 255)
BLUE = (135, 206, 250)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# ----------------------------
# Pygame Display Setup (for non-headless mode)
# ----------------------------
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Multi-Agent AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)