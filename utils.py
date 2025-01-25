from config import *
import numpy as np
import random

# Function to get the current state
def get_state(bird_x, bird_y, pipes, pipe_width, bird_velocity, pipe_velocity):
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
    """Create a pair of pipes (top and bottom) as Rect objects"""
    gap_y = random.randint(100, SCREEN_HEIGHT - 300)
    top_pipe = pygame.Rect(SCREEN_WIDTH, 0, PIPE_WIDTH, gap_y)
    bottom_pipe = pygame.Rect(SCREEN_WIDTH, gap_y + 200, PIPE_WIDTH, SCREEN_HEIGHT - gap_y - 200)
    return top_pipe, bottom_pipe

# Function to check collisions
def check_collision(bird_rect, pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return True
    if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
        return True
    return False

