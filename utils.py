"""
utils.py

Utility functions for state representation, pipe creation, and collision detection.
"""

import pygame
import random
import numpy as np
from config import SCREEN_HEIGHT, SCREEN_WIDTH, PIPE_WIDTH

def get_state(bird_x, bird_y, pipes, pipe_width, bird_velocity, pipe_velocity):
    """
    Generate the normalized state representation for the AI.
    
    Args:
        bird_x (float): Bird's horizontal position.
        bird_y (float): Bird's vertical position.
        pipes (list): List of current pipe rectangles.
        pipe_width (int): The width of a pipe.
        bird_velocity (float): The current vertical velocity of the bird.
        pipe_velocity (float): The velocity of the pipes.
    
    Returns:
        np.array: A normalized state vector.
    """
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
        # Default state when no pipes are present
        return np.array([
            bird_y / SCREEN_HEIGHT,
            bird_velocity / 10.0,
            1.0, 0.0, 1.0, 1.0,
            pipe_velocity / 10.0
        ])

def create_pipe():
    """
    Create a pair of pipes (top and bottom) with a gap in between.
    
    Returns:
        tuple: (top_pipe, bottom_pipe) as pygame.Rect objects.
    """
    gap_y = random.randint(100, SCREEN_HEIGHT - 300)
    top_pipe = pygame.Rect(SCREEN_WIDTH, 0, PIPE_WIDTH, gap_y)
    bottom_pipe = pygame.Rect(SCREEN_WIDTH, gap_y + 200, PIPE_WIDTH, SCREEN_HEIGHT - gap_y - 200)
    return top_pipe, bottom_pipe

def check_collision(bird_rect, pipes):
    """
    Check if the bird collides with any pipes or the screen boundaries.
    
    Args:
        bird_rect (pygame.Rect): The rectangle representing the bird.
        pipes (list): List of pipe rectangles.
    
    Returns:
        bool: True if a collision is detected, else False.
    """
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return True
    if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
        return True
    return False
