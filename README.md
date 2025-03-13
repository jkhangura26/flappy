# Flappy Bird AI

This repository implements a Flappy Bird clone where an AI agent, trained using Deep Q-Learning (DQN), learns to play the game. The project is built using Python, PyGame, and PyTorch.

## Features

- Flappy Bird gameplay with AI agent
- Deep Q-Network implementation
- Replay buffer for experience replay
- Adjustable parameters for training and gameplay

## How It Works

- Game Environment: Built with PyGame, the environment includes bird, pipes, and scoring logic.
- Deep Q-Network: A neural network approximates the Q-values for state-action pairs.
- Replay Buffer: Stores experiences for off-policy learning.
- Multi-Agent System: Multiple birds learn simultaneously in parallel
- Performance Optimizations: Fixed time-step simulation, headless mode options, and efficient collision handling.

