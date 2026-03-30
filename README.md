# Atari Games DQN Project

## Task

Build and train Deep Q-Network (DQN) agents for three environments:
- CartPole-v1 (classic control)
- SpaceInvadersNoFrameskip-v4 (Atari)
- MsPacmanNoFrameskip-v4 (Atari)

## Description

This project demonstrates training DQN agents with Stable Baselines3 and Gymnasium (Atari) wrappers. The notebook `combined_atari_training.ipynb` contains the implementations for all three tasks, including training, evaluation, and saving models.

## Installation

```bash
python -m pip install --upgrade pip
pip install gymnasium gymnasium[atari] stable-baselines3[extra] torch
```

## Usage

Run the notebook in a Jupyter environment:

```bash
jupyter notebook Atari games.ipynb
```

Inside the notebook, execute setup cells first, then train accordingly:

- `train_cartpole(total_timesteps=100_000)`
- `train_spaceinvaders(total_timesteps=400_000)`
- `train_mspacman(total_timesteps=400_000)`

For quick testing, reduce timesteps (example below):

```python
train_spaceinvaders(10_000)
```

