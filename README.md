# Atari Games DQN Project

This repository contains sample scripts for training Deep Q-Network agents for Atari-like games.

## Files
- `atari_cartpole.py`: DQN for CartPole-v1
- `atari_spaceinvaders.py`: DQN for SpaceInvadersNoFrameskip-v4
- `atari_mspacman.py`: DQN for MsPacmanNoFrameskip-v4
- `blog_post.md`: explanation of approach

## Setup
```bash
python -m pip install --upgrade pip
pip install gymnasium gymnasium[atari] stable-baselines3[extra] torch
```

## Train
```bash
python atari_cartpole.py
python atari_spaceinvaders.py
python atari_mspacman.py
```
