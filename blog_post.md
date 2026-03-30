https://dev.to/asmniinsds/atari-deep-q-network-projectatari-deep-q-network-project-45g6

# Atari Deep Q-Network Project

## Overview
This project implements 3 reinforcement learning agents using Deep Q-Networks (DQN) to play:
- CartPole-v1 (classic control)
- SpaceInvadersNoFrameskip-v4
- MsPacmanNoFrameskip-v4

The solution is intended for educational Bootcamp-style demonstration and can be extended for full Atari training on GPU.

## Approach
1. Environment setup with OpenAI Gym / Gymnasium.
2. DQN agent training with Stable Baselines3.
3. Atari image preprocessing via `AtariWrapper`.
4. Periodic evaluation with `evaluate_policy`.

## Running the models
```bash
pip install -U gymnasium[atari] stable-baselines3[extra] torch
python atari_cartpole.py
python atari_spaceinvaders.py
python atari_mspacman.py
```

## Results
Expected: CartPole learns to solve within ~100k steps, Atari agents show incremental reward improvements with additional timesteps.

## Improvements
- Add better hyperparameter search (learning rate, buffer size, exploration decay)
- Add experience replay inspection and prioritized replay
- Utilize GPU with `tensorboard` logging and model checkpoints
- Train on full 49 Atari games with a shared/transfer learning agent
