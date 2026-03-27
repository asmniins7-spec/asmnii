import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_mspacman(total_timesteps=400_000):
    env = gym.make("MsPacmanNoFrameskip-v4")
    env = AtariWrapper(env)
    model = DQN("CnnPolicy", env, verbose=1, tensorboard_log="./tb_mspacman")
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(MODEL_DIR, "mspacman_dqn")
    model.save(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"MsPacman evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    train_mspacman()
