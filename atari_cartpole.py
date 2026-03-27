import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_cartpole(total_timesteps=100_000):
    env = gym.make("CartPole-v1")
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tb_cartpole")
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(MODEL_DIR, "cartpole_dqn")
    model.save(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"CartPole evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    train_cartpole()
