# agent/train_agent.py
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.microgrid_env import MicrogridEnv

def load_profile(path=None):
    if path is None:
        # generate default profiles on the fly
        from data.generate_data import generate_day_profiles
        return generate_day_profiles(seed=0)
    return pd.read_csv(path)

def make_env(profile_df):
    def _init():
        env = MicrogridEnv(profile_df=profile_df,
                           battery_capacity_kwh=10.0,
                           battery_power_max_kw=5.0,
                           eff=0.95)
        return env
    return _init

def train(save_dir="models", timesteps=20000):
    os.makedirs(save_dir, exist_ok=True)
    profile = load_profile()  # DataFrame 24 steps
    env = DummyVecEnv([make_env(profile)])
    model = PPO("MlpPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=timesteps)
    model_path = os.path.join(save_dir, "ppo_microgrid")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model, env

if __name__ == "__main__":
    print("yo")
    train(timesteps=40000)