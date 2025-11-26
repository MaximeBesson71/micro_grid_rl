import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.microgrid_env import MicrogridEnv

def evaluate(model_path="models/ppo_microgrid", profile=None):
    if profile is None:
        from data.generate_data import generate_day_profiles
        profile = generate_day_profiles(seed=123)

    env = MicrogridEnv(profile_df=profile)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        rec = {
            "hour": int(env.current_step-1),
            "reward": reward,
            "cost": info.get("cost"),
            "soc": info.get("soc"),
            "p_batt": info.get("p_batt"),
            "solar": float(profile.loc[env.current_step-1, "solar_kW"]),
            "load": float(profile.loc[env.current_step-1, "load_kW"]),
            "price": float(profile.loc[env.current_step-1, "price_eur_per_kWh"])
        }
        records.append(rec)

    df = pd.DataFrame(records)
    print("Total cost (EUR):", df["cost"].sum())

    # Plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(df["hour"], df["solar"], label="Solar (kW)")
    axs[0].plot(df["hour"], df["load"], label="Load (kW)")
    axs[0].legend()
    axs[0].set_ylabel("kW")

    axs[1].plot(df["hour"], df["p_batt"], label="Battery power (kW)")
    axs[1].plot(df["hour"], df["soc"], label="SOC")
    axs[1].legend()
    axs[1].set_ylabel("kW / SOC")

    axs[2].plot(df["hour"], df["price"], label="Price (EUR/kWh)")
    axs[2].bar(df["hour"], df["cost"], alpha=0.4, label="Cost (EUR)")
    axs[2].legend()
    axs[2].set_ylabel("EUR")

    axs[2].set_xlabel("Hour")
    plt.tight_layout()
    plt.show()
    return df

if __name__ == "__main__":
    df = evaluate(model_path="models/ppo_microgrid")