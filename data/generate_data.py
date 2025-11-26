import numpy as np
import pandas as pd

def generate_day_profiles(seed=0):
    rng = np.random.RandomState(seed)
    hours = np.arange(24)

    # Solaire : pic autour de midi
    solar = np.maximum(0, 5 * np.sin((hours - 6) * np.pi / 12))  # kW max ~5
    solar += rng.normal(scale=0.2, size=24)  # bruit léger
    solar = np.clip(solar, 0, None)

    # Consommation : matin et soir
    base = 1.0 + 0.2 * rng.randn(24)
    load = base + 1.5 * np.exp(-0.5 * ((hours - 8) / 1.5)**2)  # matin
    load += 2.0 * np.exp(-0.5 * ((hours - 19) / 2.0)**2)      # soir
    load = np.clip(load, 0.3, None)

    # Prix : heures pleines (17-21) plus chères
    price = 0.15 + 0.05 * np.sin(hours * 2 * np.pi / 24)  # variation
    price[17:22] += 0.10

    df = pd.DataFrame({
        "hour": hours,
        "solar_kW": solar,
        "load_kW": load,
        "price_eur_per_kWh": price
    })
    return df

if __name__ == "__main__":
    df = generate_day_profiles(seed=42)
    df.to_csv("data/day_profile.csv", index=False)
    print("Saved data/day_profile.csv")