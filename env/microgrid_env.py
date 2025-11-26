import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MicrogridEnv(gym.Env):
    """
    Environnement simple pour un micro-réseau (résolution : 1 heure).
    State: [solar_kW, load_kW, price, soc, hour]
    Action: continuous in [-1,1] -> mapped to power in kW (negative: discharge)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, profile_df: pd.DataFrame, battery_capacity_kwh=10.0,
                 battery_power_max_kw=5.0, eff=0.95, timestep_hours=1.0):
        super().__init__()
        self.profile = profile_df.reset_index(drop=True)
        self.T = len(self.profile)
        self.t = 0

        # Battery parameters
        self.capacity = float(battery_capacity_kwh)
        self.p_max = float(battery_power_max_kw)
        self.eff = float(eff)
        self.dt = float(timestep_hours)

        # Observation space: solar, load, price, soc (0-1), hour (0-23)
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([20.0, 20.0, 5.0, 1.0, 23.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: continuous in [-1,1] -> scale to power [-p_max, p_max]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # state variables
        self.soc = 0.5  # initial SOC
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = 0.5  # 50% initial
        self.t = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.profile.loc[self.current_step]
        obs = np.array([
            float(row["solar_kW"]),
            float(row["load_kW"]),
            float(row["price_eur_per_kWh"]),
            float(self.soc),
            float(row["hour"])
        ], dtype=np.float32)
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)[0]
        # Map action to power in kW
        p_batt = action * self.p_max  # >0 => charging, <0 => discharging
        row = self.profile.loc[self.current_step]
        solar = float(row["solar_kW"])
        load = float(row["load_kW"])
        price = float(row["price_eur_per_kWh"])

        # Battery energy change (kWh)
        if p_batt >= 0:
            # charging: battery receives p_batt * dt, but actual SOC increase accounts for efficiency
            energy_into_batt = p_batt * self.dt * self.eff
            usable = (1.0 - self.soc) * self.capacity
            energy_charged = min(energy_into_batt, usable)
            soc_delta = energy_charged / self.capacity
            # power drawn from grid to charge = p_batt (limited by p_max)
            grid_from_batt = p_batt
        else:
            # discharging: battery provides -p_batt to meet load, but we lose 1/eff
            energy_out = (-p_batt) * self.dt / self.eff
            available = self.soc * self.capacity
            energy_discharged = min(energy_out, available)
            soc_delta = - energy_discharged / self.capacity
            # effective delivered power to load from battery
            delivered = energy_discharged / self.dt * self.eff  # ~ -p_batt or limited
            grid_from_batt = p_batt  # negative

        # Update SOC
        self.soc = float(np.clip(self.soc + soc_delta, 0.0, 1.0))

        # Net grid exchange: positive = buy from grid, negative = sell to grid
        # power balance per hour: load - solar - battery_discharge
        batt_power_delivered = 0.0
        if p_batt >= 0:
            batt_power_delivered = 0.0  # battery is charging, not delivering
        else:
            # delivered power = min(-p_batt, available)
            batt_power_delivered = min(-p_batt, (self.soc + (-soc_delta)) * self.capacity / self.dt * self.eff)

        net_load = load - solar - batt_power_delivered  # kW
        # positive -> need to buy from grid; negative -> sell to grid
        # cost for this timestep:
        energy_from_grid = net_load * self.dt  # kWh (can be negative)
        cost = energy_from_grid * price  # EUR (negative means revenue)

        # Penalties for SOC limits (soft)
        penalty = 0.0
        if self.soc <= 0.01 or self.soc >= 0.99:
            penalty += 0.1  # small penalty to avoid extremes

        reward = -cost - penalty

        self.current_step += 1
        done = self.current_step >= self.T
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "cost": float(cost),
            "soc": float(self.soc),
            "net_load": float(net_load),
            "p_batt": float(p_batt)
        }
        return obs, float(reward), done, False, info

    def render(self):
        pass