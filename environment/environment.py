# environment.py
import gymnasium as gym
import highway_env
import numpy as np
from .custom_reward_env import CustomHighwayRewardWrapper

def flatten_obs(obs):
    return np.array(obs).flatten()

def make_env(render=False):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 12,
            "features": ["x", "y", "vx", "vy"],
            "absolute": True
        },
        "action": {"type": "DiscreteMetaAction"},
        "duration": 60,
        "simulation_frequency": 25,
        "policy_frequency": 2,
        "screen_width": 780,
        "screen_height": 420,
        "scaling": 4.0,
        "offscreen_rendering": False,
        # highway-env base rewards
        "reward_speed_range": [20, 30],
        "collision_reward": -1,
        "high_speed_reward": 0.4,
        "right_lane_reward": 0.1,
        "lane_change_reward": -0.05,
        "normalize_reward": True,
        "offroad_terminal": True
    }

    env = gym.make("highway-v0", config=config, render_mode="human" if render else None)
    env = CustomHighwayRewardWrapper(env)
    return env
