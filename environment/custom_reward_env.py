# custom_reward_env.py
import gymnasium as gym
import numpy as np

class CustomHighwayRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for highway-env.
    Includes:
    - collision penalty (very high)
    - tailgate penalty
    - safe-following bonus
    - overtake bonus
    - unnecessary lane-change penalty
    - empty-lane bonus
    """
    def __init__(self, env):
        super().__init__(env)

        # ✅ tuned reward magnitudes
        self.collision_penalty = -40.0
        self.tailgate_penalty = -5.0
        self.safe_follow_bonus = +0.3
        self.overtake_bonus = +1.5
        self.unnecessary_lane_change_penalty = -1.2
        self.empty_lane_bonus = +0.5  # reward for empty road ahead

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # get ego car
        ego = self.env.unwrapped.vehicle  # use unwrapped to access vehicle

        # ------------------------------
        # 1️⃣ Distance to car ahead
        # ------------------------------
        front_vehicle, dist = self._get_front_vehicle(ego)
        safe_distance = max(5, ego.speed * 0.5)

        if dist is not None and dist < safe_distance:
            reward += self.tailgate_penalty
        elif dist is not None and dist > safe_distance + 5:
            reward += self.safe_follow_bonus

        # ------------------------------
        # 2️⃣ Empty-lane reward
        # ------------------------------
        # Give bonus if no vehicle within 20 meters ahead
        if dist is None or dist > 20.0:
            reward += self.empty_lane_bonus

        # ------------------------------
        # 3️⃣ Overtake bonus
        # ------------------------------
        if front_vehicle is not None and (ego.speed - front_vehicle.speed) > 5:
            reward += self.overtake_bonus

        # ------------------------------
        # 4️⃣ Unnecessary lane change penalty
        # ------------------------------
        if action in [3, 4]:  # left/right lane
            if front_vehicle is None:
                reward += self.unnecessary_lane_change_penalty

        # ------------------------------
        # 5️⃣ Collision penalty
        # ------------------------------
        if info.get("crashed", False):
            reward += self.collision_penalty

        return obs, reward, terminated, truncated, info

    def _get_front_vehicle(self, ego):
        """Return closest front vehicle and distance."""
        closest_vehicle = None
        closest_dist = float('inf')

        for v in self.env.unwrapped.road.vehicles:
            if v is ego:
                continue
            # same lane only
            if v.lane_index[2] != ego.lane_index[2]:
                continue
            dx = v.position[0] - ego.position[0]
            if 0 < dx < closest_dist:
                closest_dist = dx
                closest_vehicle = v

        if closest_vehicle is None:
            return None, None
        return closest_vehicle, closest_dist
