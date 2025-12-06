import torch
from environment.environment import make_env, flatten_obs
from agent.dqn_agent import DQNAgent
from environment.custom_reward_env import CustomHighwayRewardWrapper
import numpy as np

def main():
    # 1️⃣ Create environment exactly as in training
    env = make_env(render=True)
    env = CustomHighwayRewardWrapper(env)

    # 2️⃣ Reset environment
    obs, info = env.reset()
    obs = flatten_obs(obs)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    # 3️⃣ Load DQN agent
    agent = DQNAgent(state_dim, action_dim)

    try:
        agent.model.load_state_dict(
            torch.load("dqn_highway_simple.pth", map_location=agent.device)
        )
        print("Loaded model dqn_highway_simple.pth")
    except FileNotFoundError:
        print("❌ No saved model found")
        return

    agent.model.eval()
    agent.epsilon = 0.0     # 🚨 FORCE NO EXPLORATION
    agent.epsilon_min = 0.0

    done = False
    total_reward = 0.0
    step = 0

    while not done:
        step += 1

        # 4️⃣ Fully greedy action (no randomness)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            q_values = agent.model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        # Safety clamp
        action = np.clip(action, 0, action_dim - 1)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = flatten_obs(next_obs)

        # ❌ Removed reward clipping
        # ❌ Removed obs clipping

        speed = info.get('speed', 0)
        crashed = info.get('crashed', False)

        print(
            f"Step {step}: Action={action}, "
            f"Reward={reward:.2f}, Speed={speed:.2f}, Crashed={crashed}"
        )

        obs = next_obs
        done = terminated or truncated
        total_reward += reward

    print(f"🏁 Test finished — Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
