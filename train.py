# train.py
import torch
import numpy as np
from environment.environment import make_env, flatten_obs
from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer

# -------------------- Hyperparameters --------------------
EPISODES = 700
TARGET_UPDATE_EPISODES = 5       # copy weights every N episodes
BATCH_SIZE = 64
BUFFER_CAPACITY = 150_000
MIN_REPLAY_SIZE = 5_000           # warmup before batch training
MAX_STEPS_PER_EPISODE = 2000
PRINT_EVERY = 1
RENDER = True                      # render window

# -------------------- Main Training Loop --------------------
def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    env = make_env(render=RENDER)

    # infer dimensions
    obs, info = env.reset()
    s0 = flatten_obs(obs)
    state_dim = s0.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device=str(device))
    replay = ReplayBuffer(capacity=BUFFER_CAPACITY)

    best_reward = -1e9

    # -------------------- Pre-fill Replay Buffer --------------------
    print("Warming up replay buffer with random actions...")
    while len(replay) < MIN_REPLAY_SIZE:
        obs, info = env.reset()
        obs = flatten_obs(obs)
        done = False
        steps = 0
        while not done and len(replay) < MIN_REPLAY_SIZE and steps < MAX_STEPS_PER_EPISODE:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_obs(next_obs)
            done = bool(terminated or truncated)
            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            steps += 1

    print("Replay buffer warmup done:", len(replay))

    # -------------------- Episode Loop --------------------
    for ep in range(1, EPISODES + 1):
        obs, info = env.reset()
        obs = flatten_obs(obs)

        done = False
        total_reward = 0.0
        step = 0

        while not done and step < MAX_STEPS_PER_EPISODE:
            step += 1
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_obs(next_obs)
            done = bool(terminated or truncated)

            replay.push(obs, action, reward, next_obs, done)
            total_reward += float(reward)
            obs = next_obs

            # batch training
            states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
            agent.batch_train(states, actions, rewards, next_states, dones)

            if RENDER:
                env.render()

        # update target network periodically
        if ep % TARGET_UPDATE_EPISODES == 0:
            agent.update_target()

        # decay epsilon once per episode
        agent.decay_epsilon()

        if ep % PRINT_EVERY == 0:
            print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), "dqn_highway_simple.pth")

    env.close()
    print("Training finished. Best reward:", best_reward)


if __name__ == "__main__":
    main()
