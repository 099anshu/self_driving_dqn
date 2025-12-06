# Import libraries
import random   # For random sampling
import numpy as np  # For handling arrays

# --------------------------
# Replay Buffer Class
# --------------------------
class ReplayBuffer:
    """
    ReplayBuffer stores the agent's experiences (state, action, reward, next_state, done)
    so that we can sample random batches for training.
    This helps break correlation between consecutive experiences,
    which stabilizes training.
    """
    def __init__(self, capacity=150000):
        """
        Initialize the buffer.
        :param capacity: maximum number of experiences the buffer can hold
        """
        self.buffer = []       # List to store experiences
        self.capacity = capacity  # Maximum size of the buffer

    def push(self, s, a, r, ns, d):
        """
        Add a new experience to the buffer.
        :param s: state
        :param a: action
        :param r: reward
        :param ns: next state
        :param d: done (True if episode ended, else False)
        """
        # If buffer is full, remove the oldest experience
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Removes first element (oldest)
        # Add the new experience as a tuple
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.
        :param batch_size: number of experiences to sample
        :return: separate arrays for states, actions, rewards, next_states, dones
        """
        # Ensure batch_size does not exceed the number of experiences stored
        batch_size = min(batch_size, len(self.buffer))
        # Randomly sample 'batch_size' experiences
        batch = random.sample(self.buffer, batch_size)
        # Unpack the batch into separate lists
        s, a, r, ns, d = zip(*batch)
        # Convert lists to numpy arrays for easier processing in training
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        """
        Return the current number of experiences in the buffer
        """
        return len(self.buffer)
