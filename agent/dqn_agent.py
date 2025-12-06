# Import libraries needed for building the agent
import torch                 # PyTorch library for creating and training neural networks
import torch.nn as nn        # nn module contains layers, activation functions, etc.
import torch.optim as optim  # optim module contains optimizers to update network weights
import numpy as np           # numpy for handling arrays, math, and random numbers

# --------------------------
# Dueling DQN Network Class
# --------------------------
class DuelingNetwork(nn.Module):
    """
    This is the neural network that predicts Q-values using the dueling architecture.
    Q-value = V(s) + (A(s, a) - mean(A(s, *)))
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()  # Call PyTorch's base class constructor
        # Feature extractor: converts raw state into higher-level features
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Fully connected layer: input -> hidden_dim
            nn.ReLU(),                         # Activation: adds non-linearity
            nn.Linear(hidden_dim, hidden_dim), # Another hidden layer
            nn.ReLU()                          # Activation again
        )

        # Value stream: predicts how good the state is overall (scalar V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # Shrinks features by half
            nn.ReLU(),                               # Non-linearity
            nn.Linear(hidden_dim // 2, 1)           # Output: 1 number representing V(s)
        )

        # Advantage stream: predicts advantage of each action (vector A(s,a))
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # Shrinks features
            nn.ReLU(),                               # Non-linearity
            nn.Linear(hidden_dim // 2, action_dim)  # Output: one number per action
        )

    # Forward pass: defines how input flows through the network
    def forward(self, x):
        features = self.feature(x)           # Step 1: extract features from input
        value = self.value_stream(features)  # Step 2: compute value of state
        adv = self.adv_stream(features)      # Step 3: compute advantages for each action
        adv_mean = adv.mean(dim=1, keepdim=True)  # Step 4: compute mean advantage (for normalization)
        q = value + (adv - adv_mean)         # Step 5: combine value + advantage → final Q-values
        return q                             # Step 6: return Q-values for all actions

# --------------------------
# DQN Agent Class
# --------------------------
class DQNAgent:
    """
    This is the agent that interacts with the environment.
    It decides actions, stores knowledge, and learns from experience.
    """
    def __init__(
        self,
        state_dim,         # Number of features in the state (like position, velocity, etc.)
        action_dim,        # Number of possible actions the agent can take
        lr=3e-4,           # Learning rate: controls how fast the network updates weights
        gamma=0.99,        # Discount factor: how much we care about future rewards
        epsilon=1.0,       # Initial probability of taking a random action (exploration)
        epsilon_min=0.05,  # Minimum exploration probability
        epsilon_decay=0.995, # How fast epsilon decreases after each episode
        device: str | None = None,  # CPU or GPU to run computations
    ):
        # Choose device: GPU if available, otherwise CPU
        # torch.device("mps") is for Mac GPUs; otherwise, fallback to CPU
        self.device = torch.device(device) if device else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )

        # Save important parameters for later use
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Future reward importance

        # Create main Q-network: learns to predict Q-values
        self.model = DuelingNetwork(state_dim, action_dim).to(self.device)
        # Create target Q-network: more stable, used for calculating target Q-values
        self.target_model = DuelingNetwork(state_dim, action_dim).to(self.device)
        self.update_target()  # Copy main network weights to target at the start

        # Optimizer: Adam automatically adjusts learning rates for each weight
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Loss function: measures difference between predicted Q and target Q
        self.loss_fn = nn.MSELoss()

        # Epsilon-greedy parameters for exploration vs exploitation
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # --------------------------
    # Choose an action
    # --------------------------
    def act(self, state):
        """
        Choose an action given the current state.
        Uses epsilon-greedy: mostly best action, sometimes random.
        """
        # Step 1: Decide whether to explore (random action) or exploit (best action)
        if np.random.rand() < self.epsilon:  # Random number between 0 and 1
            return int(np.random.randint(0, self.action_dim))  # Random action

        # Step 2: Convert the state to a PyTorch tensor
        # unsqueeze(0) adds batch dimension (PyTorch expects batch even if size=1)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Step 3: Predict Q-values without tracking gradients (we're just choosing action, not training)
        with torch.no_grad():
            q = self.model(s)  # Get Q-values for all possible actions
            action = int(q.argmax(dim=1).item())  # Choose action with highest Q-value

        # Step 4: Return the chosen action
        return action

    # --------------------------
    # Train on a single experience
    # --------------------------
    def train_step(self, state, action, reward, next_state, done):
        """
        Train the network on a single experience tuple (s, a, r, s', done)
        """
        # Convert everything to tensors and move to device
        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        r  = torch.FloatTensor([reward]).to(self.device)
        d  = torch.FloatTensor([float(done)]).to(self.device)  # 1 if episode ended, else 0
        a  = torch.LongTensor([int(action)]).to(self.device)

        # Step 1: Predict Q-value for the chosen action using main network
        q_values = self.model(s)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        # gather picks the Q-value corresponding to the action taken

        # Step 2: Compute target Q-value using Double DQN
        with torch.no_grad():
            # Online network predicts the best next action
            next_q_online = self.model(ns)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Target network evaluates that action (more stable)
            next_q_target = self.target_model(ns)
            selected_next_q = next_q_target.gather(1, next_actions).squeeze(1)

            # Target Q-value formula: r + gamma * max Q(next) * (1 - done)
            expected_q = r + self.gamma * selected_next_q * (1.0 - d)

        # Step 3: Compute loss and backpropagate
        loss = self.loss_fn(q_value.view(-1), expected_q.view(-1).detach())
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()              # Compute new gradients
        self.optimizer.step()        # Update network weights

    # --------------------------
    # Train on a batch of experiences
    # --------------------------
    def batch_train(self, states, actions, rewards, next_states, dones):
        """
        Train the network on multiple experiences at once (batch)
        This is faster and more stable than training on single experiences
        """
        # Convert batch inputs to tensors
        s  = torch.FloatTensor(states).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)

        # Step 1: Predict Q-values for all actions in the batch
        q_values = self.model(s)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        # Step 2: Compute target Q-values for the batch using Double DQN
        with torch.no_grad():
            next_q_online = self.model(ns)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_model(ns)
            selected_next_q = next_q_target.gather(1, next_actions).squeeze(1)
            expected_q = r + self.gamma * selected_next_q * (1.0 - d)

        # Step 3: Compute loss and update weights
        loss = self.loss_fn(q_value, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --------------------------
    # Update target network
    # --------------------------
    def update_target(self):
        """
        Copy main network weights to target network.
        Target network changes slowly → stabilizes training.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    # --------------------------
    # Decay epsilon
    # --------------------------
    def decay_epsilon(self):
        """
        Reduce exploration rate over time so agent gradually exploits more
        """
        # epsilon never goes below epsilon_min
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
