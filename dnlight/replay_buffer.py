"""
Prioritized Experience Replay (PER) for DNLight.

Uses a SumTree data structure for O(log N) proportional sampling.
Priorities are based on TD error: p_i = (|delta_i| + eps)^alpha.
"""
import numpy as np


class SumTree:
    """
    Binary tree where each leaf holds a priority value.
    Parent nodes store the sum of their children, enabling
    O(log N) proportional sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Update parent sums after a priority change."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find the leaf index for a given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    @property
    def max_priority(self):
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 1.0
        return float(np.max(self.tree[leaf_start:leaf_end]))

    def add(self, priority, data):
        """Add a new experience with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """Update the priority of a leaf node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """
        Get the leaf index, priority, and data for a cumulative sum s.

        Returns:
            (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Stores transitions and samples them proportional to their
    TD-error-based priority.

    Hyperparameters (from spec):
        capacity:  500,000
        alpha:     0.6  (priority exponent)
        beta_start: 0.5 (IS weight correction, anneals to 1.0)
        epsilon:   1e-6 (small constant for non-zero priority)
    """

    def __init__(self, capacity=500_000, alpha=0.6,
                 beta_start=0.5, beta_end=1.0,
                 beta_anneal_steps=200_000, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self._step = 0

    def __len__(self):
        return self.tree.n_entries

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition with maximum current priority.
        New experiences get max priority to ensure they are sampled at least once.
        """
        max_p = self.tree.max_priority
        if max_p == 0:
            max_p = 1.0
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_p, experience)

    def sample(self, batch_size):
        """
        Sample a batch proportional to priorities.

        Returns:
            states, actions, rewards, next_states, dones,
            is_weights (importance sampling), tree_indices
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        # Anneal beta
        self._step += 1
        frac = min(self._step / self.beta_anneal_steps, 1.0)
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)

            if data is None:
                # Fallback: sample again from anywhere
                s = np.random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities, dtype=np.float64)
        sampling_probs = priorities / (self.tree.total + 1e-10)
        n = self.tree.n_entries

        is_weights = np.power(n * sampling_probs + 1e-10, -self.beta)
        is_weights /= is_weights.max()  # Normalize
        is_weights = is_weights.astype(np.float32)

        # Unpack batch
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, is_weights, indices

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on new TD errors.

        p_i = (|delta_i| + epsilon)^alpha
        """
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
