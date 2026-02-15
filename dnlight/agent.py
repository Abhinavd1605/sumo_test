"""
DNLight Agent: Double Dueling DQN with NoisyNet.

Architecture:
    Input(state_dim) -> Dense(512, SiLU) + Dropout(0.2) -> NoisyLinear(512)
      |-- Value stream:     NoisyLinear(256) -> NoisyLinear(1)   -> V(s)
      |-- Advantage stream: NoisyLinear(256) -> NoisyLinear(n_a) -> A(s,a)
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

Features:
    - Double DQN: online network selects actions, target evaluates
    - Dueling architecture: separate V and A streams
    - NoisyNet: structured exploration via learned noise
    - Hybrid exploration: NoisyNet + epsilon-greedy with decay
    - Gradient clipping (max norm 5.0)
"""
import os
import numpy as np
import tensorflow as tf
from dnlight.noise import NoisyLinear
from dnlight.replay_buffer import PrioritizedReplayBuffer
from dnlight.attention import AttentionComm


# ── Hyperparameters (from spec) ──────────────────────────────────────
BATCH_SIZE = 512
GAMMA = 0.99           # discount factor
LR = 3e-4              # Adam learning rate
TARGET_UPDATE = 1500    # steps between target sync
GRAD_CLIP = 5.0        # gradient clipping max norm
DROPOUT = 0.2
NOISY_STD = 0.1
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 50_000     # steps
BUFFER_SIZE = 500_000
HIDDEN_DIM = 512
VALUE_DIM = 256
ADV_DIM = 256
NOISE_DECAY_STEPS = 100_000  # steps to decay noise from 1.0 to 0.05


def build_dueling_network(state_dim: int, action_dim: int,
                          name: str = "online") -> tf.keras.Model:
    """
    Build the Dueling DQN network with NoisyNet layers.

    Args:
        state_dim: Flattened input state dimension.
        action_dim: Number of discrete actions.
        name: Model name prefix.

    Returns:
        A Keras Model that outputs Q-values of shape (batch, action_dim).
    """
    inputs = tf.keras.Input(shape=(state_dim,), name=f"{name}_input")

    # Shared feature layer
    x = tf.keras.layers.Dense(
        HIDDEN_DIM, activation='silu', name=f"{name}_shared_dense"
    )(inputs)
    x = tf.keras.layers.Dropout(DROPOUT, name=f"{name}_dropout")(x)
    x = NoisyLinear(HIDDEN_DIM, noisy_std=NOISY_STD,
                    name=f"{name}_shared_noisy")(x)
    x = tf.keras.layers.Activation('silu', name=f"{name}_shared_act")(x)

    # ── Value stream ─────────────────────────────────────────────
    v = NoisyLinear(VALUE_DIM, noisy_std=NOISY_STD,
                    name=f"{name}_val_noisy1")(x)
    v = tf.keras.layers.Activation('silu', name=f"{name}_val_act")(v)
    v = NoisyLinear(1, noisy_std=NOISY_STD,
                    name=f"{name}_val_noisy2")(v)  # (batch, 1)

    # ── Advantage stream ─────────────────────────────────────────
    a = NoisyLinear(ADV_DIM, noisy_std=NOISY_STD,
                    name=f"{name}_adv_noisy1")(x)
    a = tf.keras.layers.Activation('silu', name=f"{name}_adv_act")(a)
    a = NoisyLinear(action_dim, noisy_std=NOISY_STD,
                    name=f"{name}_adv_noisy2")(a)  # (batch, action_dim)

    # ── Dueling aggregation ──────────────────────────────────────
    # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    q = v + (a - tf.reduce_mean(a, axis=-1, keepdims=True))

    model = tf.keras.Model(inputs=inputs, outputs=q, name=f"{name}_dqn")
    return model


class DNLightAgent:
    """
    DNLight agent using Double Dueling DQN + NoisyNet + PER.

    Manages online/target networks, experience replay, exploration,
    and training logic.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 use_attention: bool = False):
        """
        Args:
            state_dim: Flattened state dimension.
            action_dim: Number of discrete actions.
            use_attention: Whether to use attention communication.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention

        # Build online and target networks
        self.online_net = build_dueling_network(
            state_dim, action_dim, name="online"
        )
        self.target_net = build_dueling_network(
            state_dim, action_dim, name="target"
        )
        self.sync_target()  # initialize target with online weights

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        # PER buffer
        self.buffer = PrioritizedReplayBuffer(capacity=BUFFER_SIZE)

        # Attention (optional, for multi-agent)
        if use_attention:
            self.attention = AttentionComm(state_dim, message_dim=64)
        else:
            self.attention = None

        # Exploration state
        self.epsilon = EPS_START
        self.global_step = 0
        self.noise_scale = 1.0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using hybrid exploration (NoisyNet + epsilon-greedy).

        During training:
            - With probability epsilon: random action
            - Otherwise: NoisyNet-driven Q-value selection

        During evaluation:
            - Pure greedy (no noise, no epsilon)

        Args:
            state: State vector of shape (state_dim,).
            training: Whether we are training (enables exploration).

        Returns:
            Chosen action index.
        """
        if training:
            # Update epsilon
            self.epsilon = max(
                EPS_END,
                EPS_START - (EPS_START - EPS_END) *
                (self.global_step / EPS_DECAY)
            )

            # Epsilon-greedy component
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_dim)

        # Forward pass through online network
        state_t = tf.constant(state.reshape(1, -1), dtype=tf.float32)
        q_values = self.online_net(state_t, training=training)
        return int(tf.argmax(q_values[0]).numpy())

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones,
                    is_weights):
        """
        Perform one training step with Double DQN loss.

        TD target: y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
        Loss: IS_weight * Huber(y - Q_online(s, a))

        Returns:
            (loss, td_errors)
        """
        # Double DQN: online selects best action, target evaluates
        next_q_online = self.online_net(next_states, training=False)
        best_actions = tf.argmax(next_q_online, axis=-1)  # (batch,)

        next_q_target = self.target_net(next_states, training=False)
        best_action_onehot = tf.one_hot(best_actions, self.action_dim)
        next_q_values = tf.reduce_sum(
            next_q_target * best_action_onehot, axis=-1
        )  # (batch,)

        # TD target
        targets = rewards + GAMMA * next_q_values * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            action_onehot = tf.one_hot(actions, self.action_dim)
            q_selected = tf.reduce_sum(
                q_values * action_onehot, axis=-1
            )  # (batch,)

            # TD errors for PER priority update
            td_errors = targets - q_selected

            # Huber loss weighted by importance sampling
            huber = tf.keras.losses.Huber(reduction='none')
            element_losses = huber(targets, q_selected)
            loss = tf.reduce_mean(is_weights * element_losses)

        # Gradient clipping
        grads = tape.gradient(loss, self.online_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP)
        self.optimizer.apply_gradients(
            zip(grads, self.online_net.trainable_variables)
        )

        return loss, td_errors

    def train(self) -> dict:
        """
        Sample from PER, perform a training step, update priorities.

        Returns:
            Dict with 'loss', 'td_error_mean', 'epsilon', 'noise_scale'
            or empty dict if buffer too small.
        """
        if len(self.buffer) < BATCH_SIZE:
            return {}

        self.global_step += 1

        # Reset noise in NoisyNet layers
        self._reset_noise(self.online_net)

        # Update global noise scale (linear decay)
        self.noise_scale = max(
            0.05,
            1.0 - (self.global_step / NOISE_DECAY_STEPS) * 0.95
        )
        self._set_noise_scale(self.online_net, self.noise_scale)

        # Sample from PER
        (states, actions, rewards, next_states, dones,
         is_weights, indices) = self.buffer.sample(BATCH_SIZE)

        # Convert to tensors
        states_t = tf.constant(states, dtype=tf.float32)
        actions_t = tf.constant(actions, dtype=tf.int32)
        rewards_t = tf.constant(rewards, dtype=tf.float32)
        next_states_t = tf.constant(next_states, dtype=tf.float32)
        dones_t = tf.constant(dones, dtype=tf.float32)
        is_weights_t = tf.constant(is_weights, dtype=tf.float32)

        # Train
        loss, td_errors = self._train_step(
            states_t, actions_t, rewards_t, next_states_t,
            dones_t, is_weights_t
        )

        # Update PER priorities
        td_np = td_errors.numpy()
        self.buffer.update_priorities(indices, td_np)

        # Sync target if needed
        if self.global_step % TARGET_UPDATE == 0:
            self.sync_target()

        return {
            'loss': float(loss.numpy()),
            'td_error_mean': float(np.mean(np.abs(td_np))),
            'epsilon': self.epsilon,
            'noise_scale': self.noise_scale,
        }

    def sync_target(self):
        """Copy online network weights to target network."""
        self.target_net.set_weights(self.online_net.get_weights())

    def _reset_noise(self, model: tf.keras.Model):
        """Reset noise in all NoisyLinear layers."""
        for layer in model.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def _set_noise_scale(self, model: tf.keras.Model, scale: float):
        """Set global noise scale in all NoisyLinear layers."""
        for layer in model.layers:
            if isinstance(layer, NoisyLinear):
                layer.noise_scale.assign(scale)

    def store_transition(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def save(self, path: str):
        """Save online network weights and training state."""
        import json
        self.online_net.save_weights(path)
        meta = {
            'epsilon': float(self.epsilon),
            'noise_scale': float(self.noise_scale),
            'step_count': int(self.global_step),
        }
        with open(path + '.meta.json', 'w') as f:
            json.dump(meta, f)

    def load(self, path: str):
        """Load online network weights, training state, and sync target."""
        import json
        self.online_net.load_weights(path)
        self.sync_target()
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.epsilon = meta.get('epsilon', self.epsilon)
            self.noise_scale = meta.get('noise_scale', self.noise_scale)
            self.global_step = meta.get('step_count', self.global_step)
            print(f"  Restored state: eps={self.epsilon:.4f}, "
                  f"noise={self.noise_scale:.4f}, steps={self.global_step}")
