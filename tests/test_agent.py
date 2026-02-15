"""Unit tests for the DNLight agent and NoisyNet."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from dnlight.agent import DNLightAgent, build_dueling_network
from dnlight.noise import NoisyLinear
from dnlight.replay_buffer import PrioritizedReplayBuffer


def test_network_output_shape():
    """Network should output (batch, action_dim) Q-values."""
    state_dim = 56  # 8 lanes * 7 features
    action_dim = 4
    model = build_dueling_network(state_dim, action_dim, name="test")

    batch = tf.random.normal((32, state_dim))
    output = model(batch, training=False)
    assert output.shape == (32, 4), f"Expected (32, 4), got {output.shape}"
    print("  [PASS] test_network_output_shape")


def test_noisy_reset_changes_output():
    """Resetting noise should change the network output."""
    state_dim = 56
    action_dim = 4
    model = build_dueling_network(state_dim, action_dim, name="noisy_test")

    batch = tf.random.normal((8, state_dim))

    # Get output before noise reset
    out1 = model(batch, training=True).numpy()

    # Reset noise
    for layer in model.layers:
        if isinstance(layer, NoisyLinear):
            layer.reset_noise()

    out2 = model(batch, training=True).numpy()

    # Outputs should differ (with very high probability)
    assert not np.allclose(out1, out2, atol=1e-6), \
        "Outputs should differ after noise reset"
    print("  [PASS] test_noisy_reset_changes_output")


def test_target_sync():
    """After sync, target should produce identical outputs."""
    state_dim = 28
    action_dim = 4
    agent = DNLightAgent(state_dim, action_dim)

    # Ensure online and target are synced
    agent.sync_target()

    batch = tf.random.normal((8, state_dim))
    online_out = agent.online_net(batch, training=False).numpy()
    target_out = agent.target_net(batch, training=False).numpy()

    assert np.allclose(online_out, target_out, atol=1e-6), \
        "Online and target outputs should match after sync"
    print("  [PASS] test_target_sync")


def test_per_basic():
    """PER should store and sample transitions."""
    buf = PrioritizedReplayBuffer(capacity=100)

    # Add some transitions
    for i in range(50):
        state = np.random.randn(28).astype(np.float32)
        next_state = np.random.randn(28).astype(np.float32)
        buf.add(state, i % 4, -1.0, next_state, False)

    assert len(buf) == 50

    # Sample
    states, actions, rewards, next_states, dones, weights, indices = \
        buf.sample(16)

    assert states.shape == (16, 28)
    assert actions.shape == (16,)
    assert weights.shape == (16,)
    assert len(indices) == 16
    print("  [PASS] test_per_basic")


def test_agent_action_selection():
    """Agent should return valid action indices."""
    state_dim = 28
    action_dim = 4
    agent = DNLightAgent(state_dim, action_dim)

    state = np.random.randn(state_dim).astype(np.float32)

    # Training mode
    action = agent.select_action(state, training=True)
    assert 0 <= action < action_dim, f"Invalid action: {action}"

    # Eval mode
    action = agent.select_action(state, training=False)
    assert 0 <= action < action_dim, f"Invalid action: {action}"
    print("  [PASS] test_agent_action_selection")


def test_no_nan_gradients():
    """Training step should not produce NaN gradients."""
    state_dim = 28
    action_dim = 4
    agent = DNLightAgent(state_dim, action_dim)

    # Fill buffer with enough transitions
    for i in range(600):
        s = np.random.randn(state_dim).astype(np.float32)
        ns = np.random.randn(state_dim).astype(np.float32)
        agent.store_transition(s, i % 4, -1.0, ns, False)

    # Train one step
    info = agent.train()
    assert info, "Expected training info dict"
    assert not np.isnan(info['loss']), f"Loss is NaN!"
    assert not np.isnan(info['td_error_mean']), f"TD error is NaN!"
    print("  [PASS] test_no_nan_gradients")


if __name__ == "__main__":
    print("Running agent tests...")
    test_network_output_shape()
    test_noisy_reset_changes_output()
    test_target_sync()
    test_per_basic()
    test_agent_action_selection()
    test_no_nan_gradients()
    print("All agent tests passed!")
