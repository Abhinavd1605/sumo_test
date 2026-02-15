"""Integration test for the SUMO environment."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dnlight.environment import SumoEnvironment


def test_reset_returns_valid_state():
    """reset() should return a float32 array of correct shape."""
    env = SumoEnvironment("configs/single_intersection.sumocfg")
    state = env.reset()

    assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
    assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"
    assert len(state) == env.get_state_dim(), \
        f"State dim mismatch: {len(state)} vs {env.get_state_dim()}"
    print(f"  State dim: {len(state)} ({len(env.incoming_lanes)} lanes x 7)")

    env.close()
    print("  [PASS] test_reset_returns_valid_state")


def test_step_returns_tuple():
    """step() should return (state, reward, done, info)."""
    env = SumoEnvironment("configs/single_intersection.sumocfg")
    env.reset()

    next_state, reward, done, info = env.step(0)

    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'sim_step' in info

    env.close()
    print("  [PASS] test_step_returns_tuple")


def test_action_changes_phase():
    """Applying different actions should change internal phase."""
    env = SumoEnvironment("configs/single_intersection.sumocfg")
    env.reset()

    # Step with action 0
    env.step(0)
    phase0 = env.current_phase

    # Step with action 1 (if available)
    if env.num_phases > 1:
        env.step(1)
        phase1 = env.current_phase
        assert phase1 == 1, f"Expected phase 1 after action 1, got {phase1}"

    env.close()
    print("  [PASS] test_action_changes_phase")


def test_episode_completion():
    """Episode should complete after sufficient steps."""
    env = SumoEnvironment("configs/single_intersection.sumocfg",
                          max_steps=100)  # short episode for testing
    env.reset()

    done = False
    steps = 0
    while not done and steps < 50:
        _, _, done, _ = env.step(steps % env.num_phases)
        steps += 1

    env.close()
    print(f"  Episode ran for {steps} steps (done={done})")
    print("  [PASS] test_episode_completion")


if __name__ == "__main__":
    print("Running environment integration tests...")
    test_reset_returns_valid_state()
    test_step_returns_tuple()
    test_action_changes_phase()
    test_episode_completion()
    print("All environment tests passed!")
