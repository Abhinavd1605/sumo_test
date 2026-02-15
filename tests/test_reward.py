"""Unit tests for the DNLight reward function."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dnlight.reward import (
    compute_penalty_coefficient,
    compute_waiting_time_deviation,
    compute_emv_reward,
    compute_social_reward,
    compute_total_reward,
)


def test_penalty_coefficient_clamping():
    """C should be clamped to [2, 3.5]."""
    # Very low std -> C = 2
    wt = np.array([10.0, 10.0, 10.0, 10.0])
    C = compute_penalty_coefficient(wt)
    assert C == 2.0, f"Expected C=2.0 for zero std, got {C}"

    # High std -> C = 3.5
    wt = np.array([0.0, 100.0, 0.0, 100.0])
    C = compute_penalty_coefficient(wt)
    assert C == 3.5, f"Expected C=3.5 for high std, got {C}"

    # Medium std -> between 2 and 3.5
    wt = np.array([5.0, 10.0, 15.0, 20.0])
    C = compute_penalty_coefficient(wt)
    assert 2.0 <= C <= 3.5, f"Expected C in [2, 3.5], got {C}"
    print("  [PASS] test_penalty_coefficient_clamping")


def test_waiting_time_deviation():
    """P_w should be C * sum(|w_i - mean(w)|)."""
    wt = np.array([10.0, 20.0, 30.0, 40.0])
    C = 2.5
    P_w = compute_waiting_time_deviation(wt, C)
    # mean = 25, deviations: 15+5+5+15 = 40
    expected = 2.5 * 40.0
    assert abs(P_w - expected) < 1e-6, f"Expected {expected}, got {P_w}"
    print("  [PASS] test_waiting_time_deviation")


def test_emv_reward_nonzero():
    """EMV reward should be positive (it's a penalty)."""
    emvs = [{
        'travel_time': 100.0,
        'wait_time': 20.0,
        'avg_speed': 5.0,
        'time_loss': 30.0,
        'n_emv': 2,
        'n_total': 50,
        'total_time_loss': 60.0,
    }]
    R, alpha, beta = compute_emv_reward(emvs)
    assert R > 0, f"Expected positive EMV reward, got {R}"
    assert 0.35 <= alpha <= 0.5, f"alpha out of range: {alpha}"
    print("  [PASS] test_emv_reward_nonzero")


def test_total_reward_negative():
    """Total reward should be negative (penalties are negated)."""
    wt = np.array([10.0, 20.0, 15.0, 25.0])
    emvs = [{
        'travel_time': 50.0, 'wait_time': 10.0,
        'avg_speed': 8.0, 'time_loss': 5.0,
        'n_emv': 1, 'n_total': 30, 'total_time_loss': 5.0,
    }]
    socials = [
        {'wait_time': 5.0, 'speed': 10.0, 'time_loss': 2.0},
        {'wait_time': 8.0, 'speed': 7.0, 'time_loss': 3.0},
    ]
    lane_data = {
        'queue_ratio': 0.3, 'flow_rate': 5.0,
        'speed_variance': 4.0, 'throughput': 10.0,
    }
    R = compute_total_reward(wt, emvs, socials, lane_data)
    assert R < 0, f"Expected negative total reward, got {R}"
    print("  [PASS] test_total_reward_negative")


def test_empty_vehicles():
    """Should handle empty vehicle lists gracefully."""
    wt = np.array([5.0, 5.0])
    R = compute_total_reward(wt, [], [], {'queue_ratio': 0, 'flow_rate': 1,
                                          'speed_variance': 0, 'throughput': 1})
    assert isinstance(R, float), f"Expected float, got {type(R)}"
    print("  [PASS] test_empty_vehicles")


if __name__ == "__main__":
    print("Running reward tests...")
    test_penalty_coefficient_clamping()
    test_waiting_time_deviation()
    test_emv_reward_nonzero()
    test_total_reward_negative()
    test_empty_vehicles()
    print("All reward tests passed!")
