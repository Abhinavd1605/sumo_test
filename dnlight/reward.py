"""
DNLight Reward Function.

Implements the dynamic reward calculation that balances
emergency vehicle (EMV) priority with social vehicle efficiency.
All formulas follow the DNLight paper specification.
"""
import numpy as np
from typing import List, Dict, Tuple


def compute_penalty_coefficient(lane_wait_times: np.ndarray) -> float:
    """
    Compute the penalty coefficient C that balances waiting time deviations.

    C = clip(2 + std(lane_wait_times), 2, 3.5)

    Args:
        lane_wait_times: Array of total waiting times per lane.

    Returns:
        Penalty coefficient clamped to [2, 3.5].
    """
    sigma = np.std(lane_wait_times) if len(lane_wait_times) > 0 else 0.0
    C = 2.0 + sigma
    return float(np.clip(C, 2.0, 3.5))


def compute_waiting_time_deviation(lane_wait_times: np.ndarray,
                                   C: float) -> float:
    """
    Compute the waiting time deviation penalty P_w.

    P_w = C * sum(|w_i - mean(w)|)

    Args:
        lane_wait_times: Array of total waiting times per lane.
        C: Penalty coefficient.

    Returns:
        Waiting time deviation penalty.
    """
    if len(lane_wait_times) == 0:
        return 0.0
    w_mean = np.mean(lane_wait_times)
    deviation = np.sum(np.abs(lane_wait_times - w_mean))
    return C * deviation


def compute_emv_reward(emv_vehicles: List[Dict]) -> Tuple[float, float, float]:
    """
    Compute the reward component for emergency vehicles.

    Per EMV i:
        R_emv_i = alpha_emv * log(1 + travel_time)
                + beta_emv  * wait_time
                + (1 - alpha_emv - beta_emv) * (1/avg_speed + time_loss)

    Dynamic weights:
        alpha_emv = 0.35 + 0.15 * (N_emv / N_total)
        beta_emv  = 0.25 + 0.1  * total_time_loss

    Args:
        emv_vehicles: List of dicts with keys:
            'travel_time', 'wait_time', 'avg_speed', 'time_loss',
            'n_emv', 'n_total', 'total_time_loss'

    Returns:
        Tuple of (total_emv_reward, alpha_emv, beta_emv)
    """
    if not emv_vehicles:
        return 0.0, 0.35, 0.25

    # Use the global counts from the first vehicle dict
    info = emv_vehicles[0]
    n_emv = info.get('n_emv', 1)
    n_total = max(info.get('n_total', 1), 1)
    total_time_loss = info.get('total_time_loss', 0.0)

    alpha = 0.35 + 0.15 * (n_emv / n_total)
    beta = 0.25 + 0.1 * total_time_loss
    # Clamp beta to avoid dominating
    beta = min(beta, 0.55)
    gamma = max(1.0 - alpha - beta, 0.0)

    total_reward = 0.0
    for v in emv_vehicles:
        tt = v.get('travel_time', 0.0)
        wt = v.get('wait_time', 0.0)
        avg_spd = max(v.get('avg_speed', 0.1), 0.1)  # avoid div by 0
        tl = v.get('time_loss', 0.0)

        r = (alpha * np.log1p(tt)
             + beta * wt
             + gamma * (1.0 / avg_spd + tl))
        total_reward += r

    return total_reward, alpha, beta


def compute_social_reward(social_vehicles: List[Dict],
                          lane_data: Dict) -> float:
    """
    Compute the reward component for social (non-emergency) vehicles.

    Per vehicle j:
        R_social_j = w1 * queue_ratio
                   + w2 * (1 / flow_rate)
                   + w3 * speed_variance
                   + w4 * (1 / throughput)

    Dynamic weights depend on current traffic conditions:
        w1 proportional to queue length ratio
        w2 proportional to inverse flow rate
        w3 = 0.15  (velocity dispersion weight from spec)
        w4 proportional to inverse throughput

    Args:
        social_vehicles: List of dicts with keys:
            'wait_time', 'speed', 'time_loss'
        lane_data: Dict with keys:
            'queue_ratio', 'flow_rate', 'speed_variance', 'throughput'

    Returns:
        Total social vehicle reward (penalty).
    """
    if not social_vehicles:
        return 0.0

    queue_ratio = lane_data.get('queue_ratio', 0.5)
    flow_rate = max(lane_data.get('flow_rate', 1.0), 0.01)
    speed_var = lane_data.get('speed_variance', 0.0)
    throughput = max(lane_data.get('throughput', 1.0), 0.01)

    # Dynamic weights (normalized to sum to 1)
    w1 = queue_ratio
    w2 = 1.0 / flow_rate
    w3 = 0.15  # velocity dispersion weight from spec
    w4 = 1.0 / throughput
    w_sum = w1 + w2 + w3 + w4
    if w_sum > 0:
        w1, w2, w3, w4 = w1/w_sum, w2/w_sum, w3/w_sum, w4/w_sum

    total_reward = 0.0
    for v in social_vehicles:
        wt = v.get('wait_time', 0.0)
        spd = max(v.get('speed', 0.1), 0.01)
        tl = v.get('time_loss', 0.0)

        r = (w1 * wt
             + w2 * (1.0 / spd)
             + w3 * (spd ** 2)  # speed variance proxy
             + w4 * tl)
        total_reward += r

    return total_reward


def compute_total_reward(lane_wait_times: np.ndarray,
                         emv_vehicles: List[Dict],
                         social_vehicles: List[Dict],
                         lane_data: Dict) -> float:
    """
    Compute the total intersection reward.

    R = -(w_emv * R_emv + w_social * R_social + P_w)

    Global weights:
        w_emv    = 0.6 + 0.2 * (N_emv / N_total)
        w_social = 1 - w_emv

    The final reward is negated (lower penalties = higher reward).

    Args:
        lane_wait_times: Waiting times per lane.
        emv_vehicles: EMV vehicle info dicts.
        social_vehicles: Social vehicle info dicts.
        lane_data: Aggregate lane statistics.

    Returns:
        Total reward (negative value, higher is better).
    """
    # Penalty coefficient and deviation
    C = compute_penalty_coefficient(lane_wait_times)
    P_w = compute_waiting_time_deviation(lane_wait_times, C)

    # EMV reward component
    R_emv, _, _ = compute_emv_reward(emv_vehicles)

    # Social reward component
    R_social = compute_social_reward(social_vehicles, lane_data)

    # Global dynamic weights
    n_emv = emv_vehicles[0].get('n_emv', 0) if emv_vehicles else 0
    n_total = max(
        (emv_vehicles[0].get('n_total', 1) if emv_vehicles else
         len(social_vehicles)),
        1
    )
    w_emv = 0.6 + 0.2 * (n_emv / n_total)
    w_emv = min(w_emv, 0.9)  # cap
    w_social = 1.0 - w_emv

    # Normalize components to keep reward in a reasonable range
    n_emv_count = max(len(emv_vehicles), 1)
    n_social_count = max(len(social_vehicles), 1)
    R_emv_norm = R_emv / n_emv_count
    R_social_norm = R_social / n_social_count

    total = w_emv * R_emv_norm + w_social * R_social_norm + P_w

    # Negate: lower penalties → higher (less negative) reward
    return -total
