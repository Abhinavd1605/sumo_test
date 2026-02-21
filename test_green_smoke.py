"""Quick smoke test for multi-intersection environment and green reward."""
import sys

def test_multi_env():
    from dnlight.environment import MultiIntersectionEnv
    env = MultiIntersectionEnv('configs/grid_network.sumocfg')
    states = env.reset()
    print(f"TLS IDs: {env.tls_ids}")
    print(f"Num intersections: {env.num_intersections}")
    print(f"State dim: {env.state_dim}")
    for k, v in states.items():
        print(f"  {k}: shape={v.shape}")

    # One step
    actions = {t: 0 for t in env.tls_ids}
    next_s, rewards, done, info = env.step(actions)
    print(f"Rewards: {rewards}")
    em = info.get('emissions', {})
    for tid, em_data in em.items():
        co2 = em_data.get('co2_mg_per_s', 0)
        print(f"  {tid} CO2: {co2:.1f} mg/s")
    print(f"Done: {done}")
    env.close()
    print("Multi-intersection smoke test PASSED")


def test_single_env_emissions():
    from dnlight.environment import SumoEnvironment
    env = SumoEnvironment('configs/single_intersection.sumocfg')
    state = env.reset()
    print(f"\nSingle-intersection state dim: {env.state_dim}")

    action = 0
    state, reward, done, info = env.step(action)
    em = info.get('emissions', {})
    print(f"Emissions: CO2={em.get('co2_mg_per_s', 0):.1f}, "
          f"Fuel={em.get('fuel_ml_per_s', 0):.1f}")
    env.close()
    print("Single-intersection emission test PASSED")


def test_carbon_tracker():
    import time
    from dnlight.carbon_tracker import CarbonTracker
    ct = CarbonTracker()
    ct.start_step()
    time.sleep(0.1)
    result = ct.end_step()
    print(f"\nCarbon tracker: {result}")
    summary = ct.get_summary()
    print(f"Summary: {summary}")
    print("Carbon tracker test PASSED")


def test_baselines():
    from dnlight.baselines import FixedTimeController
    ctrl = FixedTimeController()
    actions = []
    for _ in range(15):
        a = ctrl.select_action()
        actions.append(a)
        ctrl.update()
    print(f"\nFixed-time actions over 15 steps: {actions}")
    assert 0 in actions and 1 in actions, "Should cycle through phases"
    print("Fixed-time baseline test PASSED")


def test_green_reward():
    import numpy as np
    from dnlight.reward import compute_total_reward, compute_green_reward

    lane_wt = np.array([5.0, 10.0, 3.0, 8.0], dtype=np.float32)
    emv = []
    social = [{'wait_time': 5, 'speed': 10, 'time_loss': 1}]
    lane_data = {'queue_ratio': 0.3, 'flow_rate': 10, 'speed_variance': 2.0,
                 'throughput': 15}
    emissions = {'co2_mg_per_s': 50000, 'fuel_ml_per_s': 20, 'nox_mg_per_s': 100}

    base = compute_total_reward(lane_wt, emv, social, lane_data)
    green = compute_green_reward(lane_wt, emv, social, lane_data, emissions)
    print(f"\nBase reward: {base:.4f}")
    print(f"Green reward: {green:.4f}")
    assert green < base, "Green reward should be lower (more penalty)"
    print("Green reward test PASSED")


if __name__ == "__main__":
    test_green_reward()
    test_baselines()
    test_carbon_tracker()
    test_single_env_emissions()
    test_multi_env()
    print("\n=== ALL SMOKE TESTS PASSED ===")
