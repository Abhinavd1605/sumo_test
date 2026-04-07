"""
SUMO Environment Wrapper for DNLight.

Manages the SUMO simulation via TraCI, providing a standard
RL environment interface (reset, step) with:
  - 7-element per-lane state vectors
  - 4-action discrete action space (signal phases)
  - Dynamic reward computation
  - 10s step duration + 3s yellow buffer
"""
import os
import sys
import numpy as np
from typing import Tuple, Dict, List, Optional

import traci
import traci.constants as tc

from dnlight.reward import compute_total_reward, compute_green_reward

# Constants from spec
DETECTION_RANGE = 200.0   # metres
STEP_DURATION = 10        # seconds per RL step
YELLOW_DURATION = 3       # seconds for yellow phase
SIM_LENGTH = 3600         # seconds per episode
VEHICLE_LENGTH = 4.5      # metres (average)


class SumoEnvironment:
    """
    SUMO-TraCI environment for single-intersection traffic signal control.

    State: For each incoming lane, a 7-element vector:
        [phase, queue_length, wait_time, emv_presence,
         emv_position, emv_speed, neighbor_info]

    Action: Discrete(4) - one of 4 signal phases.
    """

    def __init__(self,
                 sumocfg_path: str,
                 use_gui: bool = False,
                 step_duration: int = STEP_DURATION,
                 yellow_duration: int = YELLOW_DURATION,
                 max_steps: int = SIM_LENGTH,
                 label: str = "default",
                 use_green_reward: bool = False,
                 alpha_co2: float = 0.3):
        """
        Args:
            sumocfg_path: Path to .sumocfg file.
            use_gui: If True, use sumo-gui instead of sumo.
            step_duration: RL step duration in simulation seconds.
            yellow_duration: Yellow phase duration in seconds.
            max_steps: Maximum simulation time in seconds.
            label: TraCI connection label.
            use_green_reward: If True, use CO2-penalized green reward.
            alpha_co2: Weight for CO2 penalty in green reward.
            seed: Fixed seed for SUMO randomization (optional).
        """
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.use_gui = use_gui
        self.step_duration = step_duration
        self.yellow_duration = yellow_duration
        self.max_steps = max_steps
        self.label = label
        self.use_green_reward = use_green_reward
        self.alpha_co2 = alpha_co2
        self.seed = None  # Updated on reset

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.conn = None  # traci connection

        # Will be populated after first reset
        self.tls_id = None             # traffic light ID
        self.incoming_lanes = []       # list of incoming lane IDs
        self.num_phases = 4            # NS-straight, NS-left, EW-straight, EW-left
        self.current_phase = 0
        self.is_yellow = False
        self.sim_step = 0
        self.episode = 0

        # State dimensions
        self.features_per_lane = 7
        self.state_dim = None  # set after discovering lanes

    @property
    def action_dim(self) -> int:
        return self.num_phases

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the simulation and return the initial state."""
        self.seed = seed
        # Close existing connection
        if self.conn is not None:
            try:
                traci.close()
            except Exception:
                pass

        # Start SUMO
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_path,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--start", "true",
        ]
        if self.seed is not None:
            sumo_cmd.extend(["--seed", str(self.seed)])
        else:
            sumo_cmd.append("--random")

        traci.start(sumo_cmd, label=self.label)
        self.conn = traci

        # Discover traffic light and lanes
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in the network!")
        self.tls_id = tls_ids[0]

        # Get incoming lanes (controlled by this TLS)
        # Use getControlledLanes which returns actual lane IDs directly
        lanes_str = traci.trafficlight.getControlledLanes(self.tls_id)
        self.incoming_lanes = sorted(set(lanes_str))

        self.state_dim = len(self.incoming_lanes) * self.features_per_lane

        # Discover the TLS state string length from the current program
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)
        if logic:
            self._state_len = len(logic[0].phases[0].state)
        else:
            self._state_len = 16  # default for 4-arm, 2-lane

        # Define 4 green phases using setRedYellowGreenState
        # State string has one char per connection. For our 4-arm, 2-lane
        # intersection, the auto-generated pattern is:
        #   GGGgrrrrGGGgrrrr  (NS green)
        #   rrrrGGGgrrrrGGGg  (EW green)
        # We split each into straight + left-turn sub-phases:
        n = self._state_len
        if n == 16:
            # 16 connections: 4 per approach (2 lanes x 2 turns)
            # Indices 0-3: N approach, 4-7: E approach,
            #          8-11: S approach, 12-15: W approach
            self._green_states = [
                'GGrgrrrrGGrgrrrr',  # Phase 0: NS straight
                'rrGgrrrrrrGgrrrr',  # Phase 1: NS left turn
                'rrrrGGrgrrrrGGrg',  # Phase 2: EW straight
                'rrrrrrGgrrrrrrGg',  # Phase 3: EW left turn
            ]
        else:
            # Generic fallback: alternate all-green / all-red
            all_g = 'G' * n
            all_r = 'r' * n
            half = n // 2
            self._green_states = [
                'G' * half + 'r' * (n - half),
                'r' * half + 'G' * (n - half),
                'G' * half + 'r' * (n - half),
                'r' * half + 'G' * (n - half),
            ]

        self.num_phases = 4
        self._all_red = 'r' * n

        # Reset state
        self.current_phase = 0
        self.is_yellow = False
        self.sim_step = 0
        self.episode += 1
        self._phase_duration = 0

        # Apply initial green phase
        traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self._green_states[0]
        )

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one RL step.

        1. If action differs from current phase, apply yellow first.
        2. Advance simulation by step_duration seconds.
        3. Compute state, reward, done.

        Args:
            action: Phase index (0 to num_phases-1).

        Returns:
            (next_state, reward, done, info)
        """
        action = int(np.clip(action, 0, self.num_phases - 1))
        info = {}

        # Handle phase switching with yellow buffer
        if action != self.current_phase:
            self._apply_yellow()
            self._simulate(self.yellow_duration)
            self.current_phase = action
            self._phase_duration = 0

        # Apply the target green phase
        self._apply_green(action)
        self._simulate(self.step_duration)
        self._phase_duration += self.step_duration

        # Get state, emissions, and reward
        state = self._get_state()
        emissions = self._get_emissions()
        reward = self._compute_reward(emissions)
        done = self.sim_step >= self.max_steps

        # Info dict
        info['sim_step'] = self.sim_step
        info['num_vehicles'] = traci.vehicle.getIDCount()
        info['phase'] = self.current_phase
        info['reward'] = reward
        info['emissions'] = emissions

        return state, reward, done, info

    def _simulate(self, duration: int):
        """Advance the SUMO simulation by `duration` seconds."""
        for _ in range(duration):
            traci.simulationStep()
            self.sim_step += 1

    def _apply_yellow(self):
        """Set yellow phase for the current green phase."""
        try:
            green_state = self._green_states[self.current_phase]
            yellow_state = green_state.replace('G', 'y').replace('g', 'y')
            traci.trafficlight.setRedYellowGreenState(
                self.tls_id, yellow_state
            )
        except Exception:
            pass

    def _apply_green(self, action: int):
        """Set the traffic light to the requested green phase."""
        try:
            action = min(action, len(self._green_states) - 1)
            traci.trafficlight.setRedYellowGreenState(
                self.tls_id, self._green_states[action]
            )
        except Exception:
            pass

    def _get_state(self) -> np.ndarray:
        """
        Extract state vector for each incoming lane.

        Per lane: [phase, queue_length, wait_time, emv_presence,
                   emv_position, emv_speed, neighbor_info]
        """
        state = []

        # Current TLS state string
        try:
            tls_state = traci.trafficlight.getRedYellowGreenState(
                self.tls_id
            )
        except Exception:
            tls_state = ""

        # Build lane->connection index mapping from controlled lanes list
        # getControlledLanes returns one lane per connection, in order
        try:
            ctrl_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        except Exception:
            ctrl_lanes = []

        for lane_idx, lane_id in enumerate(self.incoming_lanes):
            # 1. Phase: 1 if any connection for this lane is green
            phase = 0.0
            for ci, cl in enumerate(ctrl_lanes):
                if cl == lane_id and ci < len(tls_state):
                    if tls_state[ci].lower() == 'g':
                        phase = 1.0
                        break

            # 2. Queue length (metres)
            try:
                # Last step halting = number of stopped vehicles
                halting = traci.lane.getLastStepHaltingNumber(lane_id)
                queue_length = halting * VEHICLE_LENGTH
            except Exception:
                queue_length = 0.0

            # 3. Total waiting time
            try:
                wait_time = traci.lane.getWaitingTime(lane_id)
            except Exception:
                wait_time = 0.0

            # 4-6. EMV features
            emv_presence = 0.0
            emv_position = DETECTION_RANGE  # max distance if no EMV
            emv_speed = 0.0

            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for vid in veh_ids:
                    vtype = traci.vehicle.getTypeID(vid)
                    if vtype in ("ambulance", "fire_truck", "police"):
                        emv_presence = 1.0
                        # Distance from intersection ≈ lane length - position
                        lane_len = traci.lane.getLength(lane_id)
                        veh_pos = traci.vehicle.getLanePosition(vid)
                        dist = max(lane_len - veh_pos, 0.0)
                        if dist < emv_position:
                            emv_position = dist
                            emv_speed = traci.vehicle.getSpeed(vid)
            except Exception:
                pass

            # 7. Neighbor info (placeholder for single intersection)
            neighbor_info = 0.0

            state.extend([
                phase,
                queue_length / DETECTION_RANGE,   # normalize
                wait_time / 100.0,                 # normalize
                emv_presence,
                emv_position / DETECTION_RANGE,    # normalize
                emv_speed / 20.0,                  # normalize (~72 km/h max)
                neighbor_info,
            ])

        return np.array(state, dtype=np.float32)

    def _get_emissions(self) -> Dict:
        """
        Get current emission metrics from all incoming lanes.

        Returns:
            Dict with CO2, NOx, fuel consumption, and PM emissions.
        """
        total_co2 = 0.0
        total_nox = 0.0
        total_fuel = 0.0
        total_pmx = 0.0

        for lane_id in self.incoming_lanes:
            try:
                total_co2 += traci.lane.getCO2Emission(lane_id)
                total_nox += traci.lane.getNOxEmission(lane_id)
                total_fuel += traci.lane.getFuelConsumption(lane_id)
                total_pmx += traci.lane.getPMxEmission(lane_id)
            except Exception:
                pass

        return {
            'co2_mg_per_s': total_co2,
            'nox_mg_per_s': total_nox,
            'fuel_ml_per_s': total_fuel,
            'pmx_mg_per_s': total_pmx,
        }

    def _compute_reward(self, emissions: Optional[Dict] = None) -> float:
        """Compute the DNLight dynamic reward (optionally with green penalty)."""
        lane_wait_times = []
        emv_vehicles = []
        social_vehicles = []

        total_vehicles = 0
        total_emv_count = 0
        total_time_loss = 0.0

        # Gather per-lane data
        for lane_id in self.incoming_lanes:
            try:
                wt = traci.lane.getWaitingTime(lane_id)
                lane_wait_times.append(wt)
            except Exception:
                lane_wait_times.append(0.0)

        # Gather per-vehicle data
        all_veh_ids = traci.vehicle.getIDList()
        total_vehicles = len(all_veh_ids)

        for vid in all_veh_ids:
            try:
                vtype = traci.vehicle.getTypeID(vid)
                wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                speed = traci.vehicle.getSpeed(vid)
                time_loss = traci.vehicle.getTimeLoss(vid)

                if vtype in ("ambulance", "fire_truck", "police"):
                    total_emv_count += 1
                    total_time_loss += time_loss
                    try:
                        depart = traci.vehicle.getDeparture(vid)
                        travel_time = self.sim_step - depart if depart >= 0 else 0
                    except Exception:
                        travel_time = 0

                    emv_vehicles.append({
                        'travel_time': travel_time,
                        'wait_time': wait,
                        'avg_speed': max(speed, 0.1),
                        'time_loss': time_loss,
                        'n_emv': 0,
                        'n_total': 0,
                        'total_time_loss': 0,
                    })
                else:
                    social_vehicles.append({
                        'wait_time': wait,
                        'speed': max(speed, 0.01),
                        'time_loss': time_loss,
                    })
            except Exception:
                continue

        # Set global counts in EMV dicts
        for v in emv_vehicles:
            v['n_emv'] = total_emv_count
            v['n_total'] = max(total_vehicles, 1)
            v['total_time_loss'] = total_time_loss

        # Lane-level aggregate data for social reward
        lane_data = self._compute_lane_data()
        lane_wt_array = np.array(lane_wait_times, dtype=np.float32)

        # Choose reward function
        if self.use_green_reward and emissions is not None:
            reward = compute_green_reward(
                lane_wt_array, emv_vehicles, social_vehicles,
                lane_data, emissions, alpha_co2=self.alpha_co2
            )
        else:
            reward = compute_total_reward(
                lane_wt_array, emv_vehicles, social_vehicles, lane_data
            )

        return float(reward)

    def _compute_lane_data(self) -> Dict:
        """Compute aggregate lane statistics for social reward."""
        total_queue = 0.0
        total_capacity = 0.0
        total_flow = 0.0
        speeds = []

        for lane_id in self.incoming_lanes:
            try:
                halting = traci.lane.getLastStepHaltingNumber(lane_id)
                total_queue += halting
                lane_len = traci.lane.getLength(lane_id)
                total_capacity += lane_len / VEHICLE_LENGTH

                # Flow: vehicles that passed in last step
                total_flow += traci.lane.getLastStepVehicleNumber(lane_id)

                # Mean speed
                spd = traci.lane.getLastStepMeanSpeed(lane_id)
                if spd >= 0:
                    speeds.append(spd)
            except Exception:
                continue

        queue_ratio = total_queue / max(total_capacity, 1.0)
        flow_rate = max(total_flow, 0.01)
        speed_variance = float(np.var(speeds)) if speeds else 0.0
        throughput = total_flow  # vehicles per step

        return {
            'queue_ratio': queue_ratio,
            'flow_rate': flow_rate,
            'speed_variance': speed_variance,
            'throughput': max(throughput, 0.01),
        }

    def close(self):
        """Shut down the SUMO simulation."""
        try:
            traci.close()
        except Exception:
            pass

    def get_state_dim(self) -> int:
        """Return the flattened state dimension."""
        return self.state_dim or (len(self.incoming_lanes) * self.features_per_lane)

    def get_action_dim(self) -> int:
        """Return the number of actions."""
        return self.num_phases


class MultiIntersectionEnv:
    """
    Multi-intersection environment for DNLight + Green AI.

    Manages a 2x2 grid of signalized intersections. Each intersection
    has its own state/action space, and neighbor information is shared
    via the state vector's neighbor_info field.
    """

    def __init__(self,
                 sumocfg_path: str,
                 use_gui: bool = False,
                 step_duration: int = STEP_DURATION,
                 yellow_duration: int = YELLOW_DURATION,
                 max_steps: int = SIM_LENGTH,
                 label: str = "multi",
                 use_green_reward: bool = True,
                 alpha_co2: float = 0.3):
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.use_gui = use_gui
        self.step_duration = step_duration
        self.yellow_duration = yellow_duration
        self.max_steps = max_steps
        self.label = label
        self.use_green_reward = use_green_reward
        self.alpha_co2 = alpha_co2
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"

        self.tls_ids = []  # populated on reset
        self.tls_data = {}  # per-TLS: lanes, phases, state

        self.features_per_lane = 7
        self.sim_step = 0
        self.episode = 0
        self.state_dim = None
        self.num_phases = 4

    @property
    def action_dim(self) -> int:
        return self.num_phases

    @property
    def num_intersections(self) -> int:
        return len(self.tls_ids)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset simulation. Returns dict of {tls_id: state_array}.
        """
        self.seed = seed
        try:
            traci.close()
        except Exception:
            pass

        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_path,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--start", "true",
        ]
        if self.seed is not None:
            sumo_cmd.extend(["--seed", str(self.seed)])
        else:
            sumo_cmd.append("--random")

        traci.start(sumo_cmd, label=self.label)

        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.tls_data = {}

        for tls_id in self.tls_ids:
            lanes = sorted(set(
                traci.trafficlight.getControlledLanes(tls_id)
            ))
            logic = traci.trafficlight.getAllProgramLogics(tls_id)
            state_len = len(logic[0].phases[0].state) if logic else 16

            # Build 4 green phases (same logic as single intersection)
            if state_len == 16:
                green_states = [
                    'GGrgrrrrGGrgrrrr',
                    'rrGgrrrrrrGgrrrr',
                    'rrrrGGrgrrrrGGrg',
                    'rrrrrrGgrrrrrrGg',
                ]
            else:
                half = state_len // 2
                green_states = [
                    'G' * half + 'r' * (state_len - half),
                    'r' * half + 'G' * (state_len - half),
                    'G' * half + 'r' * (state_len - half),
                    'r' * half + 'G' * (state_len - half),
                ]

            self.tls_data[tls_id] = {
                'lanes': lanes,
                'state_len': state_len,
                'green_states': green_states,
                'current_phase': 0,
            }

            # Apply initial green
            traci.trafficlight.setRedYellowGreenState(
                tls_id, green_states[0]
            )

        # Set state dim from first TLS
        if self.tls_ids:
            first = self.tls_data[self.tls_ids[0]]
            self.state_dim = len(first['lanes']) * self.features_per_lane

        self.sim_step = 0
        self.episode += 1

        return self._get_all_states()

    def step(self, actions: Dict[str, int]
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, float],
                        bool, Dict]:
        """
        Step all intersections simultaneously.

        Args:
            actions: {tls_id: action_index}

        Returns:
            (states_dict, rewards_dict, done, info)
        """
        # Apply yellow and green for each intersection
        for tls_id, action in actions.items():
            data = self.tls_data[tls_id]
            action = int(np.clip(action, 0, self.num_phases - 1))

            if action != data['current_phase']:
                # Yellow
                green = data['green_states'][data['current_phase']]
                yellow = green.replace('G', 'y').replace('g', 'y')
                traci.trafficlight.setRedYellowGreenState(tls_id, yellow)

        # Simulate yellow
        self._simulate(self.yellow_duration)

        # Apply green phases
        for tls_id, action in actions.items():
            data = self.tls_data[tls_id]
            action = int(np.clip(action, 0, self.num_phases - 1))
            traci.trafficlight.setRedYellowGreenState(
                tls_id, data['green_states'][action]
            )
            data['current_phase'] = action

        # Simulate green
        self._simulate(self.step_duration)

        # Collect results
        states = self._get_all_states()
        emissions = self._get_all_emissions()
        rewards = self._compute_all_rewards(emissions)
        done = self.sim_step >= self.max_steps

        info = {
            'sim_step': self.sim_step,
            'num_vehicles': traci.vehicle.getIDCount(),
            'emissions': emissions,
        }

        return states, rewards, done, info

    def _simulate(self, duration: int):
        for _ in range(duration):
            traci.simulationStep()
            self.sim_step += 1

    def _get_all_states(self) -> Dict[str, np.ndarray]:
        """Get state vectors for all intersections."""
        states = {}
        # Compute neighbor queue info for neighbor_info field
        neighbor_queues = {}
        for tls_id in self.tls_ids:
            total_q = 0.0
            for lane_id in self.tls_data[tls_id]['lanes']:
                try:
                    total_q += traci.lane.getLastStepHaltingNumber(lane_id)
                except Exception:
                    pass
            neighbor_queues[tls_id] = total_q

        for tls_id in self.tls_ids:
            data = self.tls_data[tls_id]
            state = []

            try:
                tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                ctrl_lanes = traci.trafficlight.getControlledLanes(tls_id)
            except Exception:
                tls_state = ""
                ctrl_lanes = []

            # Average neighbor queue (excluding self)
            other_queues = [v for k, v in neighbor_queues.items() if k != tls_id]
            avg_neighbor_q = np.mean(other_queues) if other_queues else 0.0

            for lane_id in data['lanes']:
                # Phase
                phase = 0.0
                for ci, cl in enumerate(ctrl_lanes):
                    if cl == lane_id and ci < len(tls_state):
                        if tls_state[ci].lower() == 'g':
                            phase = 1.0
                            break

                # Queue
                try:
                    halting = traci.lane.getLastStepHaltingNumber(lane_id)
                    queue_length = halting * VEHICLE_LENGTH
                except Exception:
                    queue_length = 0.0

                # Wait time
                try:
                    wait_time = traci.lane.getWaitingTime(lane_id)
                except Exception:
                    wait_time = 0.0

                # EMV features
                emv_presence = 0.0
                emv_position = DETECTION_RANGE
                emv_speed = 0.0
                try:
                    for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                        vtype = traci.vehicle.getTypeID(vid)
                        if vtype in ("ambulance", "fire_truck", "police"):
                            emv_presence = 1.0
                            lane_len = traci.lane.getLength(lane_id)
                            veh_pos = traci.vehicle.getLanePosition(vid)
                            dist = max(lane_len - veh_pos, 0.0)
                            if dist < emv_position:
                                emv_position = dist
                                emv_speed = traci.vehicle.getSpeed(vid)
                except Exception:
                    pass

                # Neighbor info: normalized avg queue from neighbors
                neighbor_info = avg_neighbor_q / 20.0

                state.extend([
                    phase,
                    queue_length / DETECTION_RANGE,
                    wait_time / 100.0,
                    emv_presence,
                    emv_position / DETECTION_RANGE,
                    emv_speed / 20.0,
                    neighbor_info,
                ])

            states[tls_id] = np.array(state, dtype=np.float32)

        return states

    def _get_all_emissions(self) -> Dict[str, Dict]:
        """Get emission data for each intersection."""
        emissions = {}
        for tls_id in self.tls_ids:
            co2 = nox = fuel = pmx = 0.0
            for lane_id in self.tls_data[tls_id]['lanes']:
                try:
                    co2 += traci.lane.getCO2Emission(lane_id)
                    nox += traci.lane.getNOxEmission(lane_id)
                    fuel += traci.lane.getFuelConsumption(lane_id)
                    pmx += traci.lane.getPMxEmission(lane_id)
                except Exception:
                    pass
            emissions[tls_id] = {
                'co2_mg_per_s': co2,
                'nox_mg_per_s': nox,
                'fuel_ml_per_s': fuel,
                'pmx_mg_per_s': pmx,
            }
        return emissions

    def _compute_all_rewards(self, emissions: Dict) -> Dict[str, float]:
        """Compute reward for each intersection."""
        rewards = {}
        for tls_id in self.tls_ids:
            data = self.tls_data[tls_id]
            lane_wait_times = []
            emv_vehicles = []
            social_vehicles = []
            total_vehicles = 0
            total_emv_count = 0
            total_time_loss = 0.0

            for lane_id in data['lanes']:
                try:
                    lane_wait_times.append(
                        traci.lane.getWaitingTime(lane_id)
                    )
                except Exception:
                    lane_wait_times.append(0.0)

            all_veh_ids = traci.vehicle.getIDList()
            total_vehicles = len(all_veh_ids)

            for vid in all_veh_ids:
                try:
                    # Only count vehicles on this TLS's lanes
                    veh_lane = traci.vehicle.getLaneID(vid)
                    if veh_lane not in data['lanes']:
                        continue

                    vtype = traci.vehicle.getTypeID(vid)
                    wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    time_loss = traci.vehicle.getTimeLoss(vid)

                    if vtype in ("ambulance", "fire_truck", "police"):
                        total_emv_count += 1
                        total_time_loss += time_loss
                        try:
                            depart = traci.vehicle.getDeparture(vid)
                            travel_time = (
                                self.sim_step - depart if depart >= 0 else 0
                            )
                        except Exception:
                            travel_time = 0
                        emv_vehicles.append({
                            'travel_time': travel_time,
                            'wait_time': wait,
                            'avg_speed': max(speed, 0.1),
                            'time_loss': time_loss,
                            'n_emv': 0, 'n_total': 0,
                            'total_time_loss': 0,
                        })
                    else:
                        social_vehicles.append({
                            'wait_time': wait,
                            'speed': max(speed, 0.01),
                            'time_loss': time_loss,
                        })
                except Exception:
                    continue

            for v in emv_vehicles:
                v['n_emv'] = total_emv_count
                v['n_total'] = max(total_vehicles, 1)
                v['total_time_loss'] = total_time_loss

            lane_data = self._compute_lane_data_for(data['lanes'])
            lane_wt_array = np.array(lane_wait_times, dtype=np.float32)

            if self.use_green_reward:
                reward = compute_green_reward(
                    lane_wt_array, emv_vehicles, social_vehicles,
                    lane_data, emissions.get(tls_id, {}),
                    alpha_co2=self.alpha_co2
                )
            else:
                reward = compute_total_reward(
                    lane_wt_array, emv_vehicles, social_vehicles, lane_data
                )

            rewards[tls_id] = float(reward)

        return rewards

    def _compute_lane_data_for(self, lanes: List[str]) -> Dict:
        """Compute aggregate lane stats for specific lanes."""
        total_queue = 0.0
        total_capacity = 0.0
        total_flow = 0.0
        speeds = []

        for lane_id in lanes:
            try:
                halting = traci.lane.getLastStepHaltingNumber(lane_id)
                total_queue += halting
                total_capacity += traci.lane.getLength(lane_id) / VEHICLE_LENGTH
                total_flow += traci.lane.getLastStepVehicleNumber(lane_id)
                spd = traci.lane.getLastStepMeanSpeed(lane_id)
                if spd >= 0:
                    speeds.append(spd)
            except Exception:
                continue

        return {
            'queue_ratio': total_queue / max(total_capacity, 1.0),
            'flow_rate': max(total_flow, 0.01),
            'speed_variance': float(np.var(speeds)) if speeds else 0.0,
            'throughput': max(total_flow, 0.01),
        }

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    def get_state_dim(self) -> int:
        return self.state_dim or 56

    def get_action_dim(self) -> int:
        return self.num_phases
