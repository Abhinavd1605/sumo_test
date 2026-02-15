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

from dnlight.reward import compute_total_reward

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
                 label: str = "default"):
        """
        Args:
            sumocfg_path: Path to .sumocfg file.
            use_gui: If True, use sumo-gui instead of sumo.
            step_duration: RL step duration in simulation seconds.
            yellow_duration: Yellow phase duration in seconds.
            max_steps: Maximum simulation time in seconds.
            label: TraCI connection label.
        """
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.use_gui = use_gui
        self.step_duration = step_duration
        self.yellow_duration = yellow_duration
        self.max_steps = max_steps
        self.label = label

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

    def reset(self) -> np.ndarray:
        """Reset the simulation and return the initial state."""
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

        # Get state and reward
        state = self._get_state()
        reward = self._compute_reward()
        done = self.sim_step >= self.max_steps

        # Info dict
        info['sim_step'] = self.sim_step
        info['num_vehicles'] = traci.vehicle.getIDCount()
        info['phase'] = self.current_phase
        info['reward'] = reward

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

    def _compute_reward(self) -> float:
        """Compute the DNLight dynamic reward."""
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
                    # Approximate travel time from departure
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
                        'n_emv': 0,       # will be set below
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
