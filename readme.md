Here is a comprehensive, highly structured technical specification document designed specifically for an AI coding agent to implement the DNLight algorithm from the provided paper. It extracts all formulas, architectures, and hyperparameters into a developer-ready format.

---

# System Specification: DNLight (Deep Reinforcement Learning-Driven Traffic Signal Control)

## 1. Environment & Tech Stack

* 
**Simulator**: SUMO (Simulation of Urban MObility).


* 
**Frameworks**: Python, TensorFlow.


* 
**Simulation Length**: 3600 seconds per episode.


* 
**Step Size**: 10 seconds, with a 3-second yellow light buffer during phase switching.


* 
**Data/Map**: RESCO benchmarks (Cologne and Ingolstadt intersections), modified to include 3 types of emergency vehicles (EMVs).



## 2. System Architecture

The system operates on a three-layer Fog Computing framework:

* 
**Bottom Layer**: Four roadside sensing units (one per incoming lane direction) that collect traffic data within a 200m range and act as micro-computers for feature extraction.


* 
**Middle Layer (Fog Node)**: Aggregates information from the 4 bottom nodes, calculates dynamic rewards, executes the DRL (DNLight) agent, and sends control commands back.


* 
**Top Layer (Cloud Server)**: Asynchronous data center that receives backed-up data from the fog nodes and trains/updates the global model for the fog layer.



---

## 3. Markov Decision Process (MDP) Formulation

### State Space ()

The detection range is strictly 200m from the intersection. For a given lane , the state vector consists of 7 elements :

1. 
**Phase ()**: Current signal state (1 for green, 0 for red).


2. 
**Queue Length ()**: Queue length in meters.


3. 
**Wait Time ()**: Total accumulated waiting time for all vehicles in lane  (seconds).


4. 
**EMV Presence ()**: 1 if EMV is present, 0 otherwise.


5. 
**EMV Position ()**: Distance between the nearest EMV and the intersection.


6. 
**EMV Speed ()**: Speed of the nearest EMV.


7. 
**Neighbor Info ()**: Observation data relayed from the other 3 unit units at the intersection.



### Action Space ()

Discrete action space choosing 1 of 4 possible signal phases :

* 
: North-South straight.


* 
: North-South left turn.


* 
: East-West straight.


* 
: East-West left turn.



### Reward Function Formulation ()

The reward function dynamically calculates weights based on real-time traffic conditions to balance EMVs and social vehicles. Note: Final reward is the *negative* sum of these penalties, so lower values indicate better states.

**Penalty Coefficient ()**
Used to balance deviations in waiting times across different lanes:


Where  is the standard deviation of waiting times across lanes, clamping  to the range [2, 3.5].

**Waiting Time Deviation Penalty ()**


Where  is the waiting time for lane , and  is the average waiting time across all lanes.

**Reward for Emergency Vehicles ()**
Calculated per vehicle  using log-normalized travel time (), wait time (), average speed (), and time loss () :


Dynamic Weightings for EMVs:

* 
*   *(where  is the ratio of EMVs to total vehicles)*
* 
*  *(where  is the total time loss)*

**Reward for Social Vehicles ()**
Calculated per vehicle :


Dynamic Weightings for Social Vehicles :

* 
* 
* 
* 

**Total Intersection Reward ()**


Where global weights are calculated as :

* 
* 

---

## 4. Agent Architecture: DNLight

The agent utilizes a Double Dueling DQN architecture coupled with Prioritized Experience Replay (PER) and Parameterized Noisy Networks (NoisyNet).

### A. Network Structure

* 
**Shared Feature Layer**: Input states map to a fully connected layer (dimension 512), combined with parameterized noise. Dropout is set to 0.2. Uses SiLU activation.


* **Dueling Heads**:
* 
**Value Stream **: Extracts global environmental benefit characteristics.


* 
**Advantage Stream **: Focuses on relative value differences of specific actions.




* 
**Q-Value Aggregation**:





### B. Noisy Linear Layer (NoisyNet)

Used for structured exploration instead of standard -greedy. The parameters of the linear layers are decoupled into deterministic and random noise components:


* 
**Initialization**: .  initialized via Kaiming uniform. .


* 
**Dynamic Attenuation**: Global noise intensity decays linearly: . Locally scaled by a random uniform distribution of 50%-150% per batch.


* 
**Hybrid Exploration Rule**:





### C. Prioritized Experience Replay (PER)

Samples are prioritized based on TD error.

* 
**TD Error**: .


* 
**Priority Level**: .


* 
**Importance Sampling Weight**:  (where  linearly anneals from 0.5 to 1).



### D. Multi-Agent Communication (Attention Mechanism)

Fog nodes (units) exchange local states to construct collaborative policies.

* 
**Neighbor State Message** for Unit :




* 
**Attention Weight ()** between unit  and :




* 
**Aggregated Signal**:





---

## 5. Hyperparameter Specifications

Directly derived from the simulation parameters:

| Parameter | Value | Explanation |
| --- | --- | --- |
| **Detection Range** | 200m | Range for roadside units |
| **Batch Size ()** | 512 | Training batch size |
| **Learning Rate ()** | 0.99 | Discount factor (Note: Paper labels  as learning rate in table, but refers to it as discount factor in text) |
| **EPS_START** | 1.0 | Initial exploration rate |
| **EPS_END** | 0.02 | Final exploration rate |
| **EPS_Decay** | 50,000 | Decay steps for epsilon |
| **Target Update** | 1500 | Frequency to sync Target network with Online network |
| **Noisy_std** | 0.1 | Initial standard deviation for NoisyNet |
| **Gradient Clip ()** | 5.0 | Trimming coefficient for gradients |
| **Replay Buffer Size** | 500,000 | Capacity of experience pool |
| **C (Penalty Coeff Init)** | 2 | Initial penalty multiplier |
| **Vehicle Length** | 4.5m | Average car length calculation |
| **Communication Delay** | 5ms | Network latency between fog units |
| **Dropout** | 0.2 | Abandonment rate for networks |
| **Velocity Disp. Weight** | 0.15 | Alpha variable for velocity |

---