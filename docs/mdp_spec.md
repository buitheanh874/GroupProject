# MDP Specification (Report-aligned)

## Scope

Single intersection, 2-phase (NS/EW). Right-turn slip lanes are free-flow and excluded from MDP (not in state, not in reward).

## Time Scale

One decision step equals one signal cycle.

At cycle t:

1. The agent observes state s_t
2. The agent chooses action a_t (phase split)
3. The environment simulates one cycle under that split
4. The environment returns reward r_t and next state s_{t+1}

## Lane Grouping

Controlled lanes:

- LANES_NS_CTRL: incoming lanes (straight + left) on North–South axis controlled by TLS
- LANES_EW_CTRL: incoming lanes (straight + left) on East–West axis controlled by TLS

Slip lanes (right-turn free-flow):

- LANES_RT_NS: right-turn slip lanes for N/S approaches
- LANES_RT_EW: right-turn slip lanes for E/W approaches

Slip lanes are excluded from state and reward.

## State

Fixed-length vector:

state_raw = [q_NS, q_EW, w_NS, w_EW] with shape (4,)

Queue length:

- q_NS: number of halting vehicles on LANES_NS_CTRL (v < 0.1 m/s)
- q_EW: number of halting vehicles on LANES_EW_CTRL (v < 0.1 m/s)

Waiting time per cycle (vehicle-seconds):

- During each simulation second within the cycle, compute q_NS_step and q_EW_step using halting numbers
- w_NS = sum over the cycle of q_NS_step
- w_EW = sum over the cycle of q_EW_step

## State Normalization

Z-score for each component x:

x_norm = (x - mu_x) / (sigma_x + eps)

Then clip all components:

s_norm = clip(s_norm, -5, 5)

mu_x and sigma_x must be estimated from baseline fixed-time runs and stored in config.

## Action

Cycle-based set phase split with fixed cycle length C = 60s (default).

rho_NS + rho_EW = 1 and rho_NS, rho_EW >= rho_min

Default rho_min = 0.1.

Discrete action set (action_id 0..4):

- a0: (0.30, 0.70)
- a1: (0.40, 0.60)
- a2: (0.50, 0.50)
- a3: (0.60, 0.40)
- a4: (0.70, 0.30)

## Reward (Report-aligned)

Total waiting time within the cycle:

W_t = w_NS(t) + w_EW(t)

Reward per cycle:

r_t = - W_t

No reward difference term (do not use W_t - W_{t-1}). No fairness term in reward.

## Fairness

Fairness is future work and not part of the reward.

Anti-starvation is enforced by rho_min, ensuring each phase has a minimum green time each cycle.

## RL Default Hyperparameters

- gamma = 0.98
- learning_rate = 1e-3
- batch_size = 64
- replay_buffer_size = 100000
- target_update_freq = 1000
- eps_start = 1.0
- eps_end = 0.05
- eps_decay_steps = 50000
