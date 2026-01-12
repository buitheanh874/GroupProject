# RL Agent & Dueling DQN: Technical Analysis

> All claims cite exact file:line ranges from the codebase.

**Updated:** 2026-01-11

---

## A) RL Agent & Semi-MDP Solution

### What the Agent Observes, Chooses, and Learns From

| Component | Description | Code Citation |
|-----------|-------------|---------------|
| **State** | 12-dim vector passed to `select_action` as `np.ndarray` | [agent.py L105-131](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L105) |
| **Action** | Discrete integer $a \in \{0, ..., 14\}$ (15 actions) | [agent.py L102-103](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L102) |
| **Reward** | Scalar `float` stored with transition | [agent.py L133-142](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L133) |
| **Returns** | TD-target: $r + \gamma \cdot Q_{\text{target}}(s', a^*)$ | [agent.py L154](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L154) |

### Why This is a Semi-MDP

1. **Variable Cycle Lengths**: Actions correspond to (cycle, split) pairs where cycles are [60, 90, 120] seconds.
2. **Per-Transition Gamma**: The replay buffer stores a **per-transition gamma** (`batch.gammas`), not a fixed global gamma.

```python
# agent.py L133-142
def store_transition(self, state, action, reward, next_state, done, gamma=None):
    gamma_value = float(gamma) if gamma is not None else float(self.gamma)
    self.replay_buffer.push(..., gamma=float(gamma_value))

# agent.py L154 - TD target uses per-transition gamma
target_q = batch.rewards + batch.gammas * next_q_target * (1.0 - batch.dones)
```

---

## B) Network Architecture: DuelingDQN Class

### Architecture Overview

> Source: [dueling_dqn.py](file:///c:/Users/Dell/GroupProject2/rl/dueling_dqn.py)

**Input Dimension**: 12 (state_dim)
**Hidden Layers**: `[192, 192]` (from config)

### Layer Construction

```python
self.feature_net = nn.Sequential(
    nn.Linear(int(state_dim), hidden_1),  # 12 → 192
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),         # 192 → 192
    nn.ReLU(),
)

self.value_head = nn.Linear(hidden_2, 1)            # 192 → 1
self.advantage_head = nn.Linear(hidden_2, action_dim)  # 192 → 15
```

### Q-Value Computation (Dueling Architecture)

```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    features = self.feature_net(state)
    value = self.value_head(features)                    # V(s): shape [B, 1]
    advantage = self.advantage_head(features)            # A(s,a): shape [B, 15]
    advantage_mean = advantage.mean(dim=1, keepdim=True) # Mean-centering
    q_values = value + advantage - advantage_mean        # Q(s,a) = V(s) + A(s,a) - mean(A)
    return q_values
```

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

### ASCII Network Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DuelingDQN Network                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: state ∈ ℝ¹²                                                │
│   ┌─────────────────┐                                               │
│   │  Linear(12→192) │ ─ReLU→┐                                       │
│   └─────────────────┘       │                                       │
│                             ▼                                       │
│                    ┌─────────────────┐                              │
│                    │ Linear(192→192) │ ─ReLU→ features ∈ ℝ¹⁹²       │
│                    └─────────────────┘                              │
│                             │                                       │
│              ┌──────────────┴──────────────┐                        │
│              ▼                             ▼                        │
│   ┌────────────────────┐       ┌────────────────────┐               │
│   │  Value Head        │       │  Advantage Head    │               │
│   │  Linear(192→1)     │       │  Linear(192→15)    │               │
│   │  Output: V(s) ∈ ℝ¹ │       │  Output: A(s,·) ∈ℝ¹⁵│               │
│   └────────────────────┘       └────────────────────┘               │
│              │                             │                        │
│              └──────────┬──────────────────┘                        │
│                         ▼                                           │
│              ┌─────────────────────────┐                            │
│              │ Q(s,a) = V(s) + A(s,a) - Ā(s)                        │
│              │ Output: Q ∈ ℝ¹⁵         │                            │
│              └─────────────────────────┘                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## C) Time-Aware Gamma (`compute_gamma`)

### Code Location

> [agent.py L87-99](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L87)

```python
def compute_gamma(self, t_step: Optional[float]) -> float:
    if not self._use_time_aware_gamma or t_step is None:
        return float(self.gamma)
    t_step_value = float(t_step)
    if t_step_value <= 0.0:
        raise ValueError(f"t_step must be >0 when use_time_aware_gamma is enabled, got {t_step_value}")
    
    exponent = t_step_value / float(self._t_ref)
    result = float(self._gamma_base ** exponent)
    
    result = max(1e-9, min(result, 1.0))
    
    return result
```

### Formula

$$\gamma_t = \gamma_0^{\left( \frac{t_{\text{step}}}{t_{\text{ref}}} \right)}$$

### Why This Exists: Semi-MDP Discounting

**Problem**: With fixed γ=0.99 per step, different cycle lengths create uneven time discounting:

| Cycle | Steps in 1 hour | Effective $\gamma^{60}$ for 1 hour |
|-------|-----------------|-------------------------------------|
| 60s   | 60 steps        | $0.99^{60} = 0.547$ |
| 120s  | 30 steps        | $0.99^{30} = 0.740$ |

**Solution**: Time-aware discounting ensures **consistent discount per unit time**.

### Numeric Example

Using $\gamma_0 = 0.98$, $t_{\text{ref}} = 60$ seconds:

| Cycle Length | Exponent | Computed $\gamma_t$ |
|--------------|----------|---------------------|
| 60 sec       | $60/60 = 1.0$ | $0.98^{1.0} = 0.980$ |
| 90 sec       | $90/60 = 1.5$ | $0.98^{1.5} = 0.970$ |
| 120 sec      | $120/60 = 2.0$ | $0.98^{2.0} = 0.960$ |

**Invariance**: After 1 hour, both 60s and 120s cycles produce $\gamma_{\text{total}} = 0.98^{60} = 0.298$.

---

## D) Training Loop: Replay Buffer & Target Updates

### Replay Buffer

| Component | Location | Details |
|-----------|----------|---------|
| Buffer init | [agent.py L65-69](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L65) | Capacity=200,000, stores per-transition γ |
| Add experience | [agent.py L133-142](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L133) | `store_transition(s,a,r,s',done,gamma)` |
| Sample batch | [replay_buffer.py L55-76](file:///c:/Users/Dell/GroupProject2/rl/replay_buffer.py#L55) | Uniform random, no prioritization |

### Target Network Updates

**Hard update** every `target_update_freq` (3000) optimization steps:

```python
# agent.py L166-168
self.update_step_count += 1
if int(self.update_step_count) % int(self._config.target_update_freq) == 0:
    self.target_net.load_state_dict(self.online_net.state_dict())
```

### Optimization Step

```python
# agent.py L144-170
def update(self) -> Optional[float]:
    batch = self.replay_buffer.sample(batch_size=..., device=...)

    # Double DQN: online net selects actions, target net evaluates
    with torch.no_grad():
        next_q_online = self.online_net(batch.next_states)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
        next_q_target = self.target_net(batch.next_states).gather(1, next_actions)
        target_q = batch.rewards + batch.gammas * next_q_target * (1.0 - batch.dones)

    current_q = self.online_net(batch.states).gather(1, batch.actions)
    loss = self.loss_fn(current_q, target_q)  # MSE Loss
```

### Summary Table

| Component | Location | Details |
|-----------|----------|---------|
| Buffer init | [agent.py L65-69](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L65) | Capacity=200,000, stores per-transition γ |
| Loss function | [agent.py L63, L158](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L63) | `nn.MSELoss()` |
| Optimizer | [agent.py L62](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L62) | Adam, lr=0.0003 |
| Gradient clip | [agent.py L162-163](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L162) | `clip_grad_norm=10.0` |
| Target update | [agent.py L167-168](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L167) | Hard copy every 3000 steps |
