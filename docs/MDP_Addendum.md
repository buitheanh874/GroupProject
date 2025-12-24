# MDP Addendum — Ultimate Scenario (Clean & Realism) and Fairness Ablation

This document is an addendum to the original MDP specification. The baseline MDP (state/action/reward definition) remains unchanged. We only (i) redesign the demand scenario to be dynamic over time (“Ultimate”), and (ii) introduce an optional fairness penalty as an ablation study.

## 1. Baseline MDP (Unchanged)

We retain the original cycle-based decision process:

- **Decision step:** at the start of each control cycle, the agent observes state \(s_t\), selects an action \(a_t\), the simulator advances for one full cycle, then the environment returns \(r_t\) and \(s_{t+1}\).
- **State (4D):** \(s_t = [q_{NS}, q_{EW}, w_{NS}, w_{EW}]\), where \(q\) denotes queue length and \(w\) denotes aggregated waiting time over the **controlled lane groups** (right-turn slip lanes, if present, are excluded).
- **Action (Discrete, 5 choices):** action \(a_t\) selects a green split ratio for NS vs EW from:
  - (0.30, 0.70), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.70, 0.30)
- **Baseline reward (Pure):** \(r_t = -W_t\), where \(W_t = w_{NS}(t) + w_{EW}(t)\).

## 2. Timing Clarification for “Clean & Realism” (Option B)

The baseline MDP defines the **green budget per cycle** as \(C = 60\) seconds (the action allocates this 60s between NS and EW).

To improve realism, the simulator also includes fixed transition time:

- **Yellow:** 2 seconds per phase switch (NS→EW and EW→NS)
- **All-red:** 0 seconds

Therefore, the **wall-clock simulation time per decision step** is:

- \(T_{cycle} = 60 + 2 + 2 = 64\) seconds

This timing is fixed and **not controlled by the action**. The action still only controls the allocation inside the 60s green budget.

## 3. “Ultimate” Dynamic Demand Scenario (Scenario Design)

Instead of training separate agents on static low/high traffic, we train a single “Ultimate” agent on a long episode where demand changes over time.

- **Total duration:** 7680 seconds
- **Stages:** 4 stages × 30 cycles per stage
- **Stage length:** 30 × 64 = 1920 seconds
- **Stage boundaries (seconds):** 0–1920, 1920–3840, 3840–5760, 5760–7680

This design tests **adaptation**: the agent must reallocate green time as demand shifts during a single episode.

## 4. Final Agents (Deliverables)

### 4.1 Ultimate-Pure (Baseline, MDP-aligned)

- **Reward:** \(r_t = -W_t\)
- **Fairness weight:** \(\lambda = 0\)
- **Interpretation:** optimizes average delay aggressively; may increase worst-case waiting time under unbalanced demand.

### 4.2 Ultimate-Fair (Ablation/Extension)

We introduce an optional fairness penalty term \(F_t\) computed from the same aggregated lane-group signals (no new sensors/state variables are required).

- **Reward:** \(r^{fair}_t = -W_t - \lambda \cdot F_t\)
- **Fairness weight:** \(\lambda = 0.12\) (selected empirically)
- **Interpretation:** trades a small amount of mean delay to reduce starvation and improve robustness (typically reducing max waiting time).

Important reporting note: because \(\lambda\) differs, **raw reward values are not directly comparable between Pure and Fair** unless evaluated under the same reward configuration.

## 5. Evaluation Protocol (Two Battlefields + Cross-Eval)

### 5.1 Battlefield A — Standard (In-Distribution)

Evaluate both models on the same dynamic “Ultimate” scenario used for training to measure core performance and stability.

### 5.2 Battlefield B — Robustness (Out-of-Distribution)

Evaluate on a different **asymmetric demand** scenario that the agent did not see during training to measure generalization.

### 5.3 Cross-Eval (Reward Comparability)

To compare reward fairly, evaluate both trained models under **both** reward settings:

- Both models under \(\lambda=0\) (Pure reward function)
- Both models under \(\lambda=0.12\) (Fair reward function)

This isolates whether performance differences come from the learned policy or the reward definition.

## 6. Primary Metrics for Reporting

Report and compare agents using physically meaningful KPIs:

- **AvgWait** (mean waiting time)
- **MaxWait** (worst-case waiting time / tail risk)
- **AvgQueue** (mean queue length)
- **Arrived** (throughput)

Use reward primarily as a training signal; use KPIs for final conclusions.
