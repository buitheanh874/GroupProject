# MDP Addendum: Ultimate Scenario (Clean & Realism, Option B) and Fairness Ablation

This addendum extends the baseline MDP specification without changing its core definitions. It introduces (1) a dynamic demand episode design (“Ultimate”) to test adaptation, and (2) an optional fairness penalty as an ablation variant.

## 1. Baseline MDP (Unchanged)

The baseline MDP remains as defined in the original spec:

- **Decision step:** one control cycle. At the beginning of cycle *t*, the agent observes state *sₜ*, selects action *aₜ*, the simulator advances one full cycle, then returns reward *rₜ* and next state *sₜ₊₁*.
- **State (4D):**  
  **sₜ = [q_NS, q_EW, w_NS, w_EW]**  
  - **q_NS, q_EW:** queue length (halting vehicles) on the controlled lane groups  
  - **w_NS, w_EW:** aggregated waiting time (vehicle-seconds) accumulated over the cycle on the controlled lane groups  
  Right-turn slip lanes (if present) are excluded from controlled lane groups.
- **Action (Discrete, 5 choices):** choose a green split ratio (ρ_NS, ρ_EW) from:  
  (0.30, 0.70), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.70, 0.30)
- **Baseline reward (Pure):**  
  Let **Wₜ = w_NS + w_EW**. The implemented baseline reward is:  
  **rₜ = -Wₜ / 3600.0**  
  Division by **3600.0** rescales from vehicle-seconds to vehicle-hours for numerical stability.

## 2. Timing Clarification (Option B: Clean & Realism)

### 2.1 Green budget vs wall-clock cycle time

- **Green budget per decision step:** 60 seconds (**controlled by the action split**).
- **Fixed transition time:** yellow is enabled and fixed (not controlled by the action).
  - **yellow_sec = 2** seconds for each phase switch (NS→EW and EW→NS)
  - **all_red_sec = 0**

Therefore the **wall-clock simulation time per decision step** is:

- **T_cycle = 60 + 2 + 2 = 64 seconds**

The action still only allocates the **60s green budget** between NS and EW.

### 2.2 Phase program mapping (configuration)

The environment uses indices to identify phases. Example mapping in configs:

- **ns_green:** 0  
- **ns_yellow:** 1  
- **ew_green:** 4  
- **ew_yellow:** 5  
- **all_red:** 0 (unused when all_red_sec = 0)

## 3. Ultimate Dynamic Demand Scenario (Scenario Design)

Instead of training separate models for low/high traffic, the “Ultimate” scenario trains a single agent on one long episode where demand changes over time. This tests **adaptation** under non-stationary demand.

### 3.1 Episode structure

- **Decision steps per stage:** 30 cycles
- **Stage duration:** 30 × 64 = **1920 seconds**
- **Total duration:** 4 stages × 1920 = **7680 seconds**
- **Stage boundaries:**  
  Stage 1: 0–1920  
  Stage 2: 1920–3840  
  Stage 3: 3840–5760  
  Stage 4: 5760–7680  

### 3.2 Route file reference

- **Training route file:** `networks/BI_ultimate_clean.rou.xml`  
  This file must exist in the repository and defines the 4-stage demand pattern above.

## 4. Final Agents (Deliverables)

### 4.1 Ultimate-Pure (Baseline)

- **Fairness weight:** λ = 0.0
- **Reward:**  
  **rₜ = -(w_NS + w_EW) / 3600.0**

Interpretation: optimizes mean delay aggressively; may increase worst-case waiting time under asymmetric demand.

### 4.2 Ultimate-Fair (Ablation Variant)

This variant adds a fairness penalty based on the *largest* approach-level waiting accumulation.

- Define:  
  **Wₜ = w_NS + w_EW**  
  **Mₜ = max(w_NS, w_EW)**
- **Fairness weight:** λ = 0.12
- **Reward:**  
  **rₜ = -(Wₜ + λ · Mₜ) / 3600.0**

Reporting note: because λ differs, **raw reward values are not directly comparable** between Pure and Fair unless evaluated under the same reward configuration.

## 5. Normalization (Mandatory for Ultimate)

State normalization is computed from rollouts on the same scenario configuration:

- Output file: `configs/norm_stats_ultimate_clean.json`
- Collect command:
```bash
python scripts/collect_norm_stats.py --config configs/train_ultimate_pure.yaml --episodes 5 --seed 0 --out configs/norm_stats_ultimate_clean.json
```

If the environment loader requires the normalization file to exist before collection, create a temporary placeholder by copying an existing norm stats file and then rerun the collection command to overwrite it.

## 6. Evaluation Protocol

Primary conclusions are based on KPIs (AvgWait, MaxWait, AvgQueue, Arrived). Reward is treated as a training signal.

### 6.1 Battlefield A (In-Distribution)

Evaluate both models on the same Ultimate scenario used for training (dynamic 4-stage route).

### 6.2 Battlefield B (Out-of-Distribution Robustness)

Evaluate on an asymmetric demand route that was not used for training (separate `.rou.xml` file). The trained checkpoints remain unchanged.

### 6.3 Cross-eval (Comparable reward scale)

To compare reward on the same scale, evaluate each checkpoint under both configurations:

- Evaluate both checkpoints with **λ = 0** (Pure config)
- Evaluate both checkpoints with **λ = 0.12** (Fair config)

This can be done without code changes by selecting the desired config and providing `--model_path`.

## 7. Key Parameters (Summary)

| Item | Value |
|---|---:|
| Green budget per decision step | 60 s |
| Yellow per switch | 2 s |
| All-red | 0 s |
| Wall-clock cycle time | 64 s |
| Cycles per stage | 30 |
| Stage duration | 1920 s |
| Total episode duration | 7680 s |
| max_cycles | 120 |
| λ (Pure) | 0.0 |
| λ (Fair) | 0.12 |

## 8. References (Repository Paths)

- Baseline MDP spec: `docs/mdp_spec.md`
- Environment implementation: `env/sumo_env.py`
- Training configs:  
  `configs/train_ultimate_pure.yaml`  
  `configs/train_ultimate_fair.yaml`
- Training route: `networks/BI_ultimate_clean.rou.xml`
- Normalization stats: `configs/norm_stats_ultimate_clean.json`
