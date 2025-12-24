# \# MDP Addendum — Ultimate Scenario (Clean \& Realism) and Fairness Ablation

# 

# \*\*Purpose:\*\* This document extends the baseline MDP specification (see `docs/mdp\_spec.md`) with:

# 1\. A dynamic demand scenario design ("Ultimate") for testing adaptation

# 2\. An optional fairness penalty term for ablation study

# 3\. Evaluation protocol for comparing Pure vs Fair agents

# 

# \*\*Status:\*\* Code-aligned as of December 2024. All parameter values match implementation in `env/sumo\_env.py` and config files.

# 

# ---

# 

# \## 1. Baseline MDP (Unchanged)

# 

# The core MDP definition remains as specified in `docs/mdp\_spec.md`:

# 

# \- \*\*Decision step:\*\* One control cycle (agent observes state, selects action, environment simulates full cycle, returns reward and next state)

# \- \*\*State (4D):\*\* `s\_t = \[q\_NS, q\_EW, w\_NS, w\_EW]`

# &nbsp; - `q\_NS`, `q\_EW`: Queue length (number of halting vehicles) on controlled lanes

# &nbsp; - `w\_NS`, `w\_EW`: Aggregated waiting time (vehicle-seconds) over the cycle

# \- \*\*Action (Discrete, 5 choices):\*\* Select green split ratio (ρ\_NS, ρ\_EW) from:

# &nbsp; - (0.30, 0.70), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.70, 0.30)

# \- \*\*Baseline reward (Pure):\*\* `r\_t = -W\_t / 3600.0`, where `W\_t = w\_NS(t) + w\_EW(t)`

# &nbsp; - \*\*Note:\*\* Division by 3600.0 scales reward to vehicle-hours for numerical stability

# 

# ---

# 

# \## 2. Timing Clarification (Clean \& Realism - Option B)

# 

# \### 2.1 Cycle Structure

# 

# The baseline MDP defines a \*\*green budget per cycle\*\* as `C = 60` seconds. The action allocates this 60s between NS and EW phases.

# 

# To improve realism without adding complexity, the simulator also includes \*\*fixed transition time\*\*:

# 

# \- \*\*Yellow time:\*\* 2 seconds per phase switch (NS→EW and EW→NS)

# \- \*\*All-red time:\*\* 0 seconds (not used in current config)

# 

# \*\*Implementation note:\*\* These transition times are configured in `phase\_program` (see `configs/\_base\_train.yaml`):

# ```yaml

# phase\_program:

# &nbsp; ns\_green: 0

# &nbsp; ew\_green: 4

# &nbsp; ns\_yellow: 1

# &nbsp; ew\_yellow: 5

# &nbsp; all\_red: null

# ```

# 

# \### 2.2 Wall-Clock Cycle Time

# 

# Therefore, the \*\*total simulation time per decision step\*\* is:

# 

# ```

# T\_cycle = green\_budget + yellow\_NS + yellow\_EW

# &nbsp;       = 60s + 2s + 2s

# &nbsp;       = 64 seconds

# ```

# 

# \*\*Key point:\*\* This 64s is the actual elapsed simulation time. The action still only controls the \*\*60s green allocation\*\*. Yellow times are \*\*not controllable\*\* by the agent.

# 

# ---

# 

# \## 3. "Ultimate" Dynamic Demand Scenario (Scenario Design)

# 

# \### 3.1 Motivation

# 

# Instead of training separate agents on static low/high traffic, we train a \*\*single "Ultimate" agent\*\* on a long episode where demand changes over time. This tests the agent's \*\*adaptation capability\*\*.

# 

# \### 3.2 Scenario Parameters

# 

# \- \*\*Route file:\*\* `networks/BI\_ultimate\_clean.rou.xml` (must be created separately)

# \- \*\*Total duration:\*\* 7680 seconds (2.133 hours)

# \- \*\*Structure:\*\* 4 stages × 30 cycles per stage

# \- \*\*Stage boundaries:\*\*

# &nbsp; - Stage 1: 0–1920s (30 cycles @ 64s/cycle)

# &nbsp; - Stage 2: 1920–3840s

# &nbsp; - Stage 3: 3840–5760s

# &nbsp; - Stage 4: 5760–7680s

# 

# \### 3.3 Config Settings

# 

# Training configs using Ultimate scenario:

# \- `configs/train\_ultimate\_pure.yaml` (λ = 0.0)

# \- `configs/train\_ultimate\_fair.yaml` (λ = 0.12)

# 

# Key settings:

# ```yaml

# env:

# &nbsp; sumo:

# &nbsp;   route\_file: networks/BI\_ultimate\_clean.rou.xml

# &nbsp;   max\_cycles: 120

# &nbsp;   max\_sim\_seconds: 7680

# &nbsp;   terminate\_on\_empty: false

# ```

# 

# \*\*Important:\*\* `terminate\_on\_empty: false` ensures the episode runs for full 7680s regardless of instantaneous vehicle count.

# 

# \### 3.4 Demand Design (Route File Specification)

# 

# The `BI\_ultimate\_clean.rou.xml` file should implement a \*\*time-varying flow rate\*\*:

# 

# Example structure (to be created by M3/M4):

# ```xml

# <!-- Stage 1: Low demand (0-1920s) -->

# <flow id="f\_N2S\_stage1" begin="0" end="1920" vehsPerHour="50" .../>

# 

# <!-- Stage 2: Medium demand (1920-3840s) -->

# <flow id="f\_N2S\_stage2" begin="1920" end="3840" vehsPerHour="100" .../>

# 

# <!-- Stage 3: High demand (3840-5760s) -->

# <flow id="f\_N2S\_stage3" begin="3840" end="5760" vehsPerHour="150" .../>

# 

# <!-- Stage 4: Medium demand (5760-7680s) -->

# <flow id="f\_N2S\_stage4" begin="5760" end="7680" vehsPerHour="100" .../>

# ```

# 

# ---

# 

# \## 4. Final Agents (Deliverables)

# 

# We train \*\*two agents\*\* to compare the impact of fairness penalty:

# 

# \### 4.1 Ultimate-Pure (Baseline, MDP-aligned)

# 

# \*\*Config:\*\* `configs/train\_ultimate\_pure.yaml`

# 

# \*\*Reward function:\*\*

# ```

# r\_t = -W\_t / 3600.0

# ```

# where `W\_t = w\_NS(t) + w\_EW(t)` (total waiting time in vehicle-seconds)

# 

# \*\*Lambda fairness:\*\* `λ = 0.0`

# 

# \*\*Interpretation:\*\* Optimizes average delay aggressively. Under unbalanced demand, may allow one direction to experience higher waiting time if it improves overall efficiency.

# 

# \### 4.2 Ultimate-Fair (Ablation/Extension)

# 

# \*\*Config:\*\* `configs/train\_ultimate\_fair.yaml`

# 

# \*\*Fairness penalty term:\*\*

# 

# Instead of introducing a separate state variable, we compute the fairness term `F\_t` from existing aggregated signals:

# 

# ```

# F\_t = max(w\_NS, w\_EW)

# ```

# 

# This penalizes the \*\*worst-case waiting time\*\* among the two directions.

# 

# \*\*Full reward function:\*\*

# ```

# r\_fair = -(W\_t + λ·F\_t) / 3600.0

# &nbsp;    = -(w\_NS + w\_EW + λ·max(w\_NS, w\_EW)) / 3600.0

# ```

# 

# \*\*Lambda fairness:\*\* `λ = 0.12` (selected empirically)

# 

# \*\*Code reference:\*\* `env/sumo\_env.py`, lines 391-398:

# ```python

# lambda\_fairness = float(self.\_config.lambda\_fairness)

# total\_wait = float(w\_ns + w\_ew)

# 

# if lambda\_fairness > 0.0:

# &nbsp;   max\_wait = max(float(w\_ns), float(w\_ew))

# &nbsp;   reward = -(total\_wait + lambda\_fairness \* max\_wait) / 3600.0

# else:

# &nbsp;   reward = -total\_wait / 3600.0

# ```

# 

# \*\*Interpretation:\*\* Trades a small amount of mean delay to reduce tail risk (max waiting time). Under unbalanced demand, should provide more balanced service to both directions.

# 

# \### 4.3 Reward Comparability Note

# 

# \*\*Important:\*\* Because λ differs between Pure and Fair agents, their \*\*raw reward values are not directly comparable\*\*. During evaluation:

# 

# \- Use \*\*physically meaningful KPIs\*\* (AvgWait, MaxWait, AvgQueue, Arrived) as primary metrics

# \- If comparing rewards, evaluate \*\*both\*\* agents under \*\*both\*\* reward formulations (see Section 5.3)

# 

# ---

# 

# \## 5. Evaluation Protocol (Two Battlefields + Cross-Eval)

# 

# \### 5.1 Battlefield A — Standard (In-Distribution)

# 

# \*\*Goal:\*\* Measure core performance and stability on the training scenario.

# 

# \*\*Setup:\*\*

# \- Scenario: Same Ultimate dynamic demand used for training

# \- Route file: `networks/BI\_ultimate\_clean.rou.xml`

# \- Runs: 10+ with different seeds

# \- Controllers: Fixed, Ultimate-Pure, Ultimate-Fair

# 

# \*\*Config example:\*\* Create `configs/eval\_ultimate.yaml` based on `train\_ultimate\_pure.yaml` with evaluation settings.

# 

# \### 5.2 Battlefield B — Robustness (Out-of-Distribution)

# 

# \*\*Goal:\*\* Test generalization to unseen demand patterns.

# 

# \*\*Setup:\*\*

# \- Scenario: \*\*Asymmetric/unbalanced demand\*\* (e.g., heavy NS, light EW)

# \- Route file: `networks/BI\_unbalanced.rou.xml` (to be created)

# \- Runs: 10+ with different seeds

# \- Controllers: Fixed, Ultimate-Pure, Ultimate-Fair

# 

# \*\*Config:\*\* `configs/eval\_unbalanced.yaml` (already exists, references `BI\_unbalanced.rou.xml`)

# 

# \*\*Example asymmetric design:\*\*

# ```xml

# <!-- Heavy NS traffic -->

# <flow id="f\_N2S" vehsPerHour="200" .../>

# <flow id="f\_S2N" vehsPerHour="200" .../>

# 

# <!-- Light EW traffic -->

# <flow id="f\_E2W" vehsPerHour="50" .../>

# <flow id="f\_W2E" vehsPerHour="50" .../>

# ```

# 

# \### 5.3 Cross-Eval (Reward Comparability)

# 

# To fairly compare agents trained under different reward functions, evaluate \*\*each trained model\*\* under \*\*both\*\* reward settings:

# 

# \*\*Procedure:\*\*

# 

# 1\. Load Ultimate-Pure agent

# &nbsp;  - Evaluate with λ=0.0 (Pure reward) → record KPIs + reward\_pure

# &nbsp;  - Evaluate with λ=0.12 (Fair reward) → record KPIs + reward\_fair

# 

# 2\. Load Ultimate-Fair agent

# &nbsp;  - Evaluate with λ=0.0 (Pure reward) → record KPIs + reward\_pure

# &nbsp;  - Evaluate with λ=0.12 (Fair reward) → record KPIs + reward\_fair

# 

# \*\*Implementation note:\*\* Modify `scripts/eval.py` to accept `--eval-lambda` argument that overrides the lambda value in the environment config during evaluation only (does not affect the trained policy, only the reward calculation for logging).

# 

# \*\*Rationale:\*\* This separates \*\*learned policy quality\*\* from \*\*reward definition\*\*. If Fair agent achieves better MaxWait even under Pure reward evaluation, it proves the fairness benefit is real, not an artifact of reward scaling.

# 

# ---

# 

# \## 6. Primary Metrics for Reporting

# 

# \*\*Use these physically meaningful KPIs\*\* as the main comparison criteria:

# 

# \### 6.1 Efficiency Metrics

# \- \*\*AvgWait:\*\* Mean waiting time per arrived vehicle (seconds)

# \- \*\*AvgQueue:\*\* Mean queue length across all timesteps (vehicles)

# \- \*\*Arrived:\*\* Total vehicles that completed their routes (throughput)

# 

# \### 6.2 Fairness / Robustness Metrics

# \- \*\*MaxWait:\*\* Worst-case waiting time of any vehicle in the episode (seconds)

# \- \*\*P95Wait:\*\* 95th percentile waiting time (seconds)

# 

# \*\*Code reference:\*\* These are tracked by `env.kpi.EpisodeKpiTracker` and returned in `info\["episode\_kpi"]` at episode end.

# 

# \### 6.3 Reward as Training Signal

# 

# Reward values should be reported for \*\*training diagnostics\*\* (convergence, stability) but \*\*not\*\* as the primary comparison between Pure and Fair unless cross-eval is performed.

# 

# ---

# 

# \## 7. Config File Mapping

# 

# \### 7.1 Training Configs

# 

# | Config File | Lambda | Scenario | Purpose |

# |-------------|--------|----------|---------|

# | `train\_ultimate\_pure.yaml` | 0.0 | Ultimate | Pure agent (baseline) |

# | `train\_ultimate\_fair.yaml` | 0.12 | Ultimate | Fair agent (ablation) |

# 

# \### 7.2 Evaluation Configs

# 

# | Config File | Scenario | Purpose |

# |-------------|----------|---------|

# | `eval\_ultimate.yaml` (TBD) | Ultimate | In-distribution test |

# | `eval\_unbalanced.yaml` | Unbalanced | Out-of-distribution test |

# | `eval\_low.yaml` | Low traffic | Legacy static scenario |

# | `eval\_high.yaml` | High traffic | Legacy static scenario |

# 

# \*\*TODO (M3/M4):\*\*

# \- Create `networks/BI\_ultimate\_clean.rou.xml` with 4-stage time-varying demand

# \- Create `networks/BI\_unbalanced.rou.xml` with asymmetric NS/EW flows

# \- Create `configs/eval\_ultimate.yaml` for Battlefield A

# \- Verify `configs/eval\_unbalanced.yaml` references correct route file

# 

# ---

# 

# \## 8. Normalization Stats

# 

# \*\*Important:\*\* The Ultimate scenario uses \*\*different normalization stats\*\* than static low/high scenarios.

# 

# \*\*Config reference:\*\* `configs/norm\_stats\_ultimate\_clean.json`

# 

# This file should be generated by running:

# ```bash

# python scripts/collect\_norm\_stats.py \\

# &nbsp; --config configs/train\_ultimate\_pure.yaml \\

# &nbsp; --episodes 5 \\

# &nbsp; --seed 0 \\

# &nbsp; --out configs/norm\_stats\_ultimate\_clean.json

# ```

# 

# \*\*Both\*\* Ultimate-Pure and Ultimate-Fair training configs reference this same normalization file:

# ```yaml

# normalization:

# &nbsp; mean: \[0.0, 0.0, 0.0, 0.0]

# &nbsp; std: \[1.0, 1.0, 1.0, 1.0]

# &nbsp; file: configs/norm\_stats\_ultimate\_clean.json

# ```

# 

# ---

# 

# \## 9. Implementation Checklist

# 

# \### 9.1 Route Files (M3/M4 responsibility)

# \- \[ ] Create `networks/BI\_ultimate\_clean.rou.xml` with 4-stage demand profile

# \- \[ ] Verify `networks/BI\_unbalanced.rou.xml` exists and has asymmetric flows

# \- \[ ] Test route files run in SUMO without errors

# 

# \### 9.2 Config Files (M1 responsibility)

# \- \[x] `train\_ultimate\_pure.yaml` — exists, correct lambda=0.0

# \- \[x] `train\_ultimate\_fair.yaml` — exists, correct lambda=0.12

# \- \[x] `eval\_unbalanced.yaml` — exists

# \- \[ ] `eval\_ultimate.yaml` — needs to be created

# 

# \### 9.3 Normalization Stats (M1 + M3/M4)

# \- \[ ] Generate `configs/norm\_stats\_ultimate\_clean.json` using baseline fixed controller on Ultimate scenario

# \- \[ ] Verify mean/std values are reasonable (not all zeros)

# 

# \### 9.4 Evaluation Scripts (M2 responsibility)

# \- \[x] `scripts/eval.py` — works with existing configs

# \- \[ ] Add `--eval-lambda` argument for cross-eval (optional but recommended)

# \- \[ ] Add `scripts/plot\_eval.py` support for MaxWait and P95Wait metrics

# 

# \### 9.5 Code Verification (M1)

# \- \[x] `env/sumo\_env.py` — reward formula matches doc (lines 391-398)

# \- \[x] `env/sumo\_env.py` — timing calculation matches doc (64s cycle)

# \- \[x] `env/kpi.py` — tracks MaxWait and P95Wait correctly

# \- \[x] All configs have consistent `phase\_program` with 2s yellow

# 

# ---

# 

# \## 10. Appendix: Parameter Summary Table

# 

# | Parameter | Value | Location | Notes |

# |-----------|-------|----------|-------|

# | Green budget (C) | 60s | `env.sumo.green\_cycle\_sec` | Controlled by action |

# | Yellow time | 2s | `env.sumo.yellow\_sec` | Fixed, not controlled |

# | All-red time | 0s | `env.sumo.all\_red\_sec` | Not used |

# | Total cycle time | 64s | Derived | 60 + 2 + 2 |

# | Lambda (Pure) | 0.0 | `train\_ultimate\_pure.yaml` | No fairness penalty |

# | Lambda (Fair) | 0.12 | `train\_ultimate\_fair.yaml` | Empirically tuned |

# | Rho\_min | 0.1 | `env.sumo.rho\_min` | Min 6s green per phase |

# | Episode duration (Ultimate) | 7680s | `env.sumo.max\_sim\_seconds` | ~2.13 hours |

# | Max cycles (Ultimate) | 120 | `env.sumo.max\_cycles` | Backup limit |

# | Gamma (discount) | 0.98 | `agent.gamma` | Unchanged from baseline |

# 

# ---

# 

# \## 11. Relationship to Baseline MDP

# 

# This addendum \*\*extends\*\* but does not replace `docs/mdp\_spec.md`:

# 

# \- \*\*State, Action, Normalization:\*\* Unchanged

# \- \*\*Reward (Pure variant):\*\* Unchanged except for /3600.0 scaling

# \- \*\*Reward (Fair variant):\*\* New optional extension

# \- \*\*Scenario design:\*\* New "Ultimate" replaces static low/high for adaptation testing

# \- \*\*Evaluation protocol:\*\* Extended with cross-eval and robustness battlefield

# 

# All other aspects (lane grouping, slip lane exclusion, phase constraints, etc.) remain as specified in the baseline MDP.

# 

# ---

# 

# \## 12. References

# 

# \- Baseline MDP: `docs/mdp\_spec.md`

# \- Implementation: `env/sumo\_env.py`, `env/kpi.py`

# \- Training configs: `configs/train\_ultimate\_\*.yaml`

# \- Evaluation configs: `configs/eval\_\*.yaml`

# \- Route files: `networks/BI\_\*.rou.xml`

# 

# \*\*Last updated:\*\* December 2024 (code-aligned)

