# WHAT WE BUILT: Project Contributions Report
## Điều Khiển Đèn Giao Thông Bằng Học Tăng Cường

---

## 1. Tổng Quan Công Sức

| Metric | Con số |
|--------|--------|
| **Tổng dòng code Python** | ~22,000 LOC |
| **Modules tự viết** | 5 (env, rl, controllers, scripts, tests) |
| **Files Python** | 139 files |
| **Unit tests** | ~3,000 LOC |

---

## 2. Những Gì Chúng Tôi TỰ XÂY DỰNG

### 2.1 Environment Engineering (env/) - 2,284 LOC

**Hoàn toàn tự viết:**

- **SumoEnv** (`sumo_env.py`): SUMO environment wrapper
  - State extraction 14 chiều (queue, waiting time, occupancy)
  - Action execution với cycle/split validation
  - Episode management và reset logic
  
- **MDP Design** (`mdp_metrics.py`): 
  - Reward function: `simple_clipped` với normalization
  - KPI tracking: wait time, queue, throughput
  
- **State Normalization** (`state_normalizer.py`):
  - Z-score normalization từ collected statistics
  - Per-feature mean/std loading

- **Action Space** (`action_utils.py`):
  - Discrete action definitions (cycle × split)
  - Green time calculation với min/max constraints

### 2.2 RL Algorithm (rl/) - 1,975 LOC

**Hoàn toàn tự implement:**

- **Dueling Double DQN** (`dueling_dqn.py`):
  - Value stream + Advantage stream architecture
  - Target network soft update
  
- **DQN Agent** (`agent.py`):
  - Epsilon-greedy exploration với decay schedule
  - Huber loss training
  - Gradient clipping
  
- **Replay Buffer** (`replay_buffer.py`):
  - Uniform sampling
  - Batch operations
  
- **Parallel Training** (`parallel_collector_optimized.py`):
  - Multi-process actor-learner architecture
  - Queue-based transition collection
  - Network synchronization

### 2.3 Baseline Controllers (controllers/) - 1,102 LOC

**Hoàn toàn tự implement:**

- **FixedTimeController**: Fixed split selection
- **ActuatedController**: Gap-out logic với min/max green
- **WebsterController**: Webster formula-based split calculation
- **MaxPressureController**: Queue difference optimization
- **RandomController**: Random action baseline

### 2.4 Training & Evaluation Pipeline (scripts/) - 13,474 LOC

**Hoàn toàn tự viết:**

- **Training Scripts**:
  - `train.py`: Sequential training
  - `train_parallel_optimized.py`: 12-worker parallel training
  
- **Evaluation Scripts**:
  - `eval.py`: Multi-policy, multi-scenario evaluation
  - CLI override cho từng controller
  
- **Route Generation**:
  - `generate_train_routes_final.py`: Training routes với imbalance patterns
  - `generate_eval_routes.py`: Evaluation routes (unseen, mixed-demand)
  
- **Utilities**:
  - Config management (YAML loading, override)
  - Logging và checkpointing
  - Result aggregation

### 2.5 Testing Suite (tests/) - 3,017 LOC

- Unit tests cho mọi component
- Integration tests cho training pipeline
- Validation tests cho state/reward

---

## 3. Tools/Libraries Đã Sử Dụng (KHÔNG tự viết)

| Tool | Mục đích | Tự viết code wrapper? |
|------|----------|----------------------|
| **SUMO** | Traffic simulation | ✅ Có (SumoEnv) |
| **TraCI** | SUMO control API | ✅ Có (wrapper methods) |
| **PyTorch** | Neural network | ✅ Có (Dueling DQN) |
| **NumPy** | Array operations | Dùng trực tiếp |
| **YAML** | Config parsing | Dùng trực tiếp |

---

## 4. Thiết Kế & Quyết Định Kỹ Thuật

### 4.1 MDP Formulation (Tự thiết kế)

| Component | Quyết định |
|-----------|------------|
| **State** | 14D: queue (4) + wait (4) + occ (4) + global (2) |
| **Action** | 3 discrete: splits 0.3/0.5/0.7, cycle 60s |
| **Reward** | Clipped waiting time normalized |

### 4.2 Curriculum Learning (Tự thiết kế)

4 phases với imbalance patterns:
1. Easy (350 veh/hr) - 48%
2. Medium (500 veh/hr) - 24%
3. Hard (650 veh/hr) - 8%
4. Mix - 20%

### 4.3 Parallel Training Architecture (Tự thiết kế)

- 12 actor workers
- Centralized learner
- Queue-based async communication
- Epsilon diversity multipliers

### 4.4 Fair Evaluation Framework (Tự thiết kế)

- 4 evaluation scenarios
- 5 policies (random, fixed, actuated, webster, rl)
- 3 imbalance types per scenario
- CLI override cho riêng từng controller

---

## 5. Không Sử Dụng

❌ **Stable-Baselines3** hoặc RL libraries có sẵn
❌ **OpenAI Gym wrappers** có sẵn cho SUMO
❌ **Pre-trained models**
❌ **Copy-paste code từ tutorials**

---

## 6. Kết Luận

| Aspect | Assessment |
|--------|------------|
| **Tooling vs Coding** | **70% Coding, 30% Tooling** |
| **Originality** | Hoàn toàn tự thiết kế MDP, algorithm, pipeline |
| **Engineering Effort** | Significant (~22K LOC) |
| **Research Contribution** | Domain-specific adaptations cho Vietnam traffic |

**Đây là một dự án nghiên cứu ứng dụng với công sức kỹ thuật đáng kể, không phải chỉ sử dụng tools có sẵn.**

---

*Generated: 2026-01-24*
