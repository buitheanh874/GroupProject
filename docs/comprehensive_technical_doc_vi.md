# TÀI LIỆU KỸ THUẬT TOÀN DIỆN
## Điều Khiển Đèn Giao Thông Bằng Học Tăng Cường Đa Tác Tử

**Final Design**: Kiến trúc tối giản với cycle cố định 60s, 3 hành động rời rạc

---

# PHẦN 1: ĐỊNH NGHĨA BÀI TOÁN VÀ MÔ TẢ TÌNH HUỐNG

## 1.1 Tổng Quan Bài Toán

### Vấn Đề Cần Giải Quyết

Bài toán điều khiển đèn giao thông tại mạng lưới 9 giao lộ đô thị với các đặc điểm:

- **Mô hình mạng lưới**: Mạng lưới giả lập dạng lưới 3x3 (Synthetic 3x3 Grid - BIGNET)
- **Loại giao thông**: Hỗn hợp (xe máy 86%, ô tô 12%, xe buýt 2%)
- **Mục tiêu**: Tối thiểu hóa thời gian chờ đợi toàn mạng lưới
- **Ràng buộc**: An toàn (min green, yellow, all-red), chu kỳ cố định 60s

### Định Nghĩa Hình Thức

**Bài toán tối ưu hóa**:
$$
\min_{\pi} \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t \cdot W_t \right]
$$

Trong đó:
- $\pi$: Policy điều khiển đèn (shared DQN)
- $W_t$: Tổng thời gian chờ tại bước quyết định $t$
- $\gamma$: Hệ số chiết khấu (0.99)
- $T$: Horizon mô phỏng (1800s = 30 decision steps)

---

## 1.2 Mô Tả Tình Huống Cụ Thể

### Mạng Lưới Giao Lộ

```
        [E38]         [E39]         [E40]
          │             │             │
    ──────┼─────────────┼─────────────┼────── 
          │             │             │
        [J7] ───────── [J3] ───────── [J17]
          │             │             │
    ──────┼─────────────┼─────────────┼────── 
          │             │             │
[E33]──── J1 ───────── [J0] ───────── [J2] ──── [E34]
          │      (CENTER)│             │
    ──────┼─────────────┼─────────────┼────── 
          │             │             │
        [J6] ───────── [J4] ───────── [J14]
          │             │             │
    ──────┼─────────────┼─────────────┼────── 
          │             │             │
        [E37]         [E36]         [E35]
```

### Đặc Điểm Giao Lộ

| Thuộc tính | Giá trị |
|------------|---------|
| **Số giao lộ** | 9 (J0, J1, J2, J3, J4, J6, J7, J14, J17) |
| **Giao lộ trung tâm** | J0 (center_tls_id) |
| **Số làn mỗi hướng** | 3 làn (NS và EW) |
| **Biên mạng lưới** | 12 cạnh (E33-E40) |
| **Downstream links (J0)** | N:E3, E:E0, S:E2, W:E1 |

### Các Mức Nhu Cầu Giao Thông (Curriculum)

| Mức | Demand (xe/giờ/làn) | Episodes per Worker | Mục đích |
|-----|---------------------|---------------------|----------|
| **Easy** | 350 | 120 (48%) | Warm-up, stable learning |
| **Medium** | 500 | 60 (24%) | Main training phase |
| **Hard** | 650 | 20 (8%) | Challenge, prevent overfitting |
| **Mix** | 350/500/650 mixed | 50 (20%) | Generalization across all levels |

**Total**: 250 episodes/worker × 12 workers = **3000 episodes**

---

## 1.3 Mô Hình MDP (Markov Decision Process)

### Không Gian Trạng Thái (State Space)

**Kích thước**: 14 chiều cho mỗi TLS (KHÔNG sử dụng downstream occupancy)

```
s = [q_N, q_E, q_S, q_W, w_N, w_E, w_S, w_W, 0, 0, 0, 0, n_norm, φ_norm]
     ─────────────────  ─────────────────  ─────────────  ─────────────────
     Hàng đợi (0-3)     Chờ đợi (4-7)      [DISABLED]     Global (12-13)
```

| Nhóm | Dims | Mô tả | Đơn vị |
|------|------|-------|--------|
| **Queue** | 0-3 | Số xe xếp hàng theo 4 hướng (N/E/S/W) | xe (PCU-weighted) |
| **Wait** | 4-7 | Thời gian chờ tích lũy theo hướng | xe-giây |
| **Occupancy** | 8-11 | **DISABLED** (set to 0 for fairness) | - |
| **Global** | 12-13 | Scalar toàn mạng (normalized) | [0,1] |

### Không Gian Hành Động (Action Space)

**Kích thước**: **3 hành động rời rạc** (Global Green Time = 60s)

| Action ID | ρ_NS | ρ_EW | g_NS (s) | g_EW (s) | Mô tả |
|-----------|------|------|----------|----------|-------|
| **0** | 0.30 | 0.70 | 18.0 | 42.0 | Favor East-West |
| **1** | 0.50 | 0.50 | 30.0 | 30.0 | Balanced |
| **2** | 0.70 | 0.30 | 42.0 | 18.0 | Favor North-South |

**Tính toán thời gian xanh**:
$$
g_{NS} = C_{green} \cdot \rho_{NS}, \quad g_{EW} = C_{green} \cdot \rho_{EW}
$$

Trong đó:
- $C_{green} = 60s$ (Tổng thời gian xanh - config `green_cycle_sec`)
- $L = 10s$ (Transitions: 2 × (3s yellow + 2s all-red))
- **Tổng chu kỳ thực tế**: $C_{total} = 60 + 10 = 70s$

### Hàm Phần Thưởng (Reward Function)

**Simple Normalized Reward** (theo cấu hình hiện tại):

$$
R = -\frac{W_{total}}{N \cdot t_{ref}}
$$

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| $W_{total}$ | - | Tổng thời gian chờ (veh-sec) - **Không trọng số PCU** |
| $N$ | n_present | Số xe hiện tại trong mạng lưới |
| $t_{ref}$ | 60.0 | Hằng số chuẩn hóa (Normalizing Constant) |

**Cơ chế**:
- **PCU Disabled**: Weight = 1.0 cho mọi loại xe (Motor/Car/Bus).
- **Global Scope**: Reward tính trên toàn mạng lưới, chia sẻ cho tất cả Agent.
- **No Clipping**: Reward không bị cắt cứng về [-1, 0] để giữ tín hiệu gradient tốt hơn (tuy nhiên giá trị thường tự nhiên nằm trong khoảng này).
- **Linear**: Không bình phương thời gian chờ (`reward_exponent = 1.0`).

---

# PHẦN 2: THIẾT KẾ HỆ THỐNG VÀ THUẬT TOÁN

## 2.1 Workflow Tổng Quan

```
┌──────────────────────────────────────────────────────────────────┐
│                   PARALLEL TRAINING SYSTEM                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────┐      ┌────────────────────┐              │
│  │  12 Actor Workers  │═════▶│  Learner Process   │              │
│  │  (Collect exp)     │◀═════│  (Train DQN)       │              │
│  └────────────────────┘      └────────────────────┘              │
│         │                              │                          │
│         ▼                              ▼                          │
│  ┌────────────────────┐      ┌────────────────────┐              │
│  │  SUMO (ports       │      │  Shared Replay     │              │
│  │  9500-9511)        │      │  Buffer (200k)     │              │
│  └────────────────────┘      └────────────────────┘              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## 2.2 Kiến Trúc MARL (Multi-Agent RL)

### Parameter Sharing Architecture

```
                    ┌─────────────────────────────────┐
                    │     SHARED DUELING DQN          │
                    │        [128, 128]               │
                    │                                 │
                    │  Input: 14D state               │
                    │  Output: 3 Q-values             │
                    └─────────────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
            ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
            │ Agent   │        │ Agent   │        │ Agent   │
            │   J0    │        │   J1    │   ...  │   J17   │
            └────┬────┘        └────┬────┘        └────┬────┘
                 │                  │                  │
                 └──────────────────┴──────────────────┘
                                    │
                            ┌───────▼───────┐
                            │ GLOBAL REWARD │
                            │   (Shared)    │
                            └───────────────┘
```

### Đặc Điểm Kiến Trúc

| Thành phần | Chi tiết |
|------------|----------|
| **Network** | Dueling DQN với 2 hidden layers **[128, 128]** |
| **Input** | 14D normalized state |
| **Output** | 3 Q-values (cho 3 actions) |
| **Value Head** | Linear(128 → 1) cho V(s) |
| **Advantage Head** | Linear(128 → 3) cho A(s,a) |
| **Q-value** | Q(s,a) = V(s) + A(s,a) - mean(A) |

### Chia Sẻ Phần Thưởng

- **9 agents** dùng chung 1 policy network (parameter sharing)
- **Global reward** được chia đều cho tất cả agents
- **Implicit coordination** qua dims 12-13 (global scalars)
- **Decentralized execution**: Mỗi agent quan sát local state và chọn action độc lập

## 2.3 Thuật Toán Dueling Double DQN

### Công Thức Cập Nhật

**Target Q-value** (Double DQN):
$$
y = R + \gamma \cdot Q_{target}(s', \arg\max_{a'} Q_{online}(s', a'))
$$

**Loss function** (Huber Loss):
$$
L = \begin{cases}
\frac{1}{2}(y - Q)^2 & \text{if } |y - Q| \leq 1 \\
|y - Q| - \frac{1}{2} & \text{otherwise}
\end{cases}
$$

### Time-Aware Gamma (DISABLED)

**Note**: `use_time_aware_gamma: false` trong final design

Fixed gamma = 0.99 cho tất cả transitions (decision cycle cố định 60s).

---

# PHẦN 3: THIẾT LẬP MÔI TRƯỜNG

## 3.1 Cấu Hình SUMO

### File Mạng Lưới

| File | Mô tả |
|------|-------|
| `networks/BIGNET.net.xml` | Network definition với 9 giao lộ 3x3 grid |

### Tham Số Mô Phỏng

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `step_length_sec` | 1.0 | Bước mô phỏng (s) |
| `max_sim_seconds` | 1800 | Episode duration (30 min) |
| `max_cycles` | 60 | 60 cycles × 60s = 3600s max |
| `green_cycle_sec` | **60** | **Fixed cycle length** |
| `yellow_sec` | 3 | Yellow time (s) |
| `all_red_sec` | 2 | All-red clearance (s) |
| `g_min_sec` | 10 | Minimum green time (s) |
| `rho_min` | 0.1 | Minimum split ratio |
| `time-to-teleport` | 300 | Teleport timeout (s) |


### Cấu Hình TLS

```yaml
tls_ids: ["J0", "J1", "J2", "J3", "J4", "J6", "J7", "J14", "J17"]
center_tls_id: "J0"  # For downstream occupancy (currently disabled)
downstream_links:
  N: "E3"
  E: "E0"
  S: "E2"
  W: "E1"
```

## 3.2 Route Files và Curriculum

### Cấu Trúc Thư Mục

```
networks/variants/train_final/
├── easy/                         # 350 veh/hr/lan, 3 imbalance types
│   ├── bignet_d350_ns_heavy_seed00001.rou.xml
│   ├── bignet_d350_ew_heavy_seed00001.rou.xml
│   ├── bignet_d350_balanced_seed00001.rou.xml
│   └── ... (~300 route files)
├── medium/                       # 500 veh/hr/lan
│   └── ... (~300 route files)
├── hard/                         # 650 veh/hr/lan
│   └── ... (~300 route files)
├── manifest_easy.txt             # List of easy routes
├── manifest_medium.txt           # List of medium routes
├── manifest_hard.txt             # List of hard routes
└── manifest_mix.txt              # Mixed: 60% easy, 30% med, 10% hard
```

### Đặc Điểm Route

| Thuộc tính | Giá trị |
|------------|---------|
| **Turn ratios** | 80% thẳng, 10% trái, 10% phải |
| |
| **Imbalance types** | NS-heavy, EW-heavy, Balanced |
| **Seeds** | 100 seeds per (demand, imbalance) combination |
| **Duration** | 1800 giây |

## 3.3 Normalization Statistics

**File**: `configs/norm_final_design.json`

Được thu thập từ:
- 4 parallel scripts (easy1, easy2, medium1, medium2)
- Random subset of routes
- Merged into single JSON file

Chuẩn hóa 14 dims về mean=0, std=1 để ổn định training.

---

# PHẦN 4: QUY TRÌNH HUẤN LUYỆN (TRAINING)

## 4.1 Parallel Training Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  PARALLEL TRAINING (12 WORKERS)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. INITIALIZATION (Learner Process)                     │   │
│   │    - Load config (train_final_design.yaml)              │   │
│   │    - Initialize Dueling DQN [128, 128]                  │   │
│   │    - Create Shared Replay Buffer (200,000)              │   │
│   │    - Load normalization (norm_final_design.json)        │   │
│   │    - Spawn 12 actor processes (ports 9500-9511)         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 2. CURRICULUM PHASE LOOP (per worker)                   │   │
│   │    Phase 1 (Easy):    120 episodes - 350 veh/hr/lan    │   │
│   │    Phase 2 (Medium):   60 episodes - 500 veh/hr/lan    │   │
│   │    Phase 3 (Hard):     20 episodes - 650 veh/hr/lan    │   │
│   │    Phase 4 (Mix):      50 episodes - random mix        │   │
│   │    ─────────────────────────────────────────────────    │   │
│   │    Total: 250 episodes/worker × 12 = 3000 episodes     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 3. ACTOR-LEARNER LOOP                                   │   │
│   │    ┌──────────────────────────────────────────────────┐ │   │
│   │    │ ACTORS (parallel):                               │ │   │
│   │    │  - Reset SUMO with route from manifest           │ │   │
│   │    │  - Collect transitions with ε-greedy             │ │   │
│   │    │  - Send batches (256 transitions) to learner     │ │   │
│   │    └──────────────────────────────────────────────────┘ │   │
│   │    ┌──────────────────────────────────────────────────┐ │   │
│   │    │ LEARNER (single):                                │ │   │
│   │    │  - Receive batches from queue                    │ │   │
│   │    │  - Train DQN (batch_size=256, train_freq=4)      │ │   │
│   │    │  - Update target network (every 5000 steps)      │ │   │
│   │    │  - Sync weights to actors (every 100 updates)    │ │   │
│   │    └──────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 4. LOGGING & CHECKPOINTING                              │   │
│   │    - Log metrics every 2.0 seconds                      │   │
│   │    - Save model checkpoints periodically                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 Hyperparameters

### Network Architecture

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| `hidden_dims` | **[128, 128]** | 2 hidden layers (reduced from 192) |
| `state_dim` | 14 | Input dimension |
| `action_dim` | **3** | Output dimension (3 discrete actions) |

### Learning Parameters

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| `gamma` | 0.99 | Discount factor |
| `learning_rate` | 0.0001 | Adam optimizer LR |
| `batch_size` | 256 | Mini-batch size |
| `replay_buffer_size` | 200,000 | Replay buffer capacity |
| `target_update_freq` | 5,000 | Steps between target sync |
| `learning_starts` | 2,000 | Steps before learning |
| `train_freq` | 4 | Steps between updates |
| `clip_grad_norm` | 10.0 | Gradient clipping |
| `use_huber_loss` | true | Huber loss for stability |

### Exploration (Epsilon-Greedy)

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| `eps_start` | 0.60 | Initial epsilon |
| `eps_end` | 0.05 | Final epsilon |
| `warmup_global_steps` | 5,000 | Warmup before decay |
| `eps_decay_steps` | 40,000 | Decay duration |

**Total training steps**: ~81,000 (3000 eps × 27 steps/ep)
- Epsilon reaches 0.05 at step ~45,000 (55% training)
- Safe margin for convergence

**Worker-specific epsilon multipliers**:
```python
[0.92, 0.94, 0.96, 0.98, 0.99, 1.00, 1.01, 1.02, 1.04, 1.06, 1.08, 1.10]
```
- Tighter range [0.92, 1.10] for consistent convergence
- Adds diversity without destabilizing

### Parallel Training

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| `num_actors` | **12** | Number of parallel workers |
| `base_port` | 9500 | Starting SUMO port |
| `chunk_size` | 256 | Transitions per batch |
| `queue_max_chunks` | 200 | Queue backpressure |
| `sync_every_updates` | 100 | Network sync frequency |

## 4.3 Performance Optimizations

**All optimizations are SAFE** (do not change MDP/algorithm):

| Feature | Enabled | Mô tả |
|---------|---------|-------|
| `disable_sumo_logs` | true | Suppress file I/O |
| `use_packed_transitions` | true | Numpy serialization |
| `use_batch_replay_push` | true | Batched buffer inserts |
| `interval_logging_sec` | 2.0 | Reduce logging overhead |
| `worker0_verbose_only` | true | Only rank=0 logs verbose |
| `queue_maxsize` | 1000 | Backpressure control |

## 4.4 Lệnh Chạy Training

```bash
# Training với 12 workers
python scripts/train_parallel_optimized.py --config configs/train_final_design.yaml

# Resume từ checkpoint
python scripts/train_parallel_optimized.py \
    --config configs/train_final_design.yaml \
    --resume models/final_design/parallel_final_step30000.pt
```

**Expected output**:
```
[Worker 0] Phase: easy (120/120 episodes)
[Worker 0] Phase: medium (60/60 episodes)
[Worker 0] Phase: hard (20/20 episodes)
[Worker 0] Phase: mix (50/50 episodes)
[Learner] Total updates: 20250, Epsilon: 0.05
[Learner] Saved: models/final_design/parallel_final_step81000.pt
```

---

# PHẦN 5: QUY TRÌNH ĐÁNH GIÁ (EVALUATION)

## 5.1 Baseline Controllers (Fair Comparison)

Tất cả baseline đều sử dụng **cycle 60s** và **3 actions** như RL để đảm bảo công bằng:

| Strategy | Action Selection | Mô tả |
|----------|------------------|-------|
| **Fixed** | **Static (1 action)** | Luôn chọn Action 1 (50/50). Không thích ứng. |
| **Actuated** | **Dynamic (3 actions)** | Chọn action dựa trên Gap-out logic (Extension). |
| **Webster** | **Dynamic (3 actions)** | Chọn action dựa trên công thức Webster (Queue ratio). |
| **Random** | **Random (3 actions)** | Chọn ngẫu nhiên (Baseline kém nhất). |
| **RL** | **Learned (3 actions)** | Chọn action tối ưu hóa Q-value. |

## 5.2 Các Scenario Đánh Giá

### 4 File Config Eval (đã tạo)

| Config | Scenario | Routes | Mục đích |
|--------|----------|--------|----------|
| `eval_1_seen_routes.yaml` | Training routes | 27 (9 × 3 levels) | Test in-distribution |
| `eval_2_unseen_routes_d500.yaml` | Unseen d500 | 30 | Route generalization |
| `eval_3_unseen_demand_d750.yaml` | Unseen demand d750 | 30 | Demand generalization |
| `eval_4_mixed_demand.yaml` | 350→500→650 | 30 | Adaptive response |

### Imbalance Types (tất cả routes)

| Type | NS:EW Ratio | Mục đích |
|------|-------------|----------|
| `ns_heavy` | 65:35 | Test RL adapts to NS-heavy traffic |
| `balanced` | 50:50 | Neutral case |
| `ew_heavy` | 35:65 | Test RL adapts to EW-heavy traffic |

→ **Fixed 50/50 KHÔNG được lợi thế không công bằng** (không như routes cũ balanced-only)

## 5.3 Lệnh Chạy Evaluation (CLI Override)

### Chạy từng controller riêng biệt

```bash
# Random controller (weakest baseline)
python scripts/eval.py --config configs/eval_1_seen_routes.yaml --policies random

# Fixed controller only
python scripts/eval.py --config configs/eval_1_seen_routes.yaml --policies fixed

# Actuated controller only
python scripts/eval.py --config configs/eval_1_seen_routes.yaml --policies actuated

# Webster controller only
python scripts/eval.py --config configs/eval_1_seen_routes.yaml --policies webster

# RL model only
python scripts/eval.py --config configs/eval_1_seen_routes.yaml \
    --policies rl \
    --rl-model models/final_design/parallel_final.pt
```

### Chạy tất cả controllers

```bash
# Scenario 1: Seen routes (training routes)
python scripts/eval.py --config configs/eval_1_seen_routes.yaml

# Scenario 2: Unseen routes d500
python scripts/eval.py --config configs/eval_2_unseen_routes_d500.yaml

# Scenario 3: Unseen demand d750
python scripts/eval.py --config configs/eval_3_unseen_demand_d750.yaml

# Scenario 4: Mixed demand (350→500→650)
python scripts/eval.py --config configs/eval_4_mixed_demand.yaml
```

### Kết hợp nhiều controllers

```bash
# So sánh Random vs RL (chứng minh học được gì)
python scripts/eval.py --config configs/eval_1_seen_routes.yaml \
    --policies random,rl \
    --rl-model models/final_design/parallel_final.pt

# So sánh Fixed vs Actuated
python scripts/eval.py --config configs/eval_1_seen_routes.yaml \
    --policies fixed,actuated

# So sánh tất cả baselines vs RL
python scripts/eval.py --config configs/eval_1_seen_routes.yaml \
    --policies random,fixed,actuated,webster,rl \
    --rl-model models/final_design/parallel_final.pt
```

## 5.4 Cấu Hình Đặc Điểm

```yaml
# Tất cả eval configs đều có:
env:
  sumo:
    cycle_options_sec: [60]  # Fixed 60s
    action_splits:           # 3 actions ONLY (fair)
      - [0.30, 0.70]
      - [0.50, 0.50]
      - [0.70, 0.30]
    max_sim_seconds: 1500    # Horizon
```

## 5.5 Output

### CSV Format

```csv
policy,demand,seed,avg_wait_time_corr,completion_rate,teleport_rate,route_file
fixed,500,42,45.2,0.95,0.02,bignet_d500_ns_heavy_seed10001.rou.xml
actuated,500,42,42.1,0.96,0.01,...
webster,500,42,40.5,0.97,0.01,...
rl_full,500,42,38.2,0.98,0.005,...
```

### Output Files

| Config | Output Path |
|--------|-------------|
| eval_1 | `results/eval_1_seen_routes.csv` |
| eval_2 | `results/eval_2_unseen_routes_d500.csv` |
| eval_3 | `results/eval_3_unseen_demand_d750.csv` |
| eval_4 | `results/eval_4_mixed_demand.csv` |

---

# PHỤ LỤC: SO SÁNH CŨ/MỚI

## Thiết Kế Cũ vs Final Design

| Aspect | Old Design | **Final Design** |
|--------|------------|------------------|
| **Action space** | 15 (3 cycles × 5 splits) | **3 (1 cycle × 3 splits)** |
| **Cycle** | 60s, 90s, 120s | **60s fixed** |
| **Splits** | 0.3, 0.4, 0.5, 0.6, 0.7 | **0.3, 0.5, 0.7** |
| **Reward** | SMDP v5 + spillback | **simple_clipped** |
| **Hidden layers** | [192, 192] | **[128, 128]** |
| **Training mode** | Sequential 500 ep | **Parallel 12×250 ep** |
| **Curriculum** | 3 phases | **4 phases (+mix)** |
| **Baselines** | Variable cycles, 5 splits | **60s fixed, 3 splits** |
| **Eval routes** | Balanced only | **3 imbalance types** |

## Code References

| Module | File | Mô tả |
|--------|------|-------|
| **Training** | `scripts/train_parallel_optimized.py` | Parallel training 12 workers |
| **Evaluation** | `scripts/eval.py` | Unified eval với CLI override |
| **Route Gen** | `scripts/generate_eval_routes.py` | Tạo routes mới cho eval |
| **Environment** | `env/sumo_env.py` | SUMO wrapper |
| **Agent** | `rl/agent.py` | Dueling DQN |
| **Baselines** | `controllers/` | fixed_time, actuated, webster |
| **Config Train** | `configs/train_final_design.yaml` | Training config |
| **Config Eval** | `configs/eval_1_*.yaml` đến `eval_4_*.yaml` | 4 eval scenarios |

---

**End of Document**

