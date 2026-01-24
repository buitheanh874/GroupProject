# TÀI LIỆU KỸ THUẬT TOÀN DIỆN
## Điều Khiển Đèn Giao Thông Bằng Học Tăng Cường Đa Tác Tử

---

# PHẦN 1: ĐỊNH NGHĨA BÀI TOÁN VÀ MÔ TẢ TÌNH HUỐNG

## 1.1 Tổng Quan Bài Toán

### Vấn Đề Cần Giải Quyết

Bài toán điều khiển đèn giao thông tại mạng lưới 9 giao lộ đô thị với các đặc điểm:

- **Mô hình mạng lưới**: Mạng lưới giả lập dạng lưới 3x3 (Synthetic 3x3 Grid)
- **Loại giao thông**: Hỗn hợp (xe máy 86%, ô tô 12%, xe buýt 2%)
- **Mục tiêu**: Tối thiểu hóa thời gian chờ đợi toàn mạng lưới
- **Ràng buộc**: An toàn (min green, yellow, all-red), tránh spillback

### Định Nghĩa Hình Thức

**Bài toán tối ưu hóa**:
$$
\min_{\pi} \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t \cdot W_t \right]
$$

Trong đó:
- $\pi$: Policy điều khiển đèn
- $W_t$: Tổng thời gian chờ tại bước $t$
- $\gamma$: Hệ số chiết khấu (0.99)
- $T$: Horizon mô phỏng (1800s)

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
| **Giao lộ trung tâm** | J0 (đo spillback) |
| **Số làn mỗi hướng** | 3 làn (NS và EW) |
| **Biên mạng lưới** | 12 cạnh (E29-E40) |

### Các Mức Nhu Cầu Giao Thông

| Mức | Demand (xe/giờ) | Mục đích |
|-----|-----------------|----------|
| Thấp | 500 | Training Phase 1 |
| Trung bình | 750 | Training Phase 2 |
| Cao | 1000 | Training Phase 3 |

---

## 1.3 Mô Hình MDP (Markov Decision Process)

### Không Gian Trạng Thái (State Space)

**Kích thước**: 14 chiều cho mỗi TLS

```
s = [q_N, q_E, q_S, q_W, w_N, w_E, w_S, w_W, o_N, o_E, o_S, o_W, n_norm, φ_norm]
     ─────────────────  ─────────────────  ─────────────────  ─────────────────
     Hàng đợi (0-3)     Chờ đợi (4-7)      Chiếm dụng (8-11)  Global (12-13)
```

| Nhóm | Dims | Mô tả | Đơn vị |
|------|------|-------|--------|
| Queue | 0-3 | Số xe xếp hàng theo 4 hướng | xe |
| Wait | 4-7 | Thời gian chờ tích lũy | xe-giây |
| Occupancy | 8-11 | Chiếm dụng hạ lưu | [0,1] |
| Global | 12-13 | Scalar toàn mạng | normalized |

### Không Gian Hành Động (Action Space)

**Kích thước**: 15 hành động rời rạc = 3 chu kỳ × 5 tỉ lệ chia

| Chu kỳ (s) | ρ_NS = 0.3 | ρ_NS = 0.4 | ρ_NS = 0.5 | ρ_NS = 0.6 | ρ_NS = 0.7 |
|------------|------------|------------|------------|------------|------------|
| 60 | Action 0 | Action 1 | Action 2 | Action 3 | Action 4 |
| 90 | Action 5 | Action 6 | Action 7 | Action 8 | Action 9 |
| 120 | Action 10 | Action 11 | Action 12 | Action 13 | Action 14 |

### Hàm Phần Thưởng (Reward Function)

**Công thức SMDP v5**:
$$
R = -\frac{W_{total}}{N \cdot t_{ref}} - \left(\frac{\alpha}{M} \sum_{d} Occ_d^2\right) \cdot \frac{\Delta t}{t_{ref}}
$$

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| $t_{ref}$ | 60.0 | Thời gian tham chiếu (s) |
| $\alpha$ | 3.0 | Trọng số phạt spillback |
| $M$ | 4 | Số liên kết hạ lưu |
| $N$ | n_present | Số xe hiện tại trong mạng |

---

# PHẦN 2: THIẾT KẾ HỆ THỐNG VÀ THUẬT TOÁN

## 2.1 Workflow Tổng Quan

```
┌──────────────────────────────────────────────────────────────────┐
│                      HỆ THỐNG ĐIỀU KHIỂN                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   SUMO      │────▶│  Environment│────▶│   Agent     │        │
│  │  Simulator  │◀────│   Wrapper   │◀────│  (DQN)      │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│        │                   │                   │                 │
│        ▼                   ▼                   ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ TraCI API   │     │ State 14D   │     │ Dueling DQN │        │
│  │ (Control)   │     │ Reward Calc │     │ [192, 192]  │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 2.2 Kiến Trúc MARL (Multi-Agent RL)

### Parameter Sharing Architecture

```
                    ┌─────────────────────────────────┐
                    │     SHARED DUELING DQN          │
                    │        [192, 192]               │
                    │                                 │
                    │  Input: 14D state               │
                    │  Output: 15 Q-values            │
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
| **Network** | Dueling DQN với 2 hidden layers [192, 192] |
| **Value Head** | Linear(192 → 1) cho V(s) |
| **Advantage Head** | Linear(192 → 15) cho A(s,a) |
| **Q-value** | Q(s,a) = V(s) + A(s,a) - mean(A) |

### Chia Sẻ Phần Thưởng

- **9 agents** dùng chung 1 policy network
- **Global reward** được chia đều cho tất cả agents
- **Implicit coordination** qua dims 12-13 (global scalars)

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

### Time-Aware Gamma

Để xử lý SMDP với decision step thay đổi:
$$
\gamma_{effective} = \gamma_0^{\frac{\Delta t}{t_{ref}}}
$$

Trong đó $\gamma_0 = 0.99$, $t_{ref} = 60s$.

---

# PHẦN 3: THIẾT LẬP MÔI TRƯỜNG

## 3.1 Workflow Thiết Lập

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT SETUP                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Network Definition        2. Route Generation              │
│   ┌───────────────────┐        ┌───────────────────┐            │
│   │ BIGNET.net.xml    │        │ Route Generator   │            │
│   │ - 9 junctions     │        │ - Demand levels   │            │
│   │ - Lane configs    │        │ - Turn ratios     │            │
│   │ - TLS programs    │        │ - Vehicle types   │            │
│   └───────────────────┘        └───────────────────┘            │
│            │                            │                        │
│            ▼                            ▼                        │
│   3. Normalization             4. Configuration                  │
│   ┌───────────────────┐        ┌───────────────────┐            │
│   │ norm_turn801010   │        │ train_500ep.yaml  │            │
│   │ - Mean/Std 14D    │        │ - Agent params    │            │
│   │ - 7560 samples    │        │ - Curriculum      │            │
│   └───────────────────┘        └───────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 Cấu Hình SUMO

### File Mạng Lưới

| File | Mô tả |
|------|-------|
| `networks/BIGNET.net.xml` | Network definition với 9 giao lộ |

### Tham Số Mô Phỏng

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `step_length_sec` | 1.0 | Bước mô phỏng (s) |
| `max_sim_seconds` | 1800 | Thời gian tối đa (s) |
| `time-to-teleport` | 300 | Timeout teleport (s) |
| `yellow_sec` | 3 | Thời gian vàng (s) |
| `all_red_sec` | 2 | Thời gian đỏ toàn bộ (s) |
| `g_min_sec` | 10 | Thời gian xanh tối thiểu (s) |

### Cấu Hình TLS

```yaml
tls_ids: ["J0", "J1", "J2", "J3", "J4", "J6", "J7", "J14", "J17"]
center_tls_id: "J0"
downstream_links:
  N: "E3"
  E: "E0"
  S: "E2"
  W: "E1"
```

## 3.3 Route Files

### Cấu Trúc Thư Mục

```
networks/variants/train_turn801010/
├── 500/                          # ~100 route files
├── 750/                          # ~100 route files
├── 1000/                         # ~100 route files
├── manifest_mixed_phase1.txt     # Phase 1 curriculum
├── manifest_mixed_phase2.txt     # Phase 2 curriculum
└── manifest_mixed_phase3.txt     # Phase 3 curriculum
```

### Đặc Điểm Route

| Thuộc tính | Giá trị |
|------------|---------|
| **Turn ratios** | 80% thẳng, 10% trái, 10% phải |
| **Vehicle mix** | 86% xe máy, 12% ô tô, 2% bus |
| **Arrival pattern** | Poisson với λ = demand/3600 |
| **Duration** | 1800 giây |

---

# PHẦN 4: QUY TRÌNH HUẤN LUYỆN (TRAINING)

## 4.1 Workflow Huấn Luyện

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. INITIALIZATION                                        │   │
│   │    - Load config (train_500ep.yaml)                      │   │
│   │    - Initialize Dueling DQN [192, 192]                   │   │
│   │    - Create Replay Buffer (200,000)                      │   │
│   │    - Load normalization (norm_turn801010.json)           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 2. CURRICULUM PHASE LOOP                                 │   │
│   │    Phase 1 (Easy):     150 episodes - 70% d500          │   │
│   │    Phase 2 (Moderate): 200 episodes - 40% d750          │   │
│   │    Phase 3 (Hard):     150 episodes - 65% d1000         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 3. EPISODE LOOP (per phase)                              │   │
│   │    ┌──────────────────────────────────────────────────┐ │   │
│   │    │ a. Reset environment với route từ manifest       │ │   │
│   │    │ b. Collect states từ 9 TLS                       │ │   │
│   │    │ c. Select actions (ε-greedy)                     │ │   │
│   │    │ d. Execute in SUMO                               │ │   │
│   │    │ e. Compute global reward                         │ │   │
│   │    │ f. Store transitions                             │ │   │
│   │    │ g. Train (if buffer >= learning_starts)          │ │   │
│   │    └──────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 4. PERIODIC TASKS                                        │   │
│   │    - Smoke eval mỗi 25 episodes                         │   │
│   │    - Save model mỗi 50 episodes                         │   │
│   │    - Update target network mỗi 2000 steps               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 Chi Tiết Từng Bước

### Bước 1: Khởi Tạo

```python
# Load configuration
config = load_yaml_config("configs/train_500ep.yaml")

# Initialize agent with Dueling DQN
agent = DQNAgent(
    state_dim=14,
    action_dim=15,
    hidden_dims=[192, 192],
    gamma=0.99,
    learning_rate=0.0005,
    replay_buffer_size=200000,
)

# Load normalization
normalizer = StateNormalizer("configs/norm_turn801010.json")
```

### Bước 2: Curriculum Learning

| Phase | Episodes | Route Mix | Mô tả |
|-------|----------|-----------|-------|
| Phase 1 | 1-150 | 70% d500, 20% d750, 10% d1000 | Khởi động dễ |
| Phase 2 | 151-350 | 20% d500, 40% d750, 40% d1000 | Học chính |
| Phase 3 | 351-500 | 10% d500, 25% d750, 65% d1000 | Thách thức |

### Bước 3: Episode Loop

```python
for episode in range(1, 501):
    # Switch curriculum phase if needed
    phase = get_phase_for_episode(episode)
    route_file = sample_from_manifest(phase.manifest)
    
    # Reset environment
    states = env.reset(route_file)  # Dict[tls_id -> 14D state]
    
    while not done:
        # Select actions with ε-greedy
        actions = {}
        for tls_id, state in states.items():
            if random.random() < epsilon:
                actions[tls_id] = random.randint(0, 14)
            else:
                q_values = agent.get_q_values(state)
                actions[tls_id] = argmax(q_values)
        
        # Step environment
        next_states, global_reward, done, info = env.step(actions)
        
        # Store transitions (same reward for all agents)
        for tls_id in states:
            agent.store(states[tls_id], actions[tls_id], global_reward, 
                       next_states[tls_id], done)
        
        # Train if ready
        if agent.buffer_size >= 400:  # learning_starts
            agent.update()
        
        states = next_states
```

## 4.3 Hyperparameters

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| **hidden_dims** | [192, 192] | Kích thước 2 hidden layers |
| **gamma** | 0.99 | Base discount factor |
| **learning_rate** | 0.0005 | Adam optimizer LR |
| **batch_size** | 256 | Mini-batch size |
| **replay_buffer_size** | 200,000 | Replay buffer capacity |
| **target_update_freq** | 2,000 | Steps between target sync |
| **learning_starts** | 400 | Steps before learning |
| **train_freq** | 4 | Steps between updates |
| **eps_start** | 1.0 | Initial epsilon |
| **eps_end** | 0.08 | Final epsilon |
| **eps_decay_steps** | 50,000 | Epsilon decay duration |
| **clip_grad_norm** | 10.0 | Gradient clipping |

## 4.4 Lệnh Chạy Training

```bash
# Training 500 episodes với curriculum
python scripts/train.py --config configs/train_500ep.yaml

# Resume từ checkpoint
python scripts/train.py --config configs/train_500ep.yaml \
    --resume models/train_500ep/episode_300.pt
```

---

# PHẦN 5: QUY TRÌNH ĐÁNH GIÁ (EVALUATION)

## 5.1 Workflow Đánh Giá

```
┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. SETUP                                                 │   │
│   │    - Load config (eval.yaml)                             │   │
│   │    - Load trained model (best_model.pt)                  │   │
│   │    - Initialize baselines (Fixed, Actuated, Webster)     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 2. MATRIX EVALUATION                                     │   │
│   │    ┌────────────────────────────────────────────────┐   │   │
│   │    │  Demands: [500, 750, 1000]                     │   │   │
│   │    │  Seeds:   [42, 43, ..., 51] (10 seeds)         │   │   │
│   │    │  Policies: [Fixed, Actuated, Webster, RL]      │   │   │
│   │    │                                                │   │   │
│   │    │  Total runs = 3 × 10 × 4 = 120 episodes        │   │   │
│   │    └────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 3. PER-EPISODE EVALUATION                                │   │
│   │    a. Load route file cho (demand, seed)                │   │
│   │    b. Run policy (greedy, không exploration)            │   │
│   │    c. Collect KPIs:                                     │   │
│   │       - avg_wait_time_corr                              │   │
│   │       - avg_travel_time_corr                            │   │
│   │       - throughput_corr                                 │   │
│   │       - completion_rate                                 │   │
│   │       - teleport_rate                                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 4. RESULTS AGGREGATION                                   │   │
│   │    - Export to CSV                                       │   │
│   │    - Compute mean ± std per (policy, demand)             │   │
│   │    - Generate comparison tables                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 Chi Tiết Từng Bước

### Bước 1: Thiết Lập

```python
# Load config
config = load_yaml_config("configs/eval.yaml")

# Load trained RL model
agent = DQNAgent(state_dim=14, action_dim=15, hidden_dims=[192, 192])
agent.load("models/best_model.pt")
agent.eval()  # Disable exploration

# Initialize baselines
fixed = FixedTimeController(action_space, target_split=(0.5, 0.5))
actuated = ActuatedController(action_space)
webster = WebsterController(action_space)
```

### Bước 2: Ma Trận Đánh Giá

| | Demand 500 | Demand 750 | Demand 1000 |
|---|------------|------------|-------------|
| **Seed 42** | Run | Run | Run |
| **Seed 43** | Run | Run | Run |
| **...** | ... | ... | ... |
| **Seed 51** | Run | Run | Run |

**Policies đánh giá**:
- `fixed`: Tỉ lệ cố định 50/50, chu kỳ 90s
- `actuated`: Gap-out logic với min/max green
- `webster`: Công thức Webster C_opt = (1.5L+5)/(1-Y)
- `rl_full`: Model RL đã train

### Bước 3: Metrics Thu Thập

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **avg_wait_time_corr** | W / (arrived × scale) | Thời gian chờ TB đã điều chỉnh |
| **avg_travel_time_corr** | T / (arrived × scale) | Thời gian đi TB đã điều chỉnh |
| **throughput_corr** | arrived / scale | Số xe hoàn thành đã điều chỉnh |
| **completion_rate** | arrived / departed | Tỉ lệ hoàn thành (%) |
| **teleport_rate** | teleports / (arrived + teleports) | Tỉ lệ teleport (%) |

**Scale adjustment**: Điều chỉnh metrics theo completion_rate để công bằng khi so sánh.

## 5.3 Cấu Hình Đánh Giá

```yaml
eval_matrix:
  policies: [fixed, actuated, webster]  # Bỏ max_pressure
  demands: [500, 750, 1000]
  seeds: 10
  horizon: 1500
  warmup: 300
  unseen: true  # Dùng route seeds khác với training
  output: results/eval_results.csv
```

## 5.4 Lệnh Chạy Evaluation

```bash
# Đánh giá tất cả baselines
python scripts/eval.py --config configs/eval.yaml

# Đánh giá RL model
python scripts/eval.py --config configs/eval.yaml \
    --policies rl_full \
    --rl-full-model models/best_model.pt

# Đánh giá với demand cụ thể
python scripts/eval.py --config configs/eval.yaml \
    --demands 750 \
    --policies fixed,actuated,webster

# Output CSV tùy chỉnh
python scripts/eval.py --config configs/eval.yaml \
    --output results/my_eval.csv
```

## 5.5 Kết Quả Mong Đợi

### Format CSV Output

```csv
policy,demand,seed,horizon_sec,warmup_sec,avg_wait_time_corr,avg_travel_time_corr,throughput_corr,completion_rate,teleport_rate,arrived_vehicles
fixed,500,42,1500,300,45.2,120.5,450,0.95,0.02,425
actuated,500,42,1500,300,42.1,115.3,455,0.96,0.01,437
webster,500,42,1500,300,40.5,112.8,460,0.97,0.01,446
rl_full,500,42,1500,300,38.2,108.5,468,0.98,0.005,458
```

### Bảng So Sánh (Aggregated)

| Policy | Demand | Wait Time (s) | Travel Time (s) | Throughput | Completion |
|--------|--------|---------------|-----------------|------------|------------|
| Fixed | 750 | 52.3 ± 5.2 | 135.4 ± 12.1 | 680 ± 45 | 0.94 ± 0.02 |
| Actuated | 750 | 48.1 ± 4.8 | 128.6 ± 10.5 | 695 ± 42 | 0.95 ± 0.02 |
| Webster | 750 | 46.5 ± 4.5 | 125.2 ± 9.8 | 702 ± 40 | 0.96 ± 0.01 |
| **RL** | 750 | **42.8 ± 3.9** | **118.5 ± 8.2** | **720 ± 35** | **0.97 ± 0.01** |

---

# PHỤ LỤC: CODE REFERENCES

| Module | File | Mô tả |
|--------|------|-------|
| Training | `scripts/train.py` | Main training loop với curriculum |
| Evaluation | `scripts/eval.py` | Matrix evaluation script |
| Environment | `env/sumo_env.py` | SUMO environment wrapper |
| Agent | `rl/agent.py` | DQN agent implementation |
| Network | `rl/dueling_dqn.py` | Dueling DQN architecture |
| Reward | `env/mdp_metrics.py` | Reward computation |
| Baselines | `controllers/` | Fixed, Actuated, Webster |
| Config | `configs/train_500ep.yaml` | Training configuration |
| Config | `configs/eval.yaml` | Evaluation configuration |
