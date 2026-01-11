# Tổng Quan Cấu Trúc Dự Án Traffic Signal Control RL

> Tài liệu mô tả chức năng từng file, nhóm file tương tự, và các hướng tiếp cận thay thế.
> **Cập nhật:** 2026-01-11

---

## 1. Tổng Quan Thư Mục

```
GroupProject2/
├── env/           # Môi trường mô phỏng (RL Environment)
├── rl/            # Thuật toán học tăng cường
├── controllers/   # Controllers baseline truyền thống
├── scripts/       # Scripts điều khiển, training, đánh giá
├── configs/       # File cấu hình YAML
├── networks/      # BIGNET.net.xml và route variants
├── tests/         # Unit tests
└── docs/          # Tài liệu
```

---

## 2. Module `env/` — Môi Trường RL

| File | Chức Năng | Mức Độ Quan Trọng |
|------|-----------|-------------------|
| [sumo_env.py](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py) | **Core**: Môi trường SUMO chính, định nghĩa MDP (state, action, reward) | ⭐⭐⭐ Critical |
| [base_env.py](file:///c:/Users/Dell/GroupProject2/env/base_env.py) | Abstract base class cho environments | ⭐ Support |
| [mdp_metrics.py](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py) | Tính toán metrics MDP (queue counts, waiting time, fairness) | ⭐⭐ Important |
| [kpi.py](file:///c:/Users/Dell/GroupProject2/env/kpi.py) | KPI tracker (throughput, delay, teleports) | ⭐⭐ Important |
| [normalization.py](file:///c:/Users/Dell/GroupProject2/env/normalization.py) | Chuẩn hóa state vector | ⭐⭐ Important |
| [stochastic_demand.py](file:///c:/Users/Dell/GroupProject2/env/stochastic_demand.py) | Sinh demand ngẫu nhiên cho training | ⭐ Optional |

---

## 3. Module `rl/` — Thuật Toán Học Tăng Cường

| File | Chức Năng | Mức Độ Quan Trọng |
|------|-----------|-------------------|
| [agent.py](file:///c:/Users/Dell/GroupProject2/rl/agent.py) | **Core**: DQN Agent với time-aware gamma | ⭐⭐⭐ Critical |
| [dueling_dqn.py](file:///c:/Users/Dell/GroupProject2/rl/dueling_dqn.py) | Mạng Dueling DQN (Value + Advantage) | ⭐⭐⭐ Critical |
| [replay_buffer.py](file:///c:/Users/Dell/GroupProject2/rl/replay_buffer.py) | Experience Replay Buffer với per-transition gamma | ⭐⭐⭐ Critical |
| [cycle_tracker.py](file:///c:/Users/Dell/GroupProject2/rl/cycle_tracker.py) | Theo dõi cycle length distribution | ⭐ Analysis |
| [inference.py](file:///c:/Users/Dell/GroupProject2/rl/inference.py) | Inference mode (không exploration) | ⭐ Utility |
| [utils.py](file:///c:/Users/Dell/GroupProject2/rl/utils.py) | Helper functions (load YAML, set seed) | ⭐ Utility |

---

## 4. Module `controllers/` — Baseline Controllers

| File | Chức Năng | Loại |
|------|-----------|------|
| [fixed_time.py](file:///c:/Users/Dell/GroupProject2/controllers/fixed_time.py) | Fixed-time controller (cycle cố định) | Baseline |
| [max_pressure.py](file:///c:/Users/Dell/GroupProject2/controllers/max_pressure.py) | Max-Pressure controller (actuated) | Baseline |

---

## 5. Module `scripts/` — Scripts Điều Khiển

### 5.1 Training & Evaluation (Core)

| File | Chức Năng |
|------|-----------|
| [train.py](file:///c:/Users/Dell/GroupProject2/scripts/train.py) | **Main training script** - Curriculum learning, route pool |
| [eval.py](file:///c:/Users/Dell/GroupProject2/scripts/eval.py) | **Đánh giá model** - Chạy episodes và tính KPIs |
| [common.py](file:///c:/Users/Dell/GroupProject2/scripts/common.py) | **Shared utilities** - build_env, config loading |

### 5.2 Validation & Diagnostics

| File | Chức Năng |
|------|-----------|
| [validation.py](file:///c:/Users/Dell/GroupProject2/scripts/validation.py) | Validate config và environment setup |
| [doctor.py](file:///c:/Users/Dell/GroupProject2/scripts/doctor.py) | Chẩn đoán vấn đề trong codebase |
| [check_phase_sync.py](file:///c:/Users/Dell/GroupProject2/scripts/check_phase_sync.py) | Kiểm tra đồng bộ phase giữa các TLS |
| [semantic_probe_state.py](file:///c:/Users/Dell/GroupProject2/scripts/semantic_probe_state.py) | Kiểm tra semantic của state vector |
| [verify_configs.py](file:///c:/Users/Dell/GroupProject2/scripts/verify_configs.py) | Validate các file config YAML |

### 5.3 Normalization

| File | Chức Năng |
|------|-----------|
| [collect_norm_stats.py](file:///c:/Users/Dell/GroupProject2/scripts/collect_norm_stats.py) | Thu thập thống kê để normalize state |
| [collect_norm_curriculum.py](file:///c:/Users/Dell/GroupProject2/scripts/collect_norm_curriculum.py) | Norm stats cho curriculum (multi-phase) |
| [config_normalization.py](file:///c:/Users/Dell/GroupProject2/scripts/config_normalization.py) | Config normalization helpers |

### 5.4 Route Generation

| File | Chức Năng |
|------|-----------|
| [generate_randomized_routes.py](file:///c:/Users/Dell/GroupProject2/scripts/generate_randomized_routes.py) | Tạo routes ngẫu nhiên |
| [generate_jtr_data.py](file:///c:/Users/Dell/GroupProject2/scripts/generate_jtr_data.py) | Tạo Junction Turn Ratio data |
| [generate_route_batch.py](file:///c:/Users/Dell/GroupProject2/scripts/generate_route_batch.py) | Batch route generation |
| [scale_demand_batch.py](file:///c:/Users/Dell/GroupProject2/scripts/scale_demand_batch.py) | Scale demand cho curriculum |
| [route_pool_loader.py](file:///c:/Users/Dell/GroupProject2/scripts/route_pool_loader.py) | Load route pool từ manifest |

### 5.5 Visualization & Analysis

| File | Chức Năng |
|------|-----------|
| [plot_kpis.py](file:///c:/Users/Dell/GroupProject2/scripts/plot_kpis.py) | Vẽ KPI comparison charts |
| [plot_eval.py](file:///c:/Users/Dell/GroupProject2/scripts/plot_eval.py) | Vẽ evaluation results |
| [aggregate_kpis.py](file:///c:/Users/Dell/GroupProject2/scripts/aggregate_kpis.py) | Tổng hợp KPIs từ nhiều runs |
| [compare_controllers.py](file:///c:/Users/Dell/GroupProject2/scripts/compare_controllers.py) | So sánh các controllers |

---

## 6. Module `configs/` — File Cấu Hình

### Training Configs

| File | Mô Tả |
|------|-------|
| [train_1.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml) | Config training cơ bản |
| [train_bignet_short.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_bignet_short.yaml) | BIGNET 9-TLS, 300 episodes, curriculum 5 phases |
| [train_bignet_9tls.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_bignet_9tls.yaml) | Config cho 9 TLS network |
| [train_bignet_9tls_long.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_bignet_9tls_long.yaml) | Config dài hơn (teleport=300s) |

### Evaluation Configs

| File | Mô Tả |
|------|-------|
| [eval_1.yaml](file:///c:/Users/Dell/GroupProject2/configs/eval_1.yaml) | Eval config cơ bản |
| [eval_bignet_9tls.yaml](file:///c:/Users/Dell/GroupProject2/configs/eval_bignet_9tls.yaml) | Eval cho 9 TLS |
| [eval_bignet_9tls_long.yaml](file:///c:/Users/Dell/GroupProject2/configs/eval_bignet_9tls_long.yaml) | Eval dài hơn |

---

## 7. Module `networks/` — SUMO Network Files

```
networks/
├── BIGNET.net.xml       # Mạng 9 TLS chính
└── variants/
    ├── train/           # Route files cho training (manifest_scale*.txt)
    └── eval/            # Route files cho evaluation
```

---

## 8. Sơ Đồ Phụ Thuộc

```
                    ┌─────────────────┐
                    │   configs/*.yaml │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ scripts/common.py│ ← Load config, build env
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ scripts/train.py│ │ scripts/eval.py │ │ controllers/*   │
│ (Training loop) │ │ (Evaluation)    │ │ (Baselines)     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  rl/agent.py    │ ← DQN Agent
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ env/sumo_env.py │ ← SUMO Environment
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   SUMO Simulator │
                    └─────────────────┘
```

---

## 9. Core Files (Không Sửa Khi Không Cần Thiết)

1. [env/sumo_env.py](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py) - Logic MDP
2. [rl/agent.py](file:///c:/Users/Dell/GroupProject2/rl/agent.py) - Thuật toán học
3. [rl/dueling_dqn.py](file:///c:/Users/Dell/GroupProject2/rl/dueling_dqn.py) - Network architecture
4. [scripts/train.py](file:///c:/Users/Dell/GroupProject2/scripts/train.py) - Training loop
