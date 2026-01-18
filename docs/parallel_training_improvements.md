# Báo cáo: Cải tiến Parallel Training với Epsilon Decay

## Tổng quan

Tài liệu này mô tả các thay đổi kỹ thuật quan trọng được thực hiện để cải thiện hệ thống parallel training cho bài toán điều khiển đèn giao thông bằng Deep Q-Learning.

---

## 1. Vấn đề ban đầu

### 1.1. Epsilon cố định
- **Trước đây**: Mỗi worker có epsilon cố định (0.20, 0.23, 0.26, 0.29)
- **Vấn đề**: Agent không giảm exploration theo thời gian → khó hội tụ về policy tối ưu
- **Ví dụ**: Sau 1000 episodes, agent vẫn random 20-29% actions thay vì exploit kiến thức đã học

### 1.2. Resume không liên tục
- **Trước đây**: Khi resume training, epsilon reset về giá trị ban đầu
- **Vấn đề**: Policy "quên" tiến độ học → phải học lại từ đầu
- **Ví dụ**: Training đến episode 800 (ε≈0.15), crash và resume → epsilon nhảy về 0.60

### 1.3. Thiếu logging chi tiết
- **Trước đây**: Không theo dõi được diversity giữa các workers
- **Vấn đề**: Không biết workers có thực sự explore khác nhau không

---

## 2. Giải pháp đã triển khai

### 2.1. Epsilon Decay Schedule

**Công thức:**
```
Nếu global_step < warmup_steps:
    ε = ε_start × worker_multiplier
Nếu warmup_steps ≤ global_step < warmup_steps + decay_steps:
    progress = (global_step - warmup_steps) / decay_steps
    ε_base = ε_start - progress × (ε_start - ε_end)
    ε = ε_base × worker_multiplier
Nếu global_step ≥ warmup_steps + decay_steps:
    ε = ε_end × worker_multiplier
```

**Tham số:**
- `ε_start = 0.60` (exploration cao ban đầu)
- `ε_end = 0.05` (exploitation sau khi học)
- `warmup_steps = 8,000` (giữ epsilon cao để fill replay buffer)
- `decay_steps = 60,000` (giảm dần epsilon)
- `worker_multipliers = [0.85, 0.95, 1.05, 1.15]` (diversity giữa workers)

**Kết quả:**
```
Worker 0: ε = 0.60 × 0.85 = 0.51 → 0.05 × 0.85 = 0.04
Worker 1: ε = 0.60 × 0.95 = 0.57 → 0.05 × 0.95 = 0.05
Worker 2: ε = 0.60 × 1.05 = 0.63 → 0.05 × 1.05 = 0.05
Worker 3: ε = 0.60 × 1.15 = 0.69 → 0.05 × 1.15 = 0.06
```

### 2.2. Shared Global Counter

**Cơ chế:**
- Sử dụng `multiprocessing.Value('l', 0)` (long integer 64-bit)
- Shared giữa 4 workers
- Atomic increment với lock để tránh race condition

**Định nghĩa:**
> **1 global_step = 1 joint decision step = 1 lần gọi env.step()**
> 
> Với 9 TLS agents, counter tăng **1 lần** mỗi step (không phải 9 lần)

**Code:**
```python
# Increment atomic
if global_step_counter is not None:
    with global_step_counter.get_lock():
        global_step_counter.value += 1
```

### 2.3. Resume với Epsilon Continuity

**Lưu checkpoint:**
```python
agent.save_checkpoint(path, {
    "learner_updates": learner_updates,
    "agent_transitions_total": agent_transitions_total,
    "global_env_steps_total": global_env_steps_total,
    "global_decision_steps": current_global_steps,  # MỚI
})
```

**Khôi phục:**
```python
resumed_global_steps = checkpoint.get("global_decision_steps", 0)
with global_step_counter.get_lock():
    global_step_counter.value = resumed_global_steps
```

**Lưu ý quan trọng:**
- Resume chỉ khôi phục: weights, optimizer, global_step_counter
- **KHÔNG** khôi phục: replay buffer
- Đây là **fine-tuning** chứ không phải true continuation

### 2.4. Enhanced Logging

**Per-episode logs:**
```
[Worker 0] Ep 1 | steps=11 | reward=-13254.16 | ε=0.510 | frac_random=0.45 | global=51 | worker_total_steps=11
```

**Ý nghĩa:**
- `steps`: Số decision steps trong episode
- `reward`: Tổng reward (âm = waiting time + penalties)
- `ε`: Epsilon hiện tại (sau khi nhân multiplier)
- `frac_random`: Tỷ lệ actions random (verify epsilon hoạt động)
- `global`: Global counter hiện tại
- `worker_total_steps`: Tổng steps của worker này

**Worker summary (khi kết thúc):**
```
[Worker 0] === SUMMARY ===
[Worker 0]   Total episodes: 300
[Worker 0]   Total steps: 5440
[Worker 0]   Approx sim_time: 136.00 hours
[Worker 0]   Multiplier: 0.85
```

---

## 3. Cách sử dụng

### 3.1. Training mới
```bash
python scripts/train_parallel.py --config configs/train_1.yaml
```

### 3.2. Resume từ checkpoint
```bash
python scripts/train_parallel.py --config configs/train_1.yaml --resume models/1/parallel_final_step21564.pt
```

**Khi resume, console sẽ hiển thị:**
```
[Fine-tuning] Loading checkpoint: models/1/parallel_final_step21564.pt
[Fine-tuning] Note: Replay buffer NOT restored (training dynamics partially restart)
[Fine-tuning] Restored epsilon clock: global_step=21564
```

### 3.3. Verify epsilon decay
```bash
python scripts/test_epsilon_decay.py
```

---

## 4. Theo dõi training

### 4.1. Metrics quan trọng

**Worker logs (mỗi episode):**
- `reward`: Tăng dần (ít âm hơn) → policy cải thiện
- `ε`: Giảm dần theo schedule → đúng decay
- `frac_random`: Giảm dần → agent exploit nhiều hơn

**Learner logs (mỗi 100 steps hoặc 30s):**
```
Step 5000 | Trans: 45000 | Global: 5000 | Pending: 256 | UTD_agent: 0.1111 | UTD_global: 1.00 | Loss: 0.0234
```

- `Loss`: Giảm dần → model đang học
- `UTD_agent`: Update-to-data ratio (target: 0.25)
- `Trans`: Tổng agent transitions (steps × 9 TLS)

### 4.2. Checkpoint tự động

**Định kỳ:** Mỗi 5 phút (300 giây)
```
[Checkpoint] Saved: models/1/parallel_ckpt_step5000.pt (global_decision_steps=5000)
```

**Khi crash/Ctrl+C:**
```
Saved: models/1/parallel_final_step5000.pt (global_decision_steps=5000)
```

### 4.3. Dấu hiệu hội tụ

✅ **Tốt:**
- Loss giảm và ổn định
- Reward tăng dần
- frac_random giảm theo epsilon
- Worker summary: steps/worker cân bằng (max/min < 1.2×)

❌ **Cần kiểm tra:**
- Loss tăng hoặc NaN
- Reward không cải thiện sau 200+ episodes
- frac_random không giảm (epsilon không decay)
- Worker nào đó có steps quá ít/nhiều

---

## 5. Cấu trúc thí nghiệm (Ablation Study)

### 5.1. Directory structure
```
experiments/ablation_epsilon_v1/
├── configs/
│   ├── v1_fixed.yaml           # Baseline: ε cố định 0.20
│   ├── v2_decay_nomult.yaml    # Decay, không có multipliers
│   └── v3_decay_mult.yaml      # Decay + multipliers (hiện tại)
├── runs/
│   ├── v1_fixed/
│   │   └── curriculum_seed42/
│   │       ├── checkpoints/
│   │       ├── logs/
│   │       └── eval/
│   ├── v2_decay_nomult/
│   └── v3_decay_mult/
└── summary/
    ├── eval_kpi_table.csv
    └── learning_curves.png
```

### 5.2. Quy trình thí nghiệm

**Bước 1: Train 3 variants × 5 seeds**
```bash
# Variant 1: Fixed epsilon
python scripts/train_parallel.py --config experiments/ablation_epsilon_v1/configs/v1_fixed.yaml

# Variant 2: Decay, no multiplier
python scripts/train_parallel.py --config experiments/ablation_epsilon_v1/configs/v2_decay_nomult.yaml

# Variant 3: Decay + multiplier (đang chạy)
python scripts/train_parallel.py --config configs/train_1.yaml
```

**Bước 2: Eval trên 4 demands**
- d400 (in-distribution)
- d800 (in-distribution, train demand)
- d1000 (in-distribution)
- d1200 (OOD demand)

**Bước 3: Tổng hợp kết quả**
- Mean ± std cho mỗi variant × demand
- KPI chính: `avg_wait` (average waiting time)
- KPI phụ: `p95_wait`, `throughput`, `teleport_rate`

---

## 6. Tham số cấu hình

### 6.1. Epsilon schedule (configs/train_1.yaml)
```yaml
exploration:
  eps_start: 0.60
  eps_end: 0.05
  warmup_global_steps: 8000
  eps_decay_steps: 60000

parallel:
  epsilon_worker_multipliers: [0.85, 0.95, 1.05, 1.15]
```

### 6.2. Training (configs/train_1.yaml)
```yaml
train:
  episodes: 1200

agent:
  learning_starts: 2000  # Bắt đầu training sau 2000 transitions
  train_freq: 4          # Train mỗi 4 transitions
  batch_size: 128
  gamma: 0.99
  learning_rate: 0.0003

parallel:
  num_actors: 4
  chunk_size: 256
  sync_every_updates: 100
```

---

## 7. Troubleshooting

### 7.1. Training không bắt đầu
**Triệu chứng:** Chỉ thấy worker logs, không thấy learner logs

**Nguyên nhân:** Chưa đủ `learning_starts=2000` transitions

**Giải pháp:** Đợi ~25 episodes (2000 / 9 TLS / ~9 steps/ep)

### 7.2. Epsilon không giảm
**Triệu chứng:** `ε` và `frac_random` không đổi

**Kiểm tra:**
```bash
python scripts/test_epsilon_decay.py
```

### 7.3. Loss = NaN
**Nguyên nhân:** Gradient explosion

**Giải pháp:**
- Giảm learning rate: `0.0003 → 0.0001`
- Tăng gradient clipping: `clip_grad_norm: 10.0 → 5.0`

### 7.4. Resume không hoạt động
**Kiểm tra checkpoint có `global_decision_steps`:**
```python
import torch
ckpt = torch.load("models/1/parallel_final_step21564.pt")
print(ckpt.keys())  # Phải có 'global_decision_steps'
```

---

## 8. Kết luận

### 8.1. Cải tiến đã đạt được
✅ Epsilon decay schedule chuẩn khoa học  
✅ Resume training liên tục (epsilon clock)  
✅ Diversity giữa workers (multipliers)  
✅ Logging chi tiết để verify  
✅ Autosave & crash recovery  

### 8.2. Công việc tiếp theo
- [ ] Chạy đủ 15 runs (3 variants × 5 seeds)
- [ ] Eval trên 4 demands
- [ ] Tổng hợp bảng mean ± std
- [ ] So sánh với baselines (Fixed-time, Max-Pressure)

### 8.3. Thời gian ước tính
- 1 run (1200 episodes): ~2-3 giờ
- 15 runs: ~30-45 giờ
- Eval: ~2 giờ
- **Tổng**: ~2-3 ngày (chạy song song trên nhiều máy)
