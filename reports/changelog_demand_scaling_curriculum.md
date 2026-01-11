# Changelog: Demand Scaling & Curriculum Learning Implementation

> Báo cáo thay đổi kể từ [project_file_documentation.md](file:///C:/Users/Dell/.gemini/antigravity/brain/a876ab2c-cb85-44fe-97cb-3742a2560245/project_file_documentation.md.resolved)  
> Ngày: 2026-01-11

---

## 1. Tổng Quan Thay Đổi

### Mục tiêu Nghiên Cứu

**Mục tiêu chính (giống papers benchmark):**
- ✅ Giảm **travel time / delay**
- ✅ Tăng **throughput**
- ✅ Giảm **queue length**

**Extension (đóng góp mới):**
- ✅ Xử lý **deadlock/gridlock** ở high-demand scenarios
- ✅ Mở rộng demand range: 500-3000 veh/hr/lane (vs 300-700 trong literature)
- ✅ Curriculum learning để agent học từ dễ → khó, bao gồm cả deadlock recovery

### Research Contribution

> Nghiên cứu này mở rộng các phương pháp RL-based TSC hiện có (CoLight, PressLight, FRAP) theo hai hướng:
>
> 1. **Mở rộng phạm vi demand**: Từ 300-700 veh/hr/lane (benchmark) lên đến 3000 veh/hr/lane (realistic Hanoi peak conditions)
> 
> 2. **Xử lý deadlock**: Ở mức demand cao, gridlock/deadlock thường xảy ra. Agent được train qua curriculum learning để không chỉ **tối ưu các KPIs truyền thống** (delay, throughput, queue) mà còn học cách **phòng ngừa và phục hồi từ deadlock**.

### Phương pháp luận
Dựa trên các papers:
- **MetaLight** (Zang et al., AAAI 2020): Curriculum learning for TSC
- **Advanced-MPLight** (Zhang et al., arXiv 2021): Demand scaling experiments
- **CoLight/PressLight**: Benchmark demand levels (300-700 veh/hr/lane)

---

## 2. Files Mới Được Tạo

| File | Chức năng |
|------|-----------|
| [scripts/scale_demand_batch.py](file:///c:/Users/Dell/GroupProject2/scripts/scale_demand_batch.py) | Batch-generate scaled route files cho curriculum learning |
| [scripts/collect_norm_curriculum.py](file:///c:/Users/Dell/GroupProject2/scripts/collect_norm_curriculum.py) | Collect normalization stats từ TẤT CẢ curriculum phases |
| [scripts/generate_route_batch.py](file:///c:/Users/Dell/GroupProject2/scripts/generate_route_batch.py) | Batch generate routes (chưa hoạt động do thiếu SUMO tools) |

---

## 3. Files Được Chỉnh Sửa

### [env/sumo_env.py](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py)
**Mục đích:** Xử lý TraCI crash khi gridlock xảy ra

```diff
# Line 838-842: Guard check before simulation step
+ if self._traci is None or not self._connected:
+     if traci_error is None:
+         traci_error = RuntimeError("TraCI connection closed unexpectedly")
+     break

# Line 1235-1241: Extended error detection
  return (
      "connection already closed" in message
      or "connection closed" in message
+     or "socket reset" in message  # NEW: catches SUMO crash
  )
```

---

### [scripts/generate_jtr_data.py](file:///c:/Users/Dell/GroupProject2/scripts/generate_jtr_data.py)
**Mục đích:** Configurable demand baseline

```diff
- HANOI_BASE_FLOW_PER_LANE = 2000.0
+ HANOI_BASE_FLOW_PER_LANE = 1000.0  # Base for training (CoLight/FRAP use 300-700)
+ # With curriculum scaling 50%→200%, training covers 500-2000 veh/hr/lane
+ # Eval uses --base-flow 2000 for realistic Hanoi peak demand

# NEW: --base-flow CLI argument
+ parser.add_argument("--base-flow", type=float, default=HANOI_BASE_FLOW_PER_LANE,
+                     help="Base flow per lane in veh/hr (train=500, eval=2000 for Hanoi)")
```

---

### [scripts/collect_norm_stats.py](file:///c:/Users/Dell/GroupProject2/scripts/collect_norm_stats.py)
**Mục đích:** Support custom manifest cho curriculum

```diff
# NEW: --manifest CLI argument
+ parser.add_argument("--manifest", type=str, default=None,
+     help="Override route pool manifest (e.g., manifest_scale50.txt for 50%% demand)")

# Override route pool manifest if specified
+ if args.manifest is not None:
+     config.setdefault("train", {})
+     config["train"]["route_pool_manifest"] = str(args.manifest)
```

---

### [configs/train_1.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml)
**Mục đích:** Curriculum learning configuration

```yaml
# NEW: Curriculum section (lines 320-350)
curriculum:
  enabled: true  # Curriculum learning for gridlock handling
  # Base routes: 2000 veh/hr/lane (realistic Hanoi peak)
  # 25%=500, 50%=1000 (CoLight), 75%=1500, 100%=2000 (eval), 150%=3000 (stress)
  phases:
    - name: "phase1_warmup"
      episodes: 150
      demand_scale: 0.25
      route_pool_manifest: networks/variants/train/manifest_scale25.txt
      description: "500 veh/hr/lane - warmup, no gridlock"
    - name: "phase2_standard"
      episodes: 250
      demand_scale: 0.5
      route_pool_manifest: networks/variants/train/manifest_scale50.txt
      description: "1000 veh/hr/lane - CoLight/FRAP level"
    - name: "phase3_challenging"
      episodes: 250
      demand_scale: 0.75
      route_pool_manifest: networks/variants/train/manifest_scale75.txt
      description: "1500 veh/hr/lane - challenging, some congestion"
    - name: "phase4_eval"
      episodes: 200
      demand_scale: 1.0
      route_pool_manifest: networks/variants/train/manifest_scale100.txt
      description: "2000 veh/hr/lane - realistic Hanoi (= eval)"
    - name: "phase5_stress"
      episodes: 150
      demand_scale: 1.5
      route_pool_manifest: networks/variants/train/manifest_scale150.txt
      description: "3000 veh/hr/lane - stress test, gridlock handling"
```

---

## 4. Route Files Structure

### Trước đây
```
networks/variants/train/
├── bignet_train_seed*.rou.xml  (50 files, 2000 veh/hr/lane)
└── manifest_1.txt
```

### Bây giờ
```
networks/variants/train/
├── bignet_train_seed*.rou.xml  (50 base files, 2000 veh/hr/lane)
├── scaled_25/   (50 files @ 500 veh/hr/lane)
├── scaled_50/   (50 files @ 1000 veh/hr/lane)
├── scaled_75/   (50 files @ 1500 veh/hr/lane)
├── scaled_100/  (50 files @ 2000 veh/hr/lane)
├── scaled_150/  (50 files @ 3000 veh/hr/lane)
├── manifest_scale25.txt
├── manifest_scale50.txt
├── manifest_scale75.txt
├── manifest_scale100.txt
└── manifest_scale150.txt
```

**Tổng cộng:** 250 route files cho curriculum training

---

## 5. Curriculum Learning Design (v3 - Deadlock Focus)

> [!IMPORTANT]
> **Updated 2026-01-11:** Base demand changed to **1200 veh/hr/lane** to focus on deadlock handling.
> This is **2x higher** than CoLight/FRAP benchmark (300-700), enabling research on gridlock scenarios.

### Episode Distribution (v3 - Final)

| Phase | Logical | Demand | Actual Scale | % | 1000 ep | 300 ep | Purpose |
|-------|---------|--------|--------------|---|---------|--------|---------|
| 1 | 50% | 600 | 30% of 2000 | 10% | 100 | 30 | Warmup |
| 2 | 70% | 840 | 42% of 2000 | 15% | 150 | 45 | Moderate |
| 3 ⭐ | **100%** | **1200** | 60% of 2000 | **45%** | **450** | **135** | **PRIMARY EVAL** |
| 4 🔶 | 125% | 1500 | 75% of 2000 | 20% | 200 | 60 | Gridlock training |
| 5 | 150% | 1800 | 90% of 2000 | 10% | 100 | 30 | Extreme stress |

**Focus:** 70% safe (≤1200), 30% gridlock (>1200), 45% on eval level

### Route Files Structure (v3)
```
networks/variants/train/
├── bignet_train_seed*.rou.xml  (50 base files @ 2000 veh/hr/lane)
├── scaled_30/   (30% = 600 veh/hr = 50% of 1200)
├── scaled_42/   (42% = 840 veh/hr = 70% of 1200)
├── scaled_60/   (60% = 1200 veh/hr = 100% of 1200) ⭐
├── scaled_75/   (75% = 1500 veh/hr = 125% of 1200) 🔶
├── scaled_90/   (90% = 1800 veh/hr = 150% of 1200)
├── manifest_scale50.txt   → scaled_30/
├── manifest_scale70.txt   → scaled_42/
├── manifest_scale100.txt  → scaled_60/
├── manifest_scale125.txt  → scaled_75/
└── manifest_scale150.txt  → scaled_90/
```

### Evaluation Strategy

| Level | Demand | Metrics | Purpose |
|-------|--------|---------|---------|
| **Primary** ⭐ | 1200 | Delay, throughput, queue | Compare with literature |
| **Deadlock** 🔶 | 1500 | Teleport count, completion rate | **Main contribution** |
| Extreme | 1800 | Survival metrics | Stress limit |

### So sánh với Literature

| Source | Demand Level | Focus |
|--------|--------------|-------|
| CoLight/FRAP | 300-700 | Standard traffic optimization |
| PressLight | 300-700 | Max pressure, throughput |
| **This project** | **600-1800** | **Deadlock handling** 🔶 |

---

## 6. Các Lệnh Mới

### Generate scaled routes
```bash
python scripts/scale_demand_batch.py --input-dir networks/variants/train --scales 0.25 0.5 0.75 1.0 1.5 --sample 50
```

### Collect normalization từ all phases
```bash
python scripts/collect_norm_curriculum.py --config configs/train_1.yaml --episodes-per-phase 20 --out configs/norm_curriculum.json --fixed-action-id 12
```

### Collect normalization từ single phase
```bash
python scripts/collect_norm_stats.py --config configs/train_1.yaml --episodes 100 --out configs/norm_1.json --manifest networks/variants/train/manifest_scale50.txt
```

---

## 7. Files Đã Xóa

| File/Directory | Lý do |
|----------------|-------|
| `scaled_50/, scaled_75/, scaled_150/, scaled_200/` (old) | Được thay thế bằng scaled routes mới với demand levels đúng |
| `*_scaled_seed*` files | Legacy randomized routes không còn dùng |
| `manifest_scale*.txt` (old) | Được regenerate với paths đúng |

---

## 8. Existing Features Supporting Extended Objectives

### Reward Components cho Traditional Objectives
| Feature | Objective | Implementation |
|---------|-----------|----------------|
| Queue-based reward | Minimize delay | `mdp_metrics.py`: distinct_cycle queue counting |
| Waiting time penalty | Reduce travel time | `mdp_metrics.py`: waiting_total() |
| Throughput tracking | Maximize flow | `kpi.py`: vehicles_completed |

### Reward Components cho Deadlock Handling (Extension)
| Feature | Purpose | Config Parameter |
|---------|---------|------------------|
| `teleport_penalty_lambda` | Penalize vehicles stuck in gridlock | `train_1.yaml`: penalty weight |
| `deadlock_penalty` | Early warning when approaching deadlock | `sumo_env.py`: deadlock detection |
| `spillback_penalty` | Prevent queue overflow → avoid gridlock | `sumo_env.py`: spillback detection |
| `anti_flicker_penalty` | Stabilize signal changes under stress | `sumo_env.py`: action consistency |

### Curriculum Phases Supporting Both Objectives

| Phase | Demand | Traditional Focus | Deadlock Focus |
|-------|--------|-------------------|----------------|
| 1 (25%) | 500 | Learn basic coordination | No deadlock - baseline |
| 2 (50%) | 1000 | Optimize delay/throughput | Rare deadlock - prevention |
| 3 (75%) | 1500 | Balance competing flows | Occasional deadlock - early detection |
| 4 (100%) | 2000 | Eval-level performance | Frequent deadlock - active handling |
| 5 (150%) | 3000 | Stress test limits | Severe deadlock - recovery skills |

---

## 9. Next Steps

1. **Chạy normalization (100 episodes theo tỉ lệ curriculum):**
   ```bash
   python scripts/collect_norm_curriculum.py --config configs/train_1.yaml --total-episodes 100 --out configs/norm_curriculum.json --fixed-action-id 12
   ```

2. ✅ **Curriculum đã tích hợp vào train.py**
   - Tự động switch manifests between phases
   - Checkpoint saved at each phase transition
   - CSV logs include `phase_name` and `phase_episode` columns

3. **Run training với 1000 episodes theo curriculum:**
   ```bash
   python scripts/train.py --config configs/train_1.yaml
   ```

---

## 9. Technical Notes

### Demand Baseline Rationale
- **Original:** 2000 veh/hr/lane (gây gridlock)
- **Option A chosen:** Giữ base 2000, scale xuống (25%-150%)
- **Reason:** Không có SUMO tools để generate routes mới

### Normalization Strategy
- Collect từ TẤT CẢ phases để statistics đại diện cho full demand range
- 20 episodes × 5 phases = 100 episodes total
- Random route selection từ 50 routes mỗi phase

### TraCI Crash Handling
- Added "socket reset" detection
- Guard check trước mỗi simulation step
- Graceful episode termination khi SUMO crash
