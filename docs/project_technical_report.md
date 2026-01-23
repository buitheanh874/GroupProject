# BÁO CÁO KỸ THUẬT: HỆ THỐNG ĐIỀU KHIỂN ĐÈN TÍN HIỆU GIAO THÔNG ĐA TÁC TỬ

> **Cấu hình tham chiếu:** `configs/train_500ep.yaml`, `configs/eval.yaml`  
> **Phương pháp huấn luyện:** Sequential Curriculum Learning (500 episodes)

---

## 1. Thiết Kế Hệ Thống (System Design)

### 1.1 Kiến Trúc Mạng Lưới (Network Topology)

**Mô hình:** Mạng lưới gồm **9 giao lộ điều khiển tín hiệu** (9 signalized intersections) được tổ chức theo cấu trúc Trung tâm - Vệ tinh.

#### Cấu trúc giao lộ
*   **J0 (Center Node):** Giao lộ trung tâm, đóng vai trò điều phối chính và là điểm nút quan trọng nhất trong mạng lưới.
*   **Vệ tinh (Satellite Nodes):** 8 giao lộ bao quanh (J1, J2, J3, J4, J6, J7, J14, J17).

#### Cơ chế quan sát đặc biệt (Spillback Prevention)
Chỉ có giao lộ **J0** được cấu hình để quan sát `downstream_occupancy` (độ chiếm dụng hạ lưu) tại 4 hướng ra:
- **Hướng Bắc (N):** Link E3 → J3
- **Hướng Đông (E):** Link E0 → J1
- **Hướng Nam (S):** Link E2 → J4
- **Hướng Tây (W):** Link E1 → J2

**Mục đích kỹ thuật:** Phát hiện sớm hiện tượng **ùn ứ ngược (Spillback)** từ các giao lộ vệ tinh. Khi phát hiện vệ tinh bị tắc nghẽn, J0 sẽ hạn chế xả xe vào hướng đó để tránh gây **Gridlock** (khóa cứng toàn mạng lưới).

---

### 1.2 Thiết Kế MDP (Markov Decision Process)

Hệ thống sử dụng **Multi-Agent Reinforcement Learning (MARL)** với cơ chế phối hợp toàn cục (Global Coordination).

#### **A. Không Gian Trạng Thái (State Space) - 14 Chiều**

Mỗi agent quan sát một vector 14 chiều, được thiết kế để đảm bảo tính quan sát đầy đủ (Fully Observable MDP):

| Chỉ số | Tên Biến | Mô Tả | Nguồn Dữ Liệu | Chuẩn Hóa |
|--------|----------|-------|----------------|-----------|
| 0-3 | `q_N, q_E, q_S, q_W` | Số lượng xe xếp hàng tại 4 hướng | Local agent | Z-score |
| 4-7 | `wait_N, wait_E, wait_S, wait_W` | Thời gian chờ tích lũy (veh·s) | Local agent | Z-score |
| 8-11 | `occ_N, occ_E, occ_S, occ_W` | Độ chiếm dụng link hạ lưu [0,1] | Chỉ J0* | Z-score |
| 12 | `n_present_norm` | Mật độ xe toàn mạng (chuẩn hóa) | **Global broadcast** | Min-max |
| 13 | `spill_scalar_norm` | Chỉ số phạt tắc nghẽn toàn cục | **Global broadcast** | Min-max |

> **Lưu ý:** \*Các giao lộ vệ tinh (J1-J4, J6-J7, J14, J17) nhận giá trị 0 cho chiều 8-11.

**Ý nghĩa kỹ thuật của Global Broadcast (dim 12-13):**  
Hai chiều này được gửi đến **tất cả** 9 agent để đảm bảo:
1. **Tính Markov:** Phần thưởng phụ thuộc vào biến toàn cục → cần broadcast để MDP đầy đủ quan sát.
2. **Phối hợp ngầm (Implicit Coordination):** Các giao lộ vệ tinh có thể "cảm nhận" được áp lực toàn hệ thống và tự điều chỉnh (ví dụ: giữ xe lại khi `spill_scalar_norm` cao).

#### **B. Không Gian Hành Động (Action Space)**

**Tổng số hành động:** 15 (discrete actions)  
**Cấu trúc:** Mỗi hành động là một cặp `(cycle_length, split_ratio)`.

| Thành Phần | Tùy Chọn | Mô Tả |
|------------|----------|-------|
| **Cycle Options** | [60s, 90s, 120s] | Độ dài chu kỳ điều khiển |
| **Split Ratios** | [(0.30,0.70), (0.40,0.60), (0.50,0.50), (0.60,0.40), (0.70,0.30)] | Tỷ lệ phân chia thời gian xanh cho pha NS/EW |

**Bảng ánh xạ hành động:**

| Action ID | Cycle (s) | Split NS/EW | Action ID | Cycle (s) | Split NS/EW |
|-----------|-----------|-------------|-----------|-----------|-------------|
| 0 | 60 | 0.30/0.70 | 8 | 90 | 0.60/0.40 |
| 1 | 60 | 0.40/0.60 | 9 | 90 | 0.70/0.30 |
| 2 | 60 | 0.50/0.50 | 10 | 120 | 0.30/0.70 |
| 3 | 60 | 0.60/0.40 | 11 | 120 | 0.40/0.60 |
| 4 | 60 | 0.70/0.30 | 12 | 120 | 0.50/0.50 |
| 5 | 90 | 0.30/0.70 | 13 | 120 | 0.60/0.40 |
| 6 | 90 | 0.40/0.60 | 14 | 120 | 0.70/0.30 |
| 7 | 90 | 0.50/0.50 | | | |

**Ràng buộc kỹ thuật:**
- `g_min_sec = 10s`: Thời gian xanh tối thiểu (pedestrian safety)
- `yellow_sec = 3s`: Thời gian vàng (MUTCD standard)
- `all_red_sec = 2s`: Thời gian đỏ toàn phần (clearance interval)

#### **C. Hàm Phần Thưởng (Reward Function)**

**Công thức SMDP (Semi-Markov Decision Process):**

$$R = -\frac{W_{\text{total}}}{N \cdot t_{\text{ref}}} - \frac{\alpha \sum_{d} \text{Occ}_d^2}{M} \cdot \frac{\Delta t}{t_{\text{ref}}}$$

**Trong đó:**
- $W_{\text{total}}$: Tổng thời gian chờ của **toàn bộ mạng lưới** (sum across all 9 TLS)
- $N$: Số xe hiện diện trong mạng (`n_present`)
- $t_{\text{ref}} = 60s$: Thời gian chuẩn hóa
- $\alpha = 3.0$: Hệ số phạt spillback (cấu hình: `alpha_spillback`)
- $\text{Occ}_d$: Độ chiếm dụng hạ lưu tại hướng $d \in \{N,E,S,W\}$
- $M = 4$: Số hướng quan sát
- $\Delta t$: Độ dài bước quyết định (cycle + 2×yellow + 2×all_red)

**Đặc điểm kỹ thuật:**
1. **Global Shared Reward:** Tất cả 9 agent nhận **cùng một giá trị** reward → Khuyến khích hợp tác.
2. **Demand-Invariant:** Chia cho $N$ → Reward không phụ thuộc quy mô lưu lượng.
3. **Time-Exposure Normalization:** Chia cho $t_{\text{ref}}$ → Tránh "cycle hack" (chọn cycle ngắn để tăng tần suất nhận reward).
4. **Squared Spillback Penalty:** $\text{Occ}^2$ tạo gradient mượt (convex penalty) theo Varaiya 2013.

---

## 2. Phương Pháp Huấn Luyện (Training Methodology)

### 2.1 Kiến Trúc Agent (DQN Architecture)

**Thuật toán:** Double Dueling Deep Q-Network (D3QN)

**Cấu trúc mạng:**
```
Input (14 dims) 
    → FC(192) + ReLU
    → FC(192) + ReLU
    → Split:
        ├─ Value Head: FC(1)     → V(s)
        └─ Advantage Head: FC(15) → A(s,a)
    → Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

**Siêu tham số (Hyperparameters):**

| Tham Số | Giá Trị | Nguồn |
|---------|---------|-------|
| `hidden_dims` | [192, 192] | `train_500ep.yaml:177` |
| `learning_rate` | 0.0003 | Standard for from-scratch training |
| `gamma` | 0.99 | Time-aware scaling enabled |
| `batch_size` | 256 | Experience replay |
| `replay_buffer_size` | 200,000 | Transition memory |
| `target_update_freq` | 3,000 steps | Target network sync |
| `learning_starts` | 600 steps | Warmup before training |

**Exploration Schedule:**
- `eps_start = 1.0`: Bắt đầu với khám phá hoàn toàn
- `eps_end = 0.05`: Kết thúc với 5% ngẫu nhiên
- `eps_decay_steps = 50,000`: Decay tuyến tính qua 50k steps

---

### 2.2 Curriculum Learning (3 Giai Đoạn - 500 Episodes)

**Thiết kế curriculum:** Progressive Difficulty với phân phối cân bằng.

#### Giai đoạn 1: Easy Foundation (150 episodes - 30%)
- **Mục tiêu:** Học các mẫu giao thông cơ bản
- **Phân phối demand:** ~70% d500, 20% d750, 10% d1000
- **Route manifest:** `manifest_mixed_phase1.txt`
- **Horizon:** 1800s (30 phút)

#### Giai đoạn 2: Moderate Scaling (200 episodes - 40%)
- **Mục tiêu:** Học xử lý tải trung bình và cao
- **Phân phối demand:** ~20% d500, 40% d750, 40% d1000
- **Route manifest:** `manifest_mixed_phase2.txt`
- **Horizon:** 1800s

#### Giai đoạn 3: Hard Challenges (150 episodes - 30%)
- **Mục tiêu:** Tinh chỉnh trên các tình huống khó
- **Phân phối demand:** ~10% d500, 25% d750, 65% d1000
- **Route manifest:** `manifest_mixed_phase3.txt`
- **Horizon:** 1800s

**Thiết kế lý thuyết:**
- **Tránh Catastrophic Forgetting:** Vẫn giữ một phần d500, d750 ở phase 3 để model không quên cách xử lý traffic thấp.
- **Progressive Difficulty:** Tăng dần tỷ lệ d1000 (khó nhất) từ 10% → 40% → 65%.

---

### 2.3 Ablation Study: Parallel Training (Không Hiệu Quả)

**Thử nghiệm:** Chúng tôi đã thử nghiệm huấn luyện song song với 10 worker để tăng tốc độ thu thập dữ liệu.

**Cấu hình thử nghiệm:**
- **Số worker:** 10 actors chạy song song
- **Epsilon diversity:** Mỗi worker có epsilon multiplier khác nhau (0.8x - 1.25x)
- **Synchronization:** Cập nhật global model mỗi 100 bước

**Kết quả:**
- **Vấn đề 1 - Instability:** Training loss dao động mạnh do conflicts giữa các worker updates.
- **Vấn đề 2 - Communication Overhead:** Đồng bộ hóa model giữa 10 worker quá chậm trên CPU.
- **Vấn đề 3 - SUMO Port Conflicts:** Khó quản lý 10 SUMO instances đồng thời.

**Kết luận:** 
> Huấn luyện tuần tự (sequential) ổn định hơn và dễ tái tạo kết quả hơn cho nghiên cứu học thuật. Parallel training chỉ phù hợp khi có GPU cluster và infrastructure mạnh mẽ.

> **Cấu hình cuối cùng:** Chúng tôi chọn **sequential training với 500 episodes** (`train_500ep.yaml`) làm phương pháp chính.

---

## 3. Quy Trình Đánh Giá (Evaluation Protocol)

### 3.1 Ma Trận Đánh Giá (Test Matrix)

**Phương pháp:** Systematic evaluation trên không gian đa chiều.

**Các chiều đánh giá:**

| Chiều | Giá Trị | Mục Đích |
|-------|---------|----------|
| **Policies** | [fixed, max_pressure, actuated, webster, rl_full] | So sánh với baselines |
| **Demands** | [500, 750, 1000] veh/hr | Test khả năng thích ứng tải |
| **Seeds** | [42, 43, 44, ..., 51] (10 seeds) | Đảm bảo ý nghĩa thống kê |
| **Horizon** | 1500s (25 phút) | Thời gian mô phỏng |
| **Warmup** | 300s (5 phút) | Bỏ qua giai đoạn khởi động |

**Tổng số runs:** 5 policies × 3 demands × 10 seeds = **150 evaluation runs**

---

### 3.2 Các Thuật Toán Đối Chứng (Baselines)

#### **1. Fixed Time Controller**
- **Phương pháp:** Chu kỳ cố định 90s, split 50/50
- **Đặc điểm:** Không sử dụng thông tin real-time
- **Tham chiếu:** Traditional traffic engineering practice

#### **2. Max Pressure Controller**
- **Phương pháp:** Chọn split theo tỷ lệ queue pressure (Varaiya 2013)
- **Công thức:** $\rho_{NS} = \frac{q_N + q_S}{q_N + q_E + q_S + q_W}$
- **Đặc điểm:** Greedy optimization, không học
- **Tham chiếu:** Varaiya (2013), PressLight (KDD 2019)

#### **3. Actuated Controller**
- **Phương pháp:** Gap-out logic (kéo dài xanh nếu còn xe)
- **Đặc điểm:** Reactive control based on presence detection
- **Tham chiếu:** Highway Capacity Manual (HCM 2016)

#### **4. Webster Controller**
- **Phương pháp:** Công thức tối ưu chu kỳ dựa trên lưu lượng lịch sử
- **Công thức:** $C_{opt} = \frac{1.5L + 5}{1 - Y}$ (L: lost time, Y: critical ratio)
- **Tham chiếu:** Webster (1958) - Classic traffic signal timing

---

### 3.3 Các Chỉ Số Đánh Giá (KPIs)

**Chỉ số chính (Primary Metrics):**

| Metric | Định Nghĩa | Đơn Vị |
|--------|------------|--------|
| **Avg Wait Time** | Thời gian chờ trung bình mỗi xe (corrected) | giây |
| **Completion Rate** | Tỷ lệ xe hoàn thành lộ trình | % |
| **Throughput** | Số xe qua mạng lưới / giờ (corrected) | veh/hr |

**Chỉ số phụ (Secondary Metrics):**
- `avg_travel_time_corr`: Thời gian di chuyển trung bình
- `teleport_rate`: Tỷ lệ xe bị teleport (chất lượng mô phỏng)
- `avg_queue`: Chiều dài hàng đợi trung bình

**Correction Methodology:**
Tất cả metrics được "corrected" để loại bỏ xe bị teleport hoặc chưa hoàn thành:
- `*_corr = metric_value × completion_rate`

---

## 4. Workflow Vận Hành (Operational Workflows)

### 4.1 Workflow Huấn Luyện (Training Workflow)

```
┌─────────────────────────────────────────┐
│ 1. PREPARATION                          │
│  - Generate route files (80/10/10)      │
│  - Create manifest files for 3 phases   │
│  - Collect normalization statistics     │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 2. FEASIBILITY GATING                   │
│  - Run gating script on route files     │
│  - Filter out infeasible scenarios      │
│  - Ensure 95%+ completion rate          │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 3. SEQUENTIAL TRAINING (500 episodes)   │
│                                         │
│  Phase 1 (ep 0-149):   Easy routes     │
│  Phase 2 (ep 150-349): Moderate routes │
│  Phase 3 (ep 350-499): Hard routes     │
│                                         │
│  - Smoke eval every 25 episodes         │
│  - Save checkpoint every 50 episodes    │
│  - Track curriculum stats               │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 4. MODEL SELECTION                      │
│  - Select best checkpoint based on      │
│    smoke eval performance               │
│  - Validate on held-out demand          │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 5. FINAL EVALUATION                     │
│  - Run full test matrix (150 runs)      │
│  - Compare with 4 baselines             │
│  - Generate report & visualizations     │
└─────────────────────────────────────────┘
```

**Lệnh thực thi:**
```bash
# Training
python scripts/train.py --config configs/train_500ep.yaml

# Evaluation
python scripts/eval.py --config configs/eval.yaml
```

---

### 4.2 Workflow Đánh Giá (Evaluation Workflow)

```
┌─────────────────────────────────────────┐
│ INPUT                                   │
│  - Trained model: models/train_500ep/   │
│    best_model.pt                        │
│  - Eval config: configs/eval.yaml       │
│  - Route manifests per demand           │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ EVALUATION LOOP                          │
│  FOR each policy:                        │
│    FOR each demand:                      │
│      FOR each seed:                      │
│        1. Load route deterministically   │
│        2. Reset environment              │
│        3. Run episode (1500s)            │
│        4. Collect KPIs                   │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ POST-PROCESSING                         │
│  - Aggregate metrics across seeds       │
│  - Compute mean ± std                   │
│  - Statistical significance tests       │
│  - Generate comparison tables           │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ OUTPUT                                  │
│  - results/eval_results.csv             │
│  - Aggregated statistics & plots        │
└─────────────────────────────────────────┘
```

---

## 5. Tóm Tắt Đóng Góp Kỹ Thuật

### 5.1 Điểm Mạnh Của Phương Pháp
1. **State Space Design:** Kết hợp local + global broadcast → Fully observable cooperative MDP
2. **Reward Engineering:** SMDP với demand-invariant + spillback penalty
3. **Curriculum Learning:** Progressive difficulty với phân phối cân bằng
4. **Baseline Fairness:** Cùng action space và constraints

### 5.2 Hạn Chế & Hướng Phát Triển
- **Route Diversity:** Tỷ lệ 80/10/10 có thể không phản ánh đầy đủ real-world patterns
- **Scalability:** Chưa test trên mạng lưới >9 intersections
- **Hardware Dependency:** Training 500 episodes mất ~24-48h trên CPU

---

**Báo cáo được tổng hợp dựa trên:**
- `configs/train_500ep.yaml` (Sequential training configuration)
- `configs/eval.yaml` (Unified evaluation protocol)
- Codebase phiên bản ngày 22/01/2026
