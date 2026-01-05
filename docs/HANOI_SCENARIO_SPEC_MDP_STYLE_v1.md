# 📘 HANOI TRAFFIC SCENARIO SPEC — “MDP-style” Implementation Contract (v1.0)

**Mục đích:** Tài liệu này là “hợp đồng triển khai” để một AI Developer có thể code đúng hướng làm **kịch bản giao thông ngẫu nhiên** cho SUMO, phục vụ **train/eval**.  
**Trọng tâm:** random hóa **demand ở biên** + **tỉ lệ rẽ** + **cơ cấu loại xe kiểu Hà Nội**, và đảm bảo **dòng xe ở các nút trong phụ thuộc upstream** (tự suy ra theo routing).

**Ngày:** 2026-01-04  
**Scope:** sinh dữ liệu `.rou.xml` (offline), dùng được với cơ chế `train.route_pool` đã tích hợp (random chọn route file mỗi episode).

---

## 0) Non-Goals (để tránh AI code sai hướng)

- **Không** chỉnh demand “on-the-fly” trong khi SUMO đang chạy (không cần và dễ lỗi).
- **Không** can thiệp sâu vào TraCI để thay đổi turning ratios theo thời gian trong runtime.
- **Không** yêu cầu “một bộ tỉ lệ rẽ chuẩn universal”. Tỉ lệ rẽ sẽ được **mô hình hóa bằng phân phối** (prior + randomization).
- **Không** sửa MDP RL core (state/action/reward). Chỉ tập trung sinh kịch bản + config route_pool + verify.

---

## 1) Mục tiêu kỹ thuật

### 1.1. Mục tiêu tổng quát
1) **Train**: Agent học được policy không overfit 1 route cố định, chịu được biến thiên demand/turning/vehicle mix.  
2) **Eval**: So sánh controllers công bằng trên **bộ kịch bản hold-out cố định**, đại diện lưu lượng Hà Nội.

### 1.2. Điều kiện “Hà Nội realistic” (mức tối thiểu)
- **Vehicle mix prior (mục tiêu trung bình):**
  - Xe máy (motorcycle): **0.80–0.85**
  - Ô tô con (passenger): **~0.12**
  - Bus: **~0.03**
  - Others (optional): phần còn lại
- **Turning ratios**: random theo **Dirichlet** quanh một prior (mặc định đô thị).
- **Demand theo biên**: tổng lưu lượng và phân bổ theo cửa vào được random có kiểm soát, vẫn nằm trong dải “thực tế hợp lý” theo hiệu chỉnh (calibration).

> Lưu ý: “mật độ/lưu lượng thực tế mới nhất” thường khó có dataset mở đầy đủ. Spec này yêu cầu **calibration file** (YAML/JSON) để bạn cập nhật các dải lưu lượng theo khảo sát bạn có (đầu vào do nhóm dự án cung cấp).

---

## 2) Định nghĩa & Ký hiệu

### 2.1. Biên mạng (Boundary)
- **Entry edges**: các edge (hoặc lane) tại vành ngoài nơi xe **đi vào** mạng.
- **Exit edges**: các edge tại vành ngoài nơi xe **đi ra** mạng.

### 2.2. Turning movement (L/S/R)
- Với một **incoming edge** tại một junction, tập **outgoing edges** được phân loại thành:
  - **L** (Left), **S** (Straight), **R** (Right) theo góc hướng hình học.
- Nếu topology không rõ, cho phép mapping thủ công bằng config.

### 2.3. Flow và PCU
- `veh/h`: xe/giờ theo loại xe.
- `pcu/h`: quy đổi theo **hệ số PCU** (Passenger Car Unit) theo loại xe.

---

## 3) Thiết kế kịch bản (Scenario Model)

### 3.1. Randomization Layer 1 — Demand ở biên (entry split)
Mỗi scenario có tổng demand `Λ` (pcu/h) và phân bổ theo K cửa vào:

- Sample vector tỷ trọng cửa vào:
  - `π ~ Dirichlet(α_entry)` (K chiều)
- Lưu lượng tại cửa vào i:
  - `Q_i = Λ * π_i` (pcu/h)

**Control knobs:**
- `α_entry` lớn → phân bổ đều; nhỏ → lệch mạnh (một vài cửa bùng lớn).
- `Λ` lấy theo *level* (low/med/high) hoặc theo *stage intervals*.

### 3.2. Randomization Layer 2 — Turning ratios per approach (L/S/R)
Với mỗi junction (ưu tiên junction biên / outer TLS), với mỗi incoming approach:

- Prior turning mean `μ = [μ_L, μ_S, μ_R]`, tổng = 1
- Sample:
  - `θ ~ Dirichlet(κ_turn * μ)`

**Control knobs:**
- `κ_turn` lớn → ít dao động; nhỏ → dao động mạnh.

> Quy ước: turning ratios được gán theo **incoming edge** (approach-level), không theo lane-level (trừ khi bạn muốn mở rộng).

### 3.3. Randomization Layer 3 — Vehicle type mix (Hà Nội)
Mỗi scenario (hoặc mỗi entry edge), sample vehicle mix:

- Prior mean `ν = [ν_mc, ν_car, ν_bus, ν_other]`
- Sample:
  - `v ~ Dirichlet(κ_vehicle * ν)`

Sau đó, convert `Q_i` (pcu/h) sang số xe/giờ từng loại theo `v` và PCU weights.

---

## 4) Route generation strategy (Offline)

### 4.1. Output artifacts
Mỗi scenario seed tạo ra:
- `*.rou.xml` (SUMO route file) **final** dùng để train/eval.
- Optional debug:
  - `flows_*.xml` (flows input)
  - `turns_*.xml` (turning ratios input)
  - `meta_*.json` (seed + parameters + summary)

### 4.2. Generator toolchain (khuyến nghị)
Ưu tiên pipeline “flows + turns → routes” bằng SUMO routing tools. Hai lựa chọn:

**Option A (khuyến nghị nếu phù hợp):** `jtrrouter`  
- Input: net, flows, turns
- Output: routes `.rou.xml`

**Option B:** tự sinh routes (ngẫu nhiên path theo shortest path / k-shortest) nếu jtrrouter không đáp ứng.

Spec này chọn **Option A** làm default; Option B chỉ là fallback.

### 4.3. Seed & reproducibility
- Mọi scenario được xác định bởi `seed` và `calibration config`.
- Train set và Eval set **không trùng seed**.
- File names phải chứa seed: `BIGMAP_train_seed00042.rou.xml`.

---

## 5) Calibration contract (bắt buộc có 1 file)

### 5.1. File: `configs/scenario_hanoi_calibration.yaml`
Nội dung tối thiểu:

```yaml
scenario:
  net_file: networks/BIGMAP.net.xml

  # Boundary definition
  entry_edges: ["E_in_0", "E_in_1", "..."]
  exit_edges: ["E_out_0", "E_out_1", "..."]

  # Vehicle mix prior (Hanoi-like)
  vehicle_mix_mean:
    motorcycle: 0.84
    passenger: 0.12
    bus: 0.03
    other: 0.01
  vehicle_mix_kappa: 50   # higher = less random

  # PCU weights (must align with env usage if PCU-weighted reward is enabled)
  pcu_weights:
    motorcycle: 0.25
    passenger: 1.0
    bus: 3.0
    other: 1.0

  # Demand levels in PCU/h for each entry edge class
  demand:
    total_pcu_per_hour:
      low: 3000
      med: 5000
      high: 7000
    entry_dirichlet_alpha: 3.0

  # Turning prior (global default)
  turning:
    mean_LSR: [0.15, 0.70, 0.15]
    kappa: 30

  # Optional: override turning prior per junction or per approach edge
  turning_overrides:
    # junction_id:
    #   incoming_edge_id: [pL, pS, pR]
    {}

  # Optional: multi-stage intervals (seconds)
  stages:
    enabled: false
    intervals:
      # - {begin: 0, end: 1920, level: low}
      # - {begin: 1920, end: 3840, level: high}
      []
```

**AI Developer bắt buộc**: nếu thiếu `entry_edges/exit_edges`, generator phải fail với thông báo rõ ràng.

---

## 6) Files & API to implement

### 6.1. New script: `scripts/generate_hanoi_route_variants.py`
**CLI contract:**
```bash
python scripts/generate_hanoi_route_variants.py \
  --calib configs/scenario_hanoi_calibration.yaml \
  --out-dir networks/variants \
  --split train \
  --n 100 \
  --seed 42
```

**Arguments:**
- `--calib`: path calibration yaml
- `--out-dir`: output directory
- `--split`: `train` | `eval`
- `--n`: number of variants
- `--seed`: base seed

**Outputs:**
- `networks/variants/train/BIGMAP_train_seedXXXXX.rou.xml`
- `networks/variants/eval/BIGMAP_eval_seedXXXXX.rou.xml`
- optional debug artifacts under `networks/variants/_debug/...`

### 6.2. Optional helper: `scripts/inspect_net_boundaries.py`
- Auto-suggest entry/exit edges by scanning net topology (degree, incoming/outgoing).
- Output candidate lists (human confirms into calibration file).

### 6.3. Config integration
Update train/eval configs to point `train.route_pool` to generated files.

Example:
```yaml
train:
  route_pool:
    - networks/variants/train/BIGMAP_train_seed00001.rou.xml
    - networks/variants/train/BIGMAP_train_seed00002.rou.xml
```

> Do not rely on glob expansion inside YAML (nếu code loader không hỗ trợ). Nếu muốn glob, implement expansion in `train.py` hoặc in generator produce a manifest file.

---

## 7) Algorithmic details (pseudo-code)

### 7.1. Sampling scenario parameters
```python
rng = Random(seed_i)

# 1) total demand level
Lambda = sample_total_pcu(level=low/med/high, rng=rng)

# 2) entry split
pi = dirichlet(alpha_entry, K, rng)
Q_entry = Lambda * pi  # pcu/h per entry edge

# 3) vehicle mix
v_mix = dirichlet(kappa_vehicle * nu_mean, rng)

# 4) turning ratios for each (junction, incoming edge)
for each junction in target_junctions:
  for each incoming_edge:
    mu = override if present else global_mean_LSR
    theta = dirichlet(kappa_turn * mu, rng)
```

### 7.2. Convert PCU/h -> veh/h by type
Given pcu weights `w_type` and mix `v_type`:
- Choose a base `veh/h` so that sum(veh_type * w_type) = Q_pcu
- Simplest:
  1) allocate “pcu share” by mix: `pcu_type = Q_pcu * v_type`
  2) convert: `veh_type = pcu_type / w_type`

Round carefully (keep total close; allow fractional flows if SUMO supports).

### 7.3. Write SUMO inputs
- Create `flows.xml`:
  - flows per entry edge, per vehicle type, per interval
- Create `turns.xml`:
  - turn probability (incoming edge -> outgoing edge) for each junction

Then run routing tool to produce `*.rou.xml`.

---

## 8) Train/Eval protocol (must follow)

### 8.1. Split policy
- `train`: seeds in range `[seed, seed+n_train-1]`
- `eval`: seeds in disjoint range `[seed+100000, seed+100000+n_eval-1]` (or any offset policy)
- Never overlap.

### 8.2. Stability constraints
- Each route file must have at least `min_total_vehicles` (configurable) to avoid empty episodes.
- If `terminate_on_empty=true`, ensure flows produce vehicles across most of horizon.

### 8.3. Metrics & logging
- Store per-variant metadata (`meta_*.json`):
  - seed
  - total_pcu/h
  - entry pcu/h vector
  - turning ratios summary
  - vehicle mix
- This enables analysis scripts to correlate performance vs scenario properties.

---

## 9) Verification & tests (bắt buộc)

### 9.1. Script test: `scripts/test_route_pool.py`
- Run training for 10 episodes with route_pool size >= 3
- Assert log shows >= 2 distinct route files were selected

### 9.2. Generator tests: `scripts/test_scenario_generator.py`
For each generated variant:
- Validate XML well-formed
- Validate total demand within tolerance of configured Λ
- Validate vehicle mix within configured range (or record actual sampled mix)
- Validate turning probabilities sum to 1 per incoming edge (within tolerance)
- Validate route file loads in SUMO (smoke run 30–60s)

### 9.3. Reproducibility test
- Generate variants twice with same seed and assert file hashes match (or metadata match).

---

## 10) Acceptance criteria (Definition of Done)

1) `scripts/generate_hanoi_route_variants.py` generates N valid `.rou.xml` files for train and eval.
2) Train config uses `train.route_pool` with >= 2 route files and training logs show different selected routes across episodes.
3) Eval runs over hold-out route set and produces KPI CSVs for comparison.
4) Metadata exists and enables report plots (optional but recommended).

---

## 11) Implementation notes (tránh lỗi thường gặp)

- **Edge vs lane IDs**: calibration dùng edge IDs hay lane IDs phải thống nhất; nếu env expects lane IDs in lane_groups, generator vẫn có thể use edge-level flows (SUMO will assign lanes).  
- **Turning mapping**: nếu tự động phân loại L/S/R theo góc, phải xử lý trường hợp đa nhánh và U-turn (khuyến nghị: disable U-turn unless explicitly allowed).
- **Manifest file** (khuyến nghị): generator output `networks/variants/train_manifest.txt` (list file paths). Config loader có thể đọc list này để tránh YAML dài.
- **Performance**: generating 200 variants có thể tốn thời gian; cache intermediate files và chạy theo seed range.

---

## 12) Minimal example commands

```bash
# 1) Fill calibration (manually or via inspect script)
python scripts/inspect_net_boundaries.py --net networks/BIGMAP.net.xml --out configs/boundary_suggestion.json

# 2) Generate train/eval variants
python scripts/generate_hanoi_route_variants.py --calib configs/scenario_hanoi_calibration.yaml --out-dir networks/variants --split train --n 100 --seed 42
python scripts/generate_hanoi_route_variants.py --calib configs/scenario_hanoi_calibration.yaml --out-dir networks/variants --split eval --n 30 --seed 42

# 3) Update configs to include route_pool (train) and eval route list

# 4) Verify route pool selection during training
python scripts/train.py --config configs/train_sumo.yaml --episodes 10

# 5) Evaluate controllers on eval set (project-specific)
python scripts/eval.py --config configs/eval_sumo.yaml --controller all --runs 30
```

---

**Phiên bản:** 1.0  
**Tác giả:** ChatGPT (implementation contract for AI coder)
