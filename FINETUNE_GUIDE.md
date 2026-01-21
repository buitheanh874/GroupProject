# 📘 Hướng Dẫn Fine-tune Model BIGNET

## 🎯 Tổng Quan

Model `train_bignet_300ep_20260118_092429_best.pt` đã đạt kết quả ấn tượng sau 200 episodes training. Để cải thiện thêm, tôi đã chuẩn bị chiến lược fine-tuning với các điều chỉnh tối ưu.

---

## 📊 Phân Tích Model Hiện Tại

**Model đã train với `train_1_short.yaml`:**
- ✅ **200 episodes** với curriculum 3 phases (60+80+60)
- ✅ **Learning rate**: 0.0003
- ✅ **Epsilon**: 1.0 → 0.05 (decay over 20,000 steps)
- ✅ **Architecture**: [192, 192] hidden layers
- ✅ **Batch size**: 256

---

## 🎓 Chiến Lược Fine-tuning

### 1️⃣ **Giảm Learning Rate** (Critical!)
```yaml
# Original:  learning_rate: 0.0003
# Fine-tune: learning_rate: 0.0001  (giảm 3x)
```
**Lý do**: Model đã học được policy tốt, learning rate thấp hơn giúp tinh chỉnh nhẹ nhàng mà không phá hỏng những gì đã học.

### 2️⃣ **Giảm Exploration** (Epsilon thấp hơn)
```yaml
# Original:  eps_start: 1.0  → eps_end: 0.05 (over 20,000 steps)
# Fine-tune: eps_start: 0.15 → eps_end: 0.02 (over 5,000 steps)
```
**Lý do**: Model đã biết cách chọn action tốt rồi, chỉ cần explore 15% để tìm cải tiến nhỏ.

### 3️⃣ **Tập Trung Vào Scenarios Khó**
```yaml
curriculum:
  phases:
    - Phase 1 (20 eps): Warm-up với moderate difficulty
    - Phase 2 (50 eps): Hard scenarios (60% d1000) - TẬP TRUNG CHÍNH
    - Phase 3 (30 eps): Mixed hard để generalize
```
**Lý do**: Model đã giỏi với easy/medium cases, cần improve trên hard cases.

### 4️⃣ **Ít Episodes Hơn Nhưng Hiệu Quả**
```yaml
# Original:  200 episodes
# Fine-tune: 100 episodes (đủ để improve mà không overfit)
```
**Lý do**: Fine-tuning thường cần ít episodes hơn vì model đã có base knowledge tốt.

### 5️⃣ **Các Điều Chỉnh Khác**
- ✅ **Target update freq**: 3000 → 2000 (update nhanh hơn)
- ✅ **Learning starts**: 600 → 100 (bắt đầu học sớm hơn)
- ✅ **Smoke eval**: Mỗi 10 episodes (theo dõi sát hơn)

---

## 🚀 Cách Chạy Fine-tuning

### **Option 1: Chạy Default (Khuyến Nghị)**

```bash
python scripts/finetune.py
```

**Cài đặt mặc định:**
- Checkpoint: `models/BEST/train_bignet_300ep_20260118_092429_best.pt`
- Config: `configs/finetune_1.yaml`
- Episodes: 100
- Learning rate: 0.0001
- Epsilon: 0.15 → 0.02

**Kết quả sẽ được lưu tại:**
- Logs: `logs/finetune_bignet/`
- Models: `models/finetune_bignet/`
- Results: `results/finetune_bignet/`

---

### **Option 2: Fine-tune Ngắn Hơn (50 episodes)**

```bash
python scripts/finetune.py --episodes 50
```

**Phù hợp khi:**
- Bạn muốn thử nghiệm nhanh
- Tài nguyên hạn chế
- Chỉ cần improve nhẹ

---

### **Option 3: Learning Rate Thấp Hơn (Aggressive)**

```bash
python scripts/finetune.py --lr 0.00005 --episodes 100
```

**Phù hợp khi:**
- Model hiện tại đã rất tốt
- Bạn muốn cải thiện tinh vi
- Tránh catastrophic forgetting

---

### **Option 4: Custom Checkpoint**

```bash
python scripts/finetune.py --checkpoint models/bignet_short/your_model.pt
```

---

### **Option 5: Resume Fine-tuning**

```bash
# Nếu fine-tuning bị gián đoạn, resume từ checkpoint mới nhất
python scripts/finetune.py --checkpoint models/finetune_bignet/finetune_bignet_YYYYMMDD_HHMMSS_crash_epXX.pt --start-episode 51
```

---

## 📈 Dự Đoán Kết Quả

### **Cải Thiện Kỳ Vọng:**

| Metric | Before (Original) | After (Fine-tune) | Improvement |
|--------|------------------|-------------------|-------------|
| Avg Wait Time | Baseline | 5-10% giảm | ⬇️ Better |
| Throughput | High | 2-5% tăng | ⬆️ Better |
| Completion Rate | Good | 1-3% tăng | ⬆️ Better |
| Hard Scenarios | Medium | Significant ⬆️ | ⭐ Main Goal |

**Lưu ý:** Fine-tuning thường cải thiện **stability** và **hard cases** hơn là metrics tổng thể.

---

## ⏱️ Thời Gian Dự Kiến

- **100 episodes** với 2 parallel workers
- Mỗi episode: ~3-5 phút (tùy demand)
- **Tổng thời gian**: ~6-8 giờ

**Khuyến nghị:** Chạy overnight hoặc khi không dùng máy.

---

## 🔍 Monitoring & Evaluation

### **1. Theo Dõi Training**

```bash
# Xem metrics realtime
tail -f logs/finetune_bignet/finetune_bignet_*_train_metrics.csv

# Xem smoke evaluation results
tail -f logs/finetune_bignet/finetune_bignet_*_smoke_eval.csv
```

### **2. So Sánh Trước/Sau Fine-tuning**

```bash
# Evaluate original model
python scripts/eval.py --config configs/eval.yaml --checkpoint models/BEST/train_bignet_300ep_20260118_092429_best.pt --output eval_before_finetune.csv

# Evaluate fine-tuned model (sau khi train xong)
python scripts/eval.py --config configs/eval.yaml --checkpoint models/finetune_bignet/finetune_bignet_*_best.pt --output eval_after_finetune.csv

# So sánh kết quả
python scripts/compare_controllers.py eval_before_finetune.csv eval_after_finetune.csv
```

---

## 🎯 Khi Nào Dừng Fine-tuning?

### **Dấu Hiệu Tốt (Tiếp Tục):**
- ✅ Loss giảm dần hoặc ổn định
- ✅ Smoke eval metrics cải thiện
- ✅ Completion rate tăng trên hard scenarios
- ✅ Avg wait time giảm

### **Dấu Hiệu Xấu (Dừng Sớm):**
- ❌ Loss tăng đột ngột
- ❌ Smoke eval metrics giảm
- ❌ Model bắt đầu overfit (train tốt nhưng eval kém)
- ❌ Teleport rate tăng cao

**Action:** Nếu thấy dấu hiệu xấu, dừng và dùng checkpoint trước đó.

---

## 🔧 Troubleshooting

### **Issue 1: SUMO Connection Error**
```bash
# Giải pháp: Tăng timeout hoặc giảm num_actors
python scripts/finetune.py --episodes 100
# Sau đó sửa configs/finetune_1.yaml: num_actors: 1
```

### **Issue 2: Out of Memory**
```bash
# Giải pháp: Giảm batch size
# Sửa configs/finetune_1.yaml: batch_size: 128
```

### **Issue 3: Training Quá Chậm**
```bash
# Giải pháp: Giảm episodes hoặc tăng workers (nếu CPU/RAM đủ)
python scripts/finetune.py --episodes 50
```

---

## 📝 Checklist Trước Khi Chạy

- [ ] **Checkpoint tồn tại**: `models/BEST/train_bignet_300ep_20260118_092429_best.pt` ✅
- [ ] **Config file tồn tại**: `configs/finetune_1.yaml` ✅
- [ ] **Route manifests tồn tại**: `networks/variants/train_turn801010/manifest_mixed_phase*.txt` ✅
- [ ] **SUMO đã cài đặt**: `sumo --version` works ✅
- [ ] **Python packages**: `torch`, `numpy`, `pyyaml` installed ✅
- [ ] **Disk space**: Đủ ~2-5GB cho logs/models ⚠️ (Kiểm tra!)
- [ ] **Time available**: 6-8 giờ ⚠️ (Chạy overnight!)

---

## 🎉 Next Steps Sau Fine-tuning

1. **Evaluate thoroughly** trên test set với nhiều seeds
2. **Compare** với baselines (fixed, max_pressure, actuated)
3. **Visualize** improvements với plots
4. **Document** findings trong report
5. **Consider** ensemble hoặc multi-model approach nếu cần

---

## 💡 Tips & Best Practices

1. **Backup model gốc** trước khi fine-tune (đã có ở `models/BEST/` ✅)
2. **Monitor first 10-20 episodes** để đảm bảo training ổn định
3. **Save checkpoints thường xuyên** (config đã set `save_every_episodes: 20` ✅)
4. **Compare multiple checkpoints** không chỉ dựa vào episode_reward
5. **Test trên diverse scenarios** không chỉ training demand levels

---

## ❓ FAQ

**Q: Fine-tuning 100 episodes có đủ không?**  
A: Có! Fine-tuning thường cần ít episodes hơn initial training. 100 eps là optimal balance.

**Q: Có thể fine-tune nhiều lần không?**  
A: Có, nhưng cẩn thận overfitting. Recommended: tối đa 2-3 rounds.

**Q: Learning rate 0.0001 có quá thấp không?**  
A: Không, đây là best practice cho fine-tuning. Có thể thử 0.00005 nếu muốn an toàn hơn.

**Q: Có cần retrain từ đầu không?**  
A: Không cần! Fine-tuning sẽ improve model hiện tại. Chỉ retrain nếu architecture thay đổi.

**Q: Kết quả có tốt hơn đảm bảo không?**  
A: Không đảm bảo 100%, nhưng với hyperparameters đã optimize, xác suất cao là sẽ improve.

---

## 🚀 READY TO START!

**Lệnh chạy cuối cùng:**

```bash
python scripts/finetune.py
```

**Hoặc nếu muốn test nhanh (50 episodes):**

```bash
python scripts/finetune.py --episodes 50
```

Good luck! 🍀

---

**Created by**: Antigravity AI  
**Date**: 2026-01-21  
**Version**: 1.0
