# Independent Multi-Agent Baseline

## Tổng quan (Overview)

Configuration này tạo ra một **decentralized multi-agent system** với 9 agents hoàn toàn độc lập:
- Mỗi agent điều khiển 1 đèn giao thông
- **Không chia sẻ** Q-network parameters
- **Không chia sẻ** replay buffer
- **Không giao tiếp** với nhau
- Mỗi agent chỉ quan sát local state của đèn mình điều khiển

**Config này match với `train_bignet_short.yaml`** để so sánh công bằng:
- Same curriculum (6 phases, 300 episodes)
- Same network architecture (192×192)
- Same hyperparameters
- Same manifests

## So sánh với Centralized (Main Model)

| Aspect | Centralized (bignet_short) | Independent (this) | 
|--------|---------------------------|-------------------|
| **Agents** | 1 shared agent | 9 independent agents |
| **Q-networks** | 1 network (192×192) | 9 networks (192×192 each) |
| **Parameters** | ~74K | ~666K (9×) |
| **Replay buffers** | 1 shared (200K) | 9 separate (200K each) |
| **Communication** | Implicit (shared params) | None |
| **Training** | Coordinated | Independent |
| **Curriculum** | 6 phases, 300 eps | Same |
| **Manifests** | d400, d600, d800, d1000 | Same |

## Cách sử dụng (Usage)

### Training

```bash
# Independent multi-agent (9 separate agents, no communication)
python scripts/train_independent.py --config configs/train_independent.yaml

# Centralized (1 shared agent) - for comparison
python scripts/train.py --config configs/train_bignet_short.yaml
```

### Curriculum Phases (same as bignet_short)

| Phase | Episodes | Demand | Duration |
|-------|----------|--------|----------|
| phase1_easy | 30 | 400 veh/hr | 30min |
| phase2_moderate | 45 | 600 veh/hr | 30min |
| phase3a_hard_short | 112 | 800 veh/hr | 30min |
| phase3b_hard_long | 38 | 800 veh/hr | 60min |
| phase4a_challenge_short | 45 | 1000 veh/hr | 30min |
| phase4b_challenge_long | 30 | 1000 veh/hr | 60min |
| **Total** | **300** | | |

### Output Files

**Independent approach** tạo ra 9 checkpoint files per save:
```
models/independent/
├── <run_id>_J0_best.pt
├── <run_id>_J1_best.pt
├── <run_id>_J2_best.pt
├── <run_id>_J3_best.pt
├── <run_id>_J4_best.pt
├── <run_id>_J6_best.pt
├── <run_id>_J7_best.pt
├── <run_id>_J14_best.pt
└── <run_id>_J17_best.pt
```

## Kết quả mong đợi (Expected Results)

### Performance
- **Centralized** nên perform tốt hơn do có implicit coordination
- **Independent** có thể gặp conflicts giữa các agents

### Metrics để so sánh
1. **Avg Waiting Time**: Lower is better
2. **Throughput**: Higher is better  
3. **Avg Queue Length**: Lower is better
4. **Learning Curve**: Compare convergence speed

## Lưu ý quan trọng

> **Memory Usage**: Independent training cần ~3× memory (9 networks)

> **Training Time**: Chậm hơn centralized do 9× network updates

> **Fair Comparison**: Config này match hoàn toàn với bignet_short
