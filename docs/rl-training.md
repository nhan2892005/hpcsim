# Tài liệu RL Training — Green-Aware HPC Scheduling

## Tổng quan

HPCSim cung cấp hai thuật toán RL cho bài toán lập lịch HPC xanh (green-aware):

| Thuật toán | Đặc điểm | Phù hợp với |
|---|---|---|
| **MaskablePPO** | PPO + action masking | Cluster thuần GPU hoặc có CPU/MIG |
| **GAS-MARL** | Two-head (job + delay), green-aware | Workload hỗn hợp, delay cần thiết |

Cả hai đều được mở rộng để hỗ trợ:
- GPU jobs (physical GPU allocation)
- CPU-only jobs (data prep, pre/post-processing, parameter servers)
- MIG jobs (A100/H100 MIG slice, isolated compute + memory partition)
- Hybrid CPU+GPU jobs (DataLoader workers + GPU training đồng thời)

---

## State Space (121 × 12 = 1452 features)

```
Observation tensor: shape (121, 12) → flatten → (1452,)

┌────────────────────────────────────────────────────────────┐
│  Row 0..63   — Queue (MAX_QUEUE_SIZE = 64 job slots)       │
│  Row 64..95  — Running (RUN_WIN = 32 job slots)            │
│  Row 96..119 — Green forecast (GREEN_WIN = 24 time slots)  │
│  Row 120     — Cluster state (CLUSTER_WIN = 1)             │
└────────────────────────────────────────────────────────────┘
```

### Job Feature Vector (dim 12, thống nhất cho mọi loại job)

| Index | Feature | Mô tả |
|-------|---------|-------|
| 0 | `wait_time / MAX_WAIT_SEC` | Thời gian đã chờ (chuẩn hóa) |
| 1 | `n_gpu / MAX_GPUS` | Số GPU yêu cầu |
| 2 | `req_runtime / MAX_RUNTIME_SEC` | Ước tính thời gian chạy |
| 3 | `power / MAX_POWER_W` | Công suất tổng ước tính |
| 4 | `power_per_unit` | Công suất trên mỗi đơn vị tài nguyên |
| 5 | `uses_brown` | 1 nếu cần năng lượng nâu (0/1) |
| 6 | `brown_ratio` | Tỉ lệ năng lượng nâu ước tính |
| 7 | `schedulable` | 1 nếu **tất cả** tài nguyên hiện khả dụng |
| 8 | `n_cpu / MAX_CPUS` | Số CPU cores yêu cầu |
| 9 | `resource_type` | 0=GPU, 0.25=MIG, 0.5=CPU, 0.75=Hybrid |
| 10 | `n_mig / MAX_MIGS` | Số MIG slices yêu cầu |
| 11 | `cpu_schedulable` | 1 nếu CPU cores đủ (0/1) |

> **Tại sao 12 chiều thống nhất?**  
> Dùng chung một kích thước cho queue / running / green / cluster row giúp chia sẻ trọng số của `_RowEncoder` — giảm tham số, cải thiện transfer learning khi chuyển cluster.

### Cluster State Row (row 120)

| Index | Feature |
|-------|---------|
| 0 | GPU tự do / tổng GPU |
| 1 | CPU cores tự do / tổng cores |
| 2 | MIG slices tự do / tổng slices |
| 3 | Công suất đang chạy (chuẩn hóa) |
| 4 | Time-of-day (0..1, chu kỳ 24h) |
| 5 | Queue length / MAX_QUEUE_SIZE |
| 6 | Running count / RUN_WIN |
| 7 | Tỉ lệ năng lượng xanh hiện tại |
| 8..10 | Tổng CPU/MIG/GPU của cluster |

---

## Action Space

### MaskablePPO
- **Action**: `job_idx ∈ [0, MAX_QUEUE_SIZE)` — chọn 1 job từ queue
- **Mask**: `1` nếu job schedulable (tất cả tài nguyên khả dụng), `0` nếu không  
- Job CPU-only, MIG, và Hybrid đều xuất hiện trong queue; mask tự động phân biệt

### GAS-MARL
- **Action 1**: `job_idx` (giống MaskablePPO)  
- **Action 2**: `delay_action ∈ [0, ACTION2_NUM=13)`

| Delay action | Ý nghĩa |
|---|---|
| 0 | Lập lịch ngay lập tức |
| 1–5 | Chờ cho đến khi N job đang chạy hoàn thành |
| 6–12 | Trì hoãn theo thời gian: 300s, 600s, ..., 3600s |

> **Delay đặc biệt hữu ích với CPU/MIG jobs**: khi GPU đang bận nhưng CPU cores sắp giải phóng, agent có thể delay GPU job và ưu tiên CPU job trước.

---

## Reward

```
R = ReUtil − η × AvgBSLD    (sparse, cuối episode)

ReUtil  = ∫ P_green(t) dt / ∫ P_total(t) dt   (renewable utilization)
AvgBSLD = mean[(wait + exec) / max(τ, estimated_runtime)]
η       = trade-off factor (default: 0.002)
```

Với cluster CPU+GPU, `P_total(t)` bao gồm cả công suất GPU lẫn CPU.

---

## Kiến trúc Mạng

### MaskablePPO

```
obs (121×12)
    │
    ▼  _RowEncoder (shared weights, 12→128)
    │  Applied to tất cả 121 rows
    │
    ├─► Queue rows (0..63) ──► context_expand ──► job_head → logits [B, 64]
    │
    └─► Non-queue rows (64..120) ──► _SectionAggregator (mean+max pool) → context [B, 64]
                                          └──► value_head → V(s) [B, 1]
```

### GAS-MARL

```
obs (121×12)
    │
    ├─► _RowEncoder → job_enc [B,64,128] + ctx_agg → context [B,64]
    │
    ├─► job_head(job_enc, context) → job_logits [B,64]     ← Head 1
    │
    └─► sel_proj(selected_job_feat) + context → delay_head → delay_logits [B,13]  ← Head 2
```

**`_SectionAggregator`** dùng mean-pooling + max-pooling → concat → Linear để tóm tắt mỗi section thành vector cố định, giữ được thông tin cả global (mean) lẫn extreme (max).

---

## Training

### Cài đặt và chạy training

```bash
# Cài dependencies
uv sync --extra rl

# Training MaskablePPO (cluster mặc định: hpc_realistic)
uv run hpcsim train --algo maskable_ppo --epochs 300 --traj 200

# Training GAS-MARL
uv run hpcsim train --algo gas_marl --epochs 300 --traj 200

# Training cả hai cùng lúc
uv run hpcsim train --algo all --epochs 500 --traj 200

# Training với cluster có MIG (A100)
uv run hpcsim train --algo all --cluster a100_mig_cluster --epochs 300
```

### Cấu hình quan trọng

```bash
# Điều chỉnh trade-off xanh vs latency
uv run hpcsim train --algo all --eta 0.001   # ưu tiên năng lượng xanh hơn
uv run hpcsim train --algo all --eta 0.01    # ưu tiên giảm latency hơn

# Tiếp tục training từ checkpoint
uv run hpcsim train --algo maskable_ppo --resume auto  # checkpoint mới nhất
uv run hpcsim train --algo maskable_ppo --resume models/maskable_ppo/checkpoints/epoch_0200

# Training nhanh để kiểm tra (5 phút)
uv run hpcsim train --algo maskable_ppo --epochs 50 --traj 50 --cluster tiny_test
```

---

## Loại Job và Scheduler

### Cách scheduler RL xử lý từng loại tài nguyên

**CPU-only job** (`CPUJob`):
- `resource_type = ResourceType.CPU`
- Feature `schedulable` = 1 khi và chỉ khi `num_cpus_requested ≤ avail_cpus`
- Feature `n_cpu` mang giá trị > 0, `n_gpu = 0`
- Scheduler RL học ưu tiên CPU jobs khi GPU bận để tránh lãng phí CPU cores

**MIG job** (`MIGJob`):
- `resource_type = ResourceType.MIG`  
- `schedulable` = 1 khi `num_mig_requested ≤ avail_mig_slices`
- Hữu ích cho inference nhỏ — 1 A100 = 7 × `1g.10gb` slices chạy song song
- Feature `resource_type = 0.25` phân biệt với GPU jobs

**Hybrid job** (`HybridJob`):
- `resource_type = ResourceType.CPU_GPU`
- `schedulable` = 1 **chỉ khi cả GPU lẫn CPU đều đủ**
- Feature `cpu_schedulable` (index 11) báo hiệu riêng trạng thái CPU
- Agent học điều phối: delay hybrid job khi CPU đang bận, CPU jobs khi GPU bận

### Action masking với CPU/MIG

```python
# env.action_mask1() — mask [MAX_QUEUE_SIZE=64]
# 1 = job có thể schedule ngay, 0 = không đủ tài nguyên
#
# Với CPUJob: mask=1 khi num_cpus ≤ avail_cpus  (không cần GPU)
# Với MIGJob: mask=1 khi num_mig ≤ avail_mig
# Với HybridJob: mask=1 khi CẢ GPU lẫn CPU đủ
# Với TrainingJob: mask=1 khi num_gpus ≤ avail_gpus
```

---

## Đánh giá

```bash
# Đánh giá model đã train
uv run hpcsim eval --model-dir models/ --episodes 20 --output-csv eval.csv

# So sánh RL vs classical schedulers
uv run hpcsim compare \
    --classical fifo,sjf,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --runs 5 \
    --output-csv compare.csv \
    --plot
```

### Metric output

```
Scheduler       AvgJCT(s)  AvgBSLD  GPUUtil(%)  CPUUtil(%)  ReUtil(%)  Energy(kWh)
fifo            1842.3     14.2     73.1        41.2        61.3       12.4
gavel           1623.1     11.8     79.4        43.7        63.1       11.9
maskable_ppo    1501.2     10.1     82.3        52.1        71.4       10.8   ← RL
gas_marl        1489.7      9.8     83.1        54.3        73.2       10.5   ← RL
```

---

## Lưu ý khi cluster không có CPU nodes

Nếu cluster config không có CPU nodes (ví dụ `tiny_test`, `small_v100`):
- `env._total_cpus = 0`, `env._avail_cpus = 0`
- CPUJob và HybridJob sẽ không bao giờ có `schedulable = 1`
- Action mask sẽ always mask out các jobs này
- Thuật toán vẫn hoạt động bình thường — đơn giản là không schedule CPU jobs

Để tận dụng đầy đủ tính năng CPU/MIG, dùng cluster: `hpc_realistic`, `a100_mig_cluster`, `cloud_mixed`, `h100_supercluster`.

