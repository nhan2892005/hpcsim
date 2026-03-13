# Tài liệu RL Training — Green-Aware HPC Scheduling

## Tổng quan

HPCSim cung cấp hai thuật toán RL cho bài toán lập lịch HPC xanh (green-aware):

| Thuật toán | Đặc điểm | Phù hợp với |
|---|---|---|
| **MaskablePPO** | PPO + action masking | Cluster thuần GPU hoặc có CPU/MIG |
| **GAS-MARL** | Two-head (job + delay), green-aware | Workload hỗn hợp, delay cần thiết |

Cả hai đều hỗ trợ: GPU jobs, CPU-only jobs, MIG jobs, Hybrid CPU+GPU jobs.

Từ v0.3, cả hai thuật toán có thể kết hợp với **backfill policy** qua `BackfillWrapper` — một lớp độc lập hoạt động *ngoài* RL agent, lấp khoảng trống tài nguyên khi job bị blocked hoặc delayed.

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
> Dùng chung kích thước cho queue / running / green / cluster row giúp chia sẻ trọng số của `_RowEncoder` — giảm tham số, cải thiện transfer learning khi chuyển cluster.

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
- CPU-only, MIG, và Hybrid jobs đều xuất hiện trong queue; mask tự động phân biệt

### GAS-MARL
- **Action 1**: `job_idx` (giống MaskablePPO)
- **Action 2**: `delay_action ∈ [0, ACTION2_NUM=13)`

| Delay action | Ý nghĩa |
|---|---|
| 0 | Lập lịch ngay lập tức |
| 1–5 | Chờ cho đến khi N jobs đang chạy hoàn thành |
| 6–12 | Trì hoãn theo thời gian: 300s, 600s, ..., 3600s |

Khi GAS-MARL chọn delay (ac2 > 0), scheduler **không tự backfill** — nó ghi lại metadata `delay_info` vào `SchedulingDecision` để `BackfillWrapper` đọc và thực hiện backfilling theo đúng thuật toán §4.3 của paper:

```python
# GASMARLScheduler.schedule() sets:
decision.delay_info = {
    "delay_type":    ac2,              # 1-5 = wait N jobs, 6-12 = wait DT
    "release_time":  release_t,        # timestamp khi head job được giải phóng
    "head_job_id":   selected_job.job_id,
    "head_req_gpus": n_gpus_needed,
}
# BackfillWrapper đọc delay_info và tính max_finish_time:
#   type 1-5: max(ts_release, ts_shadow)  capped at current_time + 3600
#   type 6-12: max(ts_shadow, current_time + DT)
```

> **Delay đặc biệt hữu ích với CPU/MIG jobs**: khi GPU đang bận nhưng CPU cores
> sắp giải phóng, agent có thể delay GPU job và ưu tiên CPU job trước.

---

## Reward

```
R = ReUtil − η × AvgBSLD    (sparse, cuối episode)

ReUtil  = ∫ P_green(t) dt / ∫ P_total(t) dt
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
    ├─► Queue rows (0..63) + context_expand → job_head → logits [B, 64]
    │
    └─► Non-queue rows (64..120) → _SectionAggregator → context [B,64]
                                         └─► value_head → V(s) [B, 1]
```

### GAS-MARL

```
obs (121×12)
    │
    ├─► _RowEncoder → job_enc [B,64,128] + ctx_agg → context [B,64]
    │
    ├─► job_head(job_enc, context) → job_logits [B,64]            ← Head 1
    │
    └─► sel_proj(selected_job_feat) + context
            └─► delay_head → delay_logits [B,13]                  ← Head 2
```

---

## Backfilling với RL

### Tổng quan kiến trúc

Backfilling trong HPCSim **không phải** logic bên trong RL agent — đó là một `BackfillPolicy` bọc ngoài qua `BackfillWrapper`. Điều này đảm bảo:

1. **Tính độc lập**: RL agent không cần biết backfill tồn tại. Agent học policy tốt nhất *trong điều kiện có backfill* — phần thưởng phản ánh môi trường thực tế hơn.
2. **Tính composable**: Có thể bật/tắt hoặc đổi backfill policy mà không cần retrain.
3. **Đúng theo paper**: Chen et al. (FGCS 2025) thiết kế Green-Backfilling là tầng riêng biệt, không phải part của RL policy.

```
episode step:
  1. RL agent chọn job + delay action
  2. BackfillWrapper nhận decision + delay_info
  3. BackfillPolicy tìm jobs có thể backfill trong window
  4. Environment nhận tổng hợp: job chính + backfill jobs
  5. Reward tính theo cả hai — agent gián tiếp học dùng delay
     khi biết backfill sẽ lấp gap
```

### Combinations scheduler × backfill

| Primary Scheduler | --backfill none | --backfill easy | --backfill green |
|---|---|---|---|
| FIFO | baseline | ↑ util | ↑ ReUtil, ~BSLD |
| SJF | — | ↑ util | ↑ ReUtil |
| Gavel | — | ↑ util | ↑ ReUtil |
| MaskablePPO | ← học baseline | ← học với EASY | ← học với Green |
| **GAS-MARL** | ← không tối ưu | ← trung bình | **← best combo** |

> **Luật vàng**: Nếu bạn train với backfill policy nào thì **phải eval/deploy với backfill policy đó**. Eval với policy khác sẽ cho kết quả sai lệch vì agent đã học dựa trên môi trường có backfill.

### EASY-Backfilling với RL

EASY-Backfilling phù hợp với **MaskablePPO** — scheduler không có cơ chế delay, backfill lấp gap khi head job bị blocked:

```
Trigger: head job blocked (thiếu GPU)
Logic:   duyệt queue theo submit_time
         cho phép job chạy sớm nếu finish_time ≤ shadow_time
         (shadow_time = thời điểm đủ GPU cho head job)
```

### Green-Backfilling với RL

Green-Backfilling (`GreenBackfillPolicy`) được thiết kế riêng cho **GAS-MARL** — tận dụng cơ chế delay của agent để tạo window backfill có mục tiêu năng lượng:

```
Trigger 1: head job blocked (giống EASY)
Trigger 2: GAS-MARL delay action > 0 (delay_info có trong decision)

Priority: L_j = re_j × q_j × p_j  (Eq. 15, ascending)
          → ưu tiên job ngắn, ít GPU, ít tốn điện

Filter:   brown_energy_j = max(0, P_job - P_renewable) × runtime
          chỉ chấp nhận job có brown_energy < σ = 50,000 J
```

**Lý do GAS-MARL + Green là best combo** (theo paper §5.2):
- Agent học delay job cỡ lớn khi renewable thấp → tạo window
- Green-Backfilling lấp window với job nhỏ xanh
- Cộng hưởng: cả hai cùng tối ưu ReUtil

---

## Training

### Yêu cầu

```bash
uv sync --extra rl   # cài PyTorch
uv run hpcsim test --no-rl   # test không cần PyTorch
uv run hpcsim test           # test đầy đủ bao gồm RL env
```

### Cấu hình training cơ bản

```bash
# Train không backfilling (ablation baseline)
uv run hpcsim train --algo all --epochs 300 --traj 200

# Train MaskablePPO + EASY-Backfilling
uv run hpcsim train \
    --algo maskable_ppo \
    --epochs 300 \
    --traj 200 \
    --backfill easy

# Train GAS-MARL + Green-Backfilling (cấu hình tối ưu từ paper)
uv run hpcsim train \
    --algo gas_marl \
    --epochs 300 \
    --traj 200 \
    --backfill green \
    --eta 0.002

# Train cả hai cùng lúc với Green-Backfilling
uv run hpcsim train \
    --algo all \
    --epochs 500 \
    --traj 200 \
    --backfill green
```

### Tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--algo` | `all` | `maskable_ppo` / `gas_marl` / `all` |
| `--epochs` | `300` | Số epoch training |
| `--traj` | `100` | Trajectories per epoch (tăng → ổn định hơn, chậm hơn) |
| `--backfill` | `none` | `none` / `easy` / `green` |
| `--eta` | `0.002` | Trade-off ReUtil vs BSLD trong reward |
| `--duration` | `86400` | Độ dài episode (giây) |
| `--save-dir` | `models/` | Thư mục lưu model |
| `--ckpt-interval` | `50` | Lưu checkpoint mỗi N epoch |
| `--log-interval` | `10` | In progress mỗi N epoch |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--resume` | `None` | `auto` (checkpoint mới nhất) hoặc path cụ thể |
| `--no-save-best` | `False` | Tắt lưu best-reward model |

### Điều chỉnh trade-off với --eta

```bash
# η nhỏ → ưu tiên ReUtil (năng lượng xanh)
uv run hpcsim train --algo gas_marl --eta 0.001 --backfill green

# η mặc định (cân bằng, theo paper)
uv run hpcsim train --algo gas_marl --eta 0.002 --backfill green

# η lớn → ưu tiên giảm BSLD (latency)
uv run hpcsim train --algo gas_marl --eta 0.01  --backfill green
```

Khuyến nghị khởi đầu với η = 0.002 (giá trị paper). Tăng nếu BSLD quá cao, giảm nếu muốn maximize ReUtil.

### Resume từ checkpoint

```bash
# Resume checkpoint mới nhất (tự tìm)
uv run hpcsim train --algo gas_marl --resume auto --backfill green

# Resume từ checkpoint cụ thể
uv run hpcsim train --algo maskable_ppo \
    --resume models/maskable_ppo/checkpoints/epoch_0150 \
    --backfill easy \
    --epochs 300   # tiếp tục đến epoch 300
```

> ⚠️ **Quan trọng**: Khi resume, `--backfill` phải khớp với lúc ban đầu. Thay đổi backfill policy giữa chừng sẽ làm mất tính nhất quán của policy — agent sẽ cần re-explore từ đầu.

### Ablation study

```bash
# Chạy đủ 4 điều kiện để so sánh (theo paper §5.2)
uv run hpcsim train --algo gas_marl --backfill none  --save-dir models/ablation_none/
uv run hpcsim train --algo gas_marl --backfill easy  --save-dir models/ablation_easy/
uv run hpcsim train --algo gas_marl --backfill green --save-dir models/ablation_green/
uv run hpcsim train --algo maskable_ppo --backfill green --save-dir models/ablation_ppo_green/
```

### Nghiên cứu η (trade-off sensitivity)

```bash
# η sweep với Green-Backfilling (theo paper §5.4.1)
for eta in 0.001 0.002 0.003 0.005 0.008 0.01; do
    uv run hpcsim train \
        --algo gas_marl \
        --epochs 300 \
        --backfill green \
        --eta $eta \
        --save-dir models/eta_${eta}/
done
```

---

## Output và Theo dõi Training

### Cấu trúc thư mục sau training

```
models/
├── gas_marl/
│   ├── best/            ← model tốt nhất (cao nhất avg_reward)
│   │   ├── actor.pt
│   │   └── critic.pt
│   ├── checkpoints/
│   │   ├── epoch_0050/
│   │   ├── epoch_0100/
│   │   └── epoch_0300/  ← latest checkpoint
│   └── train_log.csv    ← metrics theo epoch
└── maskable_ppo/
    ├── best/
    ├── checkpoints/
    └── train_log.csv
```

### train_log.csv — Columns

| Column | Mô tả |
|--------|-------|
| `epoch` | Epoch number |
| `avg_reward` | Mean reward over trajectories |
| `avg_green` | Mean ReUtil (renewable utilization) |
| `avg_bsld` | Mean AvgBSLD |
| `std_reward` | Std reward (stability indicator) |
| `loss_actor` | Actor loss (PPO clip loss) |
| `loss_critic` | Critic loss (value MSE) |
| `entropy` | Policy entropy (exploration) |

### Theo dõi với plot

```bash
# Plot learning curve
uv run hpcsim plot --type learning-curve \
    --input models/gas_marl/train_log.csv \
    --output lc_gasmarl.png

# So sánh hai run (none vs green backfill)
uv run hpcsim plot --type learning-curve \
    --input models/ablation_none/gas_marl/train_log.csv \
    --input2 models/ablation_green/gas_marl/train_log.csv \
    --output lc_compare.png
```

---

## Đánh giá (Evaluation)

> **Quy tắc**: Luôn eval với cùng `--backfill` như lúc train.

```bash
# Eval GAS-MARL đã train với green backfill
uv run hpcsim eval \
    --model-dir models/ \
    --algo gas_marl \
    --backfill green \
    --episodes 20 \
    --output-csv eval_gasmarl_green.csv

# Eval MaskablePPO với easy backfill
uv run hpcsim eval \
    --model-dir models/ \
    --algo maskable_ppo \
    --backfill easy \
    --episodes 20 \
    --output-csv eval_ppo_easy.csv
```

### So sánh RL vs Classical

```bash
# So sánh với cùng backfill policy
uv run hpcsim compare \
    --classical fifo,sjf,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --backfill green \
    --runs 5 \
    --output-csv compare_green.csv \
    --plot compare_green.png
```

### Kết quả mong đợi (Green-Backfilling bật)

```
Scheduler       AvgJCT(s)  AvgBSLD  GPUUtil%  CPUUtil%  ReUtil%  Energy(kWh)
fifo+green      1820.1     13.8     74.2      43.1      66.4     12.1
gavel+green     1601.4     11.2     80.3      45.2      68.7     11.6
maskable_ppo+green  1495.3  9.8     83.4      53.1      73.1     10.6   ← RL
gas_marl+green  1471.2      9.2     84.1      55.8      76.3     10.2   ← best
```

---

## Loại Job và Action Masking

### CPU-only, MIG, Hybrid

**CPUJob**: mask=1 khi `num_cpus ≤ avail_cpus` (không cần GPU)

**MIGJob**: mask=1 khi `num_mig ≤ avail_mig_slices`

**HybridJob**: mask=1 **chỉ khi cả GPU lẫn CPU đều đủ** (feature `cpu_schedulable` index 11 báo riêng trạng thái CPU)

```python
# env.action_mask1() trả về [MAX_QUEUE_SIZE=64]
# Tự động đúng cho tất cả resource_type
mask = env.action_mask1()  # 1 = schedulable, 0 = không đủ tài nguyên
```

### Lưu ý cluster không có CPU nodes

Nếu cluster không có CPU nodes (`tiny_test`, `small_v100`):
- CPUJob và HybridJob sẽ luôn có mask=0 — không bao giờ được schedule
- Thuật toán vẫn hoạt động bình thường
- Để tận dụng CPU/MIG: dùng `hpc_realistic`, `a100_mig_cluster`, `cloud_mixed`

---

## Pipeline Đầy Đủ (Khuyến Nghị)

```bash
# Bước 1: Train GAS-MARL + Green-Backfilling 
uv run hpcsim train \
    --algo gas_marl \
    --epochs 300 \
    --traj 200 \
    --backfill green \
    --eta 0.002 \
    --ckpt-interval 50 \
    --save-dir models/

# Bước 2: Train MaskablePPO + EASY-Backfilling (để so sánh) 
uv run hpcsim train \
    --algo maskable_ppo \
    --epochs 300 \
    --traj 200 \
    --backfill easy \
    --save-dir models/

# Bước 3: Train baseline (không backfill, cho ablation) 
uv run hpcsim train \
    --algo all \
    --epochs 300 \
    --backfill none \
    --save-dir models/ablation_none/

# Bước 4: Eval (backfill PHẢI khớp với lúc train) 
uv run hpcsim eval \
    --model-dir models/ --algo gas_marl \
    --backfill green --episodes 30 \
    --output-csv eval_gasmarl.csv

# Bước 5: So sánh toàn diện 
uv run hpcsim compare \
    --classical fifo,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --backfill green \
    --runs 5 \
    --output-csv compare_green.csv \
    --plot compare_green.png

# Bước 6: Plot 
uv run hpcsim plot --type learning-curve \
    --input models/gas_marl/train_log.csv
uv run hpcsim plot --type benchmark --input compare_green.csv
```

---

## Best Practices

**1. Matching policy train/eval**
Luôn dùng `--backfill green` cho cả train, eval, và compare khi nghiên cứu GAS-MARL. Thay đổi backfill sau khi train làm vô hiệu kết quả vì môi trường agent đã học sẽ khác.

**2. Số epoch**
Bắt đầu với 300 epoch. Nếu `std_reward` còn cao sau 200 epoch, tăng `--traj` lên 200–300 thay vì tăng epoch để tăng tính ổn định mỗi epoch.

**3. Entropy monitoring**
Theo dõi cột `entropy` trong `train_log.csv`. Entropy giảm quá nhanh (<5 epoch) → tăng entropy bonus. Entropy không giảm sau 100 epoch → giảm learning rate.

**4. GPU training**
Dùng `--device cuda` nếu có GPU. Training 300 epoch với traj=200 mất ~45 phút trên GPU so với ~4 giờ trên CPU.

**5. Checkpoint strategy**
Dùng `--ckpt-interval 50` và lưu best model (`--no-save-best` OFF). Sau khi training xong, load từ `models/gas_marl/best/` thay vì checkpoint cuối — best model thường tốt hơn 5–10% so với epoch cuối.