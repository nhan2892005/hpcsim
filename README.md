# HPCSim — HPC GPU Cluster Scheduler Simulator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch Optional](https://img.shields.io/badge/PyTorch-optional-orange.svg)](https://pytorch.org)

**HPCSim** là bộ mô phỏng sự kiện rời rạc (discrete-event simulator) cho cụm HPC GPU không đồng nhất (heterogeneous GPU cluster), được xây dựng phục vụ nghiên cứu thuật toán lập lịch — bao gồm các bộ lập lịch học tăng cường (RL-based schedulers) tích hợp mô hình năng lượng tái tạo (solar + wind).

---

## Mục lục

1. [Cơ sở lý thuyết](#1-cơ-sở-lý-thuyết)
2. [Kiến trúc dự án](#2-kiến-trúc-dự-án)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt](#4-cài-đặt)
5. [Quick Start](#5-quick-start)
6. [Các bước chạy cơ bản](#6-các-bước-chạy-cơ-bản)
7. [Tài liệu chi tiết](#7-tài-liệu-chi-tiết)

---

## 1. Cơ sở lý thuyết

### 1.1 HPC GPU Cluster là gì?

Một **cụm HPC GPU** bao gồm nhiều node máy tính kết nối qua mạng tốc độ cao (InfiniBand, NVLink). Mỗi node chứa một số GPU (có thể không đồng nhất về loại và hiệu năng) để thực thi các tác vụ tính toán song song cường độ cao như huấn luyện mô hình AI, mô phỏng vật lý, phân tích dữ liệu lớn.

```
Cụm HPC điển hình
══════════════════════════════════════════════════
  Node 0 (high-end)  :  [V100 × 8]  128 GB RAM
  Node 1 (high-end)  :  [V100 × 8]  128 GB RAM
  Node 2 (mid-range) :  [T4   × 4]   64 GB RAM
  Node 3 (mid-range) :  [T4   × 4]   64 GB RAM
  Node 4 (economy)   :  [K80  × 2]   32 GB RAM
══════════════════════════════════════════════════
                ↑
         Kết nối InfiniBand (100 Gb/s)
```

### 1.2 Vòng đời Job trong HPC

```
SUBMITTED → QUEUED → RUNNING → COMPLETED
               ↑           ↑
          (chờ GPU)   (đang dùng GPU)
```

| Trạng thái | Mô tả |
|------------|-------|
| SUBMITTED  | Job gửi lên hệ thống |
| QUEUED     | Chờ trong hàng đợi; chưa đủ GPU khả dụng |
| RUNNING    | GPU đã được cấp phát; job đang thực thi |
| COMPLETED  | Job hoàn thành; GPU giải phóng |

**Wait time** = thời điểm bắt đầu − thời điểm gửi lên.  
**Slowdown (BSLD)** = `(wait + runtime) / max(τ, runtime)` với τ = 10s.

### 1.3 Bài toán Lập lịch HPC

Bộ lập lịch quyết định **job nào chạy trên GPU nào và khi nào**, với ràng buộc:
- Một job cần đúng số lượng GPU yêu cầu
- GPU đang bận không được chia sẻ
- Một số job yêu cầu loại GPU cụ thể (V100 cho FP16, K80 cho inference nhẹ)

Mục tiêu tối ưu điển hình:

| Metric | Ý nghĩa |
|--------|---------|
| ↓ Avg JCT | Giảm thời gian hoàn thành trung bình |
| ↓ AvgBSLD | Giảm độ trễ quy chuẩn |
| ↑ GPU Util | Tăng tỉ lệ sử dụng GPU |
| ↑ ReUtil | Tăng tỉ lệ dùng điện tái tạo |

### 1.4 Các thuật toán Lập lịch Cổ điển

| Thuật toán | Ý tưởng chính | Ưu điểm | Hạn chế |
|------------|--------------|---------|---------|
| **FIFO** | Thứ tự gửi lên | Đơn giản, công bằng | Kém hiệu quả khi job nhỏ sau job lớn |
| **SJF** | Job ngắn trước | Giảm JCT trung bình | Có thể bỏ đói job dài |
| **Backfilling** | Lấp khoảng trống GPU khi chờ job ưu tiên | Tăng utilization | Cần dự báo runtime |
| **Tiresias** | Ưu tiên theo lịch sử GPU-time | Tốt cho DL workload | Cần dữ liệu lịch sử |
| **Gavel** | Heterogeneity-aware | Tốt cho cụm không đồng nhất | Phức tạp |
| **Pollux** | Adaptive resource allocation | Tự điều chỉnh batch size | Cần hook vào training loop |

### 1.5 RL Scheduling và Năng lượng Tái tạo

Dựa trên GAS-MARL (Chen et al., FGCS 2025), bộ lập lịch RL học cách cân bằng giữa hiệu suất và sử dụng điện tái tạo:

```
State  = [job queue features × 64] + [running jobs × 32] + [energy forecast × 24h]
Action = (job_to_schedule, delay_type)
Reward = ReUtil − η × AvgBSLD   (sparse, cuối episode)
```

**ReUtil** (Renewable Energy Utilization):

```
ReUtil = ∫ min(P_green(t), P_cluster(t)) dt  /  ∫ P_cluster(t) dt
```

### 1.6 Mô hình Năng lượng Tái tạo

**Solar**: `P_solar(t) = 0.2 × 200m² × 1000 W/m² × sin(π(h−6)/14)` cho 6 ≤ h ≤ 20  
**Wind**: turbine piecewise — cut-in 2.5 m/s, rated 15 m/s, cut-out 30 m/s, P_rated = 7.2 kW  
**Wind speed**: Weibull(k=2) với AR(1) smoothing để tạo chuỗi thời gian thực tế

---

## 2. Kiến trúc Dự án

```
hpcsim/
├── src/hpcsim/
│   ├── cluster/          # Mô hình phần cứng
│   ├── workload/         # Sinh workload ngẫu nhiên
│   ├── simulator/        # Lõi mô phỏng sự kiện rời rạc
│   ├── scheduler/        # FIFO, SJF, Tiresias, Gavel, Pollux...
│   ├── metrics/          # Thu thập và tổng hợp metrics
│   ├── benchmark/        # So sánh nhiều scheduler
│   ├── energy/           # Mô hình solar + wind
│   └── rl/               # RL Schedulers (MaskablePPO, GAS-MARL)
├── examples/
├── docs/
└── pyproject.toml
```

**Luồng xử lý:**

```
WorkloadGenerator → SimulationEngine ←→ Scheduler
                           ↓                ↑
                    MetricsCollector   RenewableEnergyModule
                           ↓
                    BenchmarkRunner (bảng so sánh + plots)
```

---

## 3. Yêu cầu Hệ thống

| Chế độ | Python | RAM | GPU |
|--------|--------|-----|-----|
| Mô phỏng cổ điển | 3.10+ | 2 GB | Không cần |
| RL Training (CPU) | 3.10+ | 8 GB | Không cần (chậm ~20×) |
| RL Training (GPU) | 3.10+ | 8 GB | NVIDIA CUDA 11.8+ |

---

## 4. Cài đặt

```bash
# Cài đặt cơ bản (không RL)
pip install -e .

# Với RL training (cần PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e .[rl]

# Tất cả
pip install -e .[full]

# Kiểm tra
python -c "from hpcsim import Cluster, CLUSTER_CONFIGS; print('Core OK')"
python -c "from hpcsim.rl.env import HPCGreenEnv; print('RL OK')"
```

---

## 5. Quick Start

### 5.1 Mô phỏng đơn giản

```python
from hpcsim import (Cluster, CLUSTER_CONFIGS, create_scheduler,
                    SimulationEngine, MetricsCollector)
from hpcsim.workload.generator import WorkloadGenerator, WorkloadConfig

cluster   = Cluster(CLUSTER_CONFIGS["tiny_test"])
jobs      = WorkloadGenerator(WorkloadConfig(duration=3600)).generate()
scheduler = create_scheduler("fifo", cluster)
metrics   = MetricsCollector()

engine = SimulationEngine(cluster, scheduler, jobs, metrics, max_sim_time=3600)
engine.run()

s = metrics.summary()
print(f"Jobs done  : {s['jobs_completed']}")
print(f"Avg JCT    : {s['avg_jct']:.1f}s")
print(f"GPU Util   : {s['gpu_utilization']:.1%}")
print(f"ReUtil     : {s['renewable_energy_utilization']:.1%}")
```

### 5.2 Benchmark nhiều scheduler

```python
from hpcsim.benchmark.runner import BenchmarkRunner, BenchmarkConfig

cfg = BenchmarkConfig(
    schedulers=["fifo", "sjf", "tiresias", "gavel", "pollux"],
    cluster_config="medium_heterogeneous_gavel",
    num_runs=3,
    sim_duration=3600,
)
runner = BenchmarkRunner(cfg)
runner.print_table(runner.run())
```

```
Scheduler  | Jobs | AvgJCT  | GPUUtil% | ReUtil%
-----------+------+---------+----------+---------
fifo       |   87 |  420.2s |    48.3% |   78.1%
sjf        |  112 |  223.1s |    62.7% |   85.1%
tiresias   |  118 |  198.4s |    65.1% |   87.6%
gavel      |  134 |  259.3s |    68.9% |   94.6%
pollux     |  141 |  201.7s |    71.2% |   95.7%
```

### 5.3 Train RL Scheduler

```bash
# Train nhanh để test
python -m hpcsim.rl.train train \
    --algo maskable_ppo \
    --epochs 50 --traj 50 \
    --ckpt-interval 10 --log-interval 5

# Train đầy đủ
python -m hpcsim.rl.train train \
    --algo all --epochs 300 \
    --ckpt-interval 50 --save-dir models/
```

**Output mẫu:**

```
  ========================================================
  [MaskablePPO] Device=CUDA  epochs=300  traj/epoch=100
  checkpoint_interval=50  save_best=True  log_interval=10
  ========================================================
  Epoch    Reward     ReUtil   AvgBSLD       ETA
  ──────────────────────────────────────────────────────
  [░░░░░░░░░░░░░░░░] ep=   1  reward=-0.0023  green=0.7812  bsld=0.1643  ETA=45.2m
  [██░░░░░░░░░░░░░░] ep=  20  reward=+0.0841  green=0.8234  bsld=0.1201  ETA=38.1m
  [████░░░░░░░░░░░░] ep=  50  reward=+0.1423  green=0.8901  bsld=0.0982  ETA=28.3m
  ✓ Checkpoint → models/maskable_ppo/checkpoints/epoch_0050
  ...
  [MaskablePPO] Done  (42.3 min)
  Final model  → models/maskable_ppo
  Training log → models/maskable_ppo/train_log.csv
  Best model   → models/maskable_ppo/checkpoints/best  (reward=0.1687)
  Checkpoints  → models/maskable_ppo/checkpoints  (6 periodic + 1 best)
```

---

## 6. Các bước chạy cơ bản

### Bước 1 — Chọn cluster

```python
from hpcsim import CLUSTER_CONFIGS
# Các lựa chọn có sẵn:
# tiny_test | small_v100 | medium_heterogeneous_gavel | large_mixed | gogh_hetero
```

### Bước 2 — Sinh workload

```python
from hpcsim.workload.generator import WorkloadGenerator, WorkloadConfig

jobs = WorkloadGenerator(WorkloadConfig(
    duration=7200,           # 2 giờ mô phỏng
    arrival_rate=0.02,       # ~1 job mỗi 50 giây
    gpu_dist={1:0.3, 2:0.3, 4:0.3, 8:0.1},
    rng_seed=42,
)).generate()
```

### Bước 3 — Chạy mô phỏng & lấy metrics

```python
from hpcsim import Cluster, CLUSTER_CONFIGS, create_scheduler, SimulationEngine, MetricsCollector

cluster   = Cluster(CLUSTER_CONFIGS["medium_heterogeneous_gavel"])
scheduler = create_scheduler("gavel", cluster)
metrics   = MetricsCollector()
engine    = SimulationEngine(cluster, scheduler, jobs, metrics, max_sim_time=7200)
engine.run()
print(metrics.summary())
```

### Bước 4 — So sánh schedulers

```bash
python -m hpcsim benchmark \
    --schedulers fifo,tiresias,gavel,pollux \
    --cluster medium_heterogeneous_gavel \
    --runs 5 --duration 3600 \
    --output results.csv --plot benchmark.png
```

### Bước 5 — Train, eval, compare RL

```bash
# Train
python -m hpcsim.rl.train train --algo all --epochs 300 --ckpt-interval 50

# Resume từ checkpoint nếu bị gián đoạn
python -m hpcsim.rl.train train --algo maskable_ppo --epochs 300 \
    --resume models/maskable_ppo/checkpoints/epoch_0150

# Đánh giá
python -m hpcsim.rl.train eval --model-dir models/ --episodes 10

# So sánh tổng thể
python -m hpcsim.rl.train compare \
    --classical fifo,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --model-dir models/ --num-runs 3
```

---

## 7. Tài liệu chi tiết

| Tài liệu | Nội dung |
|----------|---------|
| [docs/simulation-guide.md](docs/simulation-guide.md) | Cluster config, job model, workload generation, metrics, benchmark |
| [docs/custom-scheduler.md](docs/custom-scheduler.md) | Viết custom scheduler, Backfilling, Green Backfilling, so sánh |
| [docs/rl-training.md](docs/rl-training.md) | Kiến trúc RL, quy trình train, checkpoint, CLI đầy đủ |
| [examples/green_rl_example.py](examples/green_rl_example.py) | Demo tích hợp đầy đủ |

---

## Tài liệu tham khảo

- Chen et al. (2025). *GAS-MARL: A Novel Bi-Objective Optimization Framework for Renewable Energy and Average Bounded Slowdown in HPC Job Scheduling*. FGCS.
- Gu et al. (2019). *Tiresias: A GPU Cluster Manager for Distributed Deep Learning*. NSDI.
- Narayanan et al. (2020). *Gavel: Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads*. OSDI.
- Qiao et al. (2021). *Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning*. OSDI.

