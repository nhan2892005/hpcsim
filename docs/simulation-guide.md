# Hướng dẫn Chi tiết về Mô phỏng

Tài liệu này mô tả chi tiết mọi thành phần của bộ mô phỏng HPCSim: cấu hình cluster, mô hình job, sinh workload, thu thập metrics, và chạy benchmark.

---

## Mục lục

1. [Kiến trúc Mô phỏng](#1-kiến-trúc-mô-phỏng)
2. [Cấu hình Cluster](#2-cấu-hình-cluster)
3. [Mô hình Job](#3-mô-hình-job)
4. [Sinh Workload](#4-sinh-workload)
5. [Simulation Engine](#5-simulation-engine)
6. [Metrics](#6-metrics)
7. [Benchmark Runner](#7-benchmark-runner)
8. [Mô hình Năng lượng Tái tạo](#8-mô-hình-năng-lượng-tái-tạo)

---

## 1. Kiến trúc Mô phỏng

HPCSim là **discrete-event simulator (DES)** — không có concept "thời gian thực" mà nhảy từ sự kiện sang sự kiện:

```
┌─────────────────────────────────────────────────────┐
│                 SimulationEngine                     │
│                                                      │
│  Priority Queue (sự kiện sắp xếp theo thời gian)    │
│  ┌──────────────────────────────────────────┐       │
│  │  t=0.0   : JOB_ARRIVE  job_id=0          │       │
│  │  t=5.3   : JOB_ARRIVE  job_id=1          │       │
│  │  t=12.1  : JOB_ARRIVE  job_id=2          │       │
│  │  t=120.0 : JOB_FINISH  job_id=0          │       │
│  │  t=180.0 : METRIC_SAMPLE                 │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  Vòng lặp xử lý sự kiện:                            │
│    while queue not empty and t < max_time:           │
│      event = queue.pop()                             │
│      process(event) → có thể push sự kiện mới       │
└─────────────────────────────────────────────────────┘
```

Các loại sự kiện:

| Sự kiện | Khi nào | Tác động |
|---------|---------|---------|
| `JOB_ARRIVE` | Theo phân phối Poisson | Thêm job vào queue, trigger scheduler |
| `JOB_FINISH` | Sau `runtime` giây | Giải phóng GPU, trigger scheduler |
| `METRIC_SAMPLE` | Mỗi 60 giây | Ghi snapshot trạng thái cluster |
| `PREEMPT` | Khi Pollux/Gavel yêu cầu | Dừng job đang chạy, đưa về queue |

---

## 2. Cấu hình Cluster

### 2.1 Cấu trúc ClusterConfig

```python
from hpcsim.cluster.cluster import ClusterConfig, NodeSpec, GPUSpec
from hpcsim.hardware import GPUType

config = ClusterConfig(
    name="my_cluster",
    nodes=[
        NodeSpec(
            name="node_0",
            gpu_types=["V100"],
            gpu_count=8,
            cpu_cores=32,
            ram_gb=128,
        ),
        NodeSpec(
            name="node_1",
            gpu_types=["T4"],
            gpu_count=4,
            cpu_cores=16,
            ram_gb=64,
        ),
    ]
)
```

### 2.2 Các Cluster Preset

```python
from hpcsim import CLUSTER_CONFIGS

# Xem tất cả preset
for name, cfg in CLUSTER_CONFIGS.items():
    total_gpus = sum(n.gpu_count for n in cfg.nodes)
    gpu_types  = set(t for n in cfg.nodes for t in n.gpu_types)
    print(f"{name:35s}  {len(cfg.nodes):2d} nodes  {total_gpus:3d} GPUs  {gpu_types}")
```

| Preset | Nodes | GPUs | GPU Types | Dùng cho |
|--------|-------|------|-----------|---------|
| `tiny_test` | 2 | 8 | V100 | Unit test, debug nhanh |
| `small_v100` | 4 | 32 | V100 | Thí nghiệm đơn giản |
| `medium_heterogeneous_gavel` | 8 | 112 | V100, T4, K80 | **Benchmark tiêu chuẩn** |
| `large_mixed` | 16 | 256 | V100, A100, T4, K80 | Nghiên cứu cụm lớn |
| `gogh_hetero` | 10 | 128 | V100, T4 | Mô phỏng cụm Gogh từ paper |

### 2.3 GPU Specs

Mỗi GPU có thông số hiệu năng ảnh hưởng đến scheduling:

```python
from hpcsim.hardware import GPU_SPECS

# V100
spec = GPU_SPECS["V100"]
print(spec.fp32_tflops)        # 14.0 TFLOPS
print(spec.memory_gb)          # 32 GB
print(spec.tdp_watts)          # 300 W (thermal design power)
print(spec.nvlink_bandwidth)   # 300 GB/s

# Tốc độ tương đối cho DL training (so với V100 = 1.0)
# V100: 1.00, A100: 2.50, T4: 0.65, K80: 0.30
```

### 2.4 Tạo Cluster tùy chỉnh

```python
from hpcsim.cluster.cluster import Cluster, ClusterConfig, NodeSpec

config = ClusterConfig(
    name="my_research_cluster",
    nodes=[
        # 2 node high-end A100
        NodeSpec("node_a100_0", gpu_types=["A100"], gpu_count=8,
                 cpu_cores=64, ram_gb=256),
        NodeSpec("node_a100_1", gpu_types=["A100"], gpu_count=8,
                 cpu_cores=64, ram_gb=256),
        # 4 node mid-range T4
        NodeSpec("node_t4_0", gpu_types=["T4"], gpu_count=4,
                 cpu_cores=16, ram_gb=64),
        NodeSpec("node_t4_1", gpu_types=["T4"], gpu_count=4,
                 cpu_cores=16, ram_gb=64),
        NodeSpec("node_t4_2", gpu_types=["T4"], gpu_count=4,
                 cpu_cores=16, ram_gb=64),
        NodeSpec("node_t4_3", gpu_types=["T4"], gpu_count=4,
                 cpu_cores=16, ram_gb=64),
    ]
)

cluster = Cluster(config)
print(f"Total GPUs: {cluster.total_gpu_count()}")        # 32
print(f"Free GPUs : {cluster.free_gpu_count()}")         # 32
print(f"GPU types : {cluster.available_gpu_types()}")   # {'A100', 'T4'}
```

---

## 3. Mô hình Job

### 3.1 Cấu trúc Job

```python
from hpcsim.workload.generator import Job

job = Job(
    job_id=42,
    submit_time=100.0,           # Giây từ đầu mô phỏng
    num_gpus=4,                  # Số GPU yêu cầu
    runtime=3600.0,              # Runtime thực tế (giây)
    gpu_type_pref=["V100"],      # Loại GPU ưu tiên (None = any)
    memory_gb_per_gpu=16.0,      # Bộ nhớ cần mỗi GPU
    power_per_gpu_w=250.0,       # Công suất tiêu thụ (W/GPU)
    is_deadline_sensitive=False, # Có SLA deadline không?
    deadline=None,               # Deadline tuyệt đối (giây)
)

# Sau khi scheduled
job.start_time = 150.0           # Khi nào bắt đầu chạy
job.finish_time = 3750.0         # Khi nào xong
job.allocated_gpus = [0,1,2,3]  # ID các GPU đã cấp phát

# Metrics tính từ job
wait_time = job.start_time - job.submit_time   # 50s
jct       = job.finish_time - job.submit_time  # 3650s
bsld      = (wait_time + job.runtime) / max(10, job.runtime)
```

### 3.2 Các trường quan trọng

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `job_id` | int | Định danh duy nhất |
| `submit_time` | float | Thời điểm gửi (giây) |
| `num_gpus` | int | Số GPU cần (1, 2, 4, 8, 16, ...) |
| `runtime` | float | Thời gian chạy thực tế (giây) |
| `runtime_estimate` | float | Dự báo runtime (có sai số) |
| `gpu_type_pref` | list[str] | Ưu tiên GPU: `["V100"]`, `None` = any |
| `power_per_gpu_w` | float | Công suất W/GPU khi chạy |
| `memory_gb_per_gpu` | float | VRAM cần mỗi GPU |
| `priority` | float | Mức ưu tiên (cao hơn = ưu tiên hơn) |
| `user_id` | str | Người dùng gửi job (dùng cho fairness) |

---

## 4. Sinh Workload

### 4.1 WorkloadConfig

```python
from hpcsim.workload.generator import WorkloadConfig

config = WorkloadConfig(
    # Thời gian mô phỏng
    duration=86400,              # 24 giờ

    # Tốc độ đến (Poisson process)
    arrival_rate=0.015,          # Job/giây (~54 job/giờ)

    # Phân phối số GPU mỗi job
    gpu_dist={
        1:  0.30,   # 30% job dùng 1 GPU
        2:  0.30,   # 30% job dùng 2 GPU
        4:  0.25,   # 25% job dùng 4 GPU
        8:  0.10,   # 10% job dùng 8 GPU
        16: 0.05,   # 5%  job dùng 16 GPU
    },

    # Phân phối runtime (log-normal)
    runtime_mean_min=30,         # Runtime trung bình tối thiểu (phút)
    runtime_mean_max=120,        # Runtime trung bình tối đa (phút)
    runtime_sigma=0.8,           # Độ lệch chuẩn log-normal

    # Sai số dự báo runtime (cho Backfilling)
    runtime_estimate_noise=0.2,  # ±20% so với runtime thực

    # Phân phối công suất (W/GPU)
    power_mean_w=220.0,
    power_std_w=40.0,

    # Loại GPU ưu tiên
    gpu_type_pref_dist={
        None:    0.50,   # 50% không quan tâm loại GPU
        "V100":  0.30,   # 30% cần V100
        "T4":    0.15,   # 15% chấp nhận T4
        "K80":   0.05,   # 5% cần K80
    },

    # Tỉ lệ job có deadline
    deadline_fraction=0.1,
    deadline_slack_factor=2.0,   # deadline = submit + slack × runtime

    rng_seed=42,                 # Để reproduce
)

jobs = WorkloadGenerator(config).generate()
print(f"Số job: {len(jobs)}")
print(f"Runtime trung bình: {sum(j.runtime for j in jobs)/len(jobs):.0f}s")
```

### 4.2 Phân phối thực tế

**Runtime distribution**: Log-normal phản ánh thực tế — phần lớn job ngắn (vài phút), một số ít cực dài (vài giờ):

```
Phân phối runtime (log-normal, μ=4.5, σ=0.8):
   0-5 min  : ████████████████  32%
  5-15 min  : ████████████      24%
 15-30 min  : ████████          16%
 30-60 min  : ██████            12%
 60-120 min : █████             10%
  > 120 min : ██                 6%
```

**Arrival process**: Poisson (inter-arrival time ~ Exponential) — xấp xỉ tốt với workload HPC thực tế.

### 4.3 Workload từ trace thực

```python
from hpcsim.workload.generator import WorkloadGenerator

# Load từ file CSV (trace thực tế)
jobs = WorkloadGenerator.from_csv(
    "traces/philly_trace.csv",
    columns={
        "submit_time": "submit",
        "num_gpus":    "num_gpu",
        "runtime":     "duration",
    },
    time_scale=1.0,   # Nếu cần scale thời gian
    max_jobs=1000,
)
```

---

## 5. Simulation Engine

### 5.1 Khởi tạo và chạy

```python
from hpcsim.simulator.engine import SimulationEngine

engine = SimulationEngine(
    cluster=cluster,
    scheduler=scheduler,
    jobs=jobs,
    metrics_collector=metrics,
    max_sim_time=86400,             # Dừng sau 24h mô phỏng
    metric_sample_interval=60.0,    # Lấy mẫu mỗi 60 giây
    renewable_config={              # Cấu hình năng lượng (tùy chọn)
        "solar_area_m2": 200,
        "wind_rated_kw": 7.2,
        "lat": 37.5,                # Vĩ độ (ảnh hưởng bức xạ)
    },
    seed=42,
)

engine.run()
print(f"Simulation time: {engine.current_time:.1f}s")
print(f"Events processed: {engine.events_processed}")
```

### 5.2 Hooks và Callbacks

```python
# Thêm callback khi job hoàn thành
def on_job_finish(job, current_time):
    print(f"[t={current_time:.0f}s] Job {job.job_id} done, "
          f"wait={job.start_time - job.submit_time:.0f}s")

engine.on_job_finish = on_job_finish
engine.run()
```

### 5.3 Scheduler Interface

Khi scheduler được gọi:

```python
# Engine gọi scheduler.schedule() sau mỗi sự kiện ARRIVE/FINISH
decision = scheduler.schedule(
    pending=list_of_waiting_jobs,   # Danh sách job đang chờ
    running=list_of_running_jobs,   # Danh sách job đang chạy
    current_time=float,             # Thời gian hiện tại
)

# decision là SchedulingDecision
# decision.assignments: List[(job, list_of_gpu_ids)]
# decision.preemptions: List[job_id]  (cho Preemptive schedulers)
```

---

## 6. Metrics

### 6.1 Danh sách metrics

```python
summary = metrics.summary()

# JCT metrics
summary["avg_jct"]           # Thời gian hoàn thành trung bình (giây)
summary["median_jct"]        # Median JCT
summary["p90_jct"]           # Percentile 90 JCT
summary["p99_jct"]           # Percentile 99 JCT

# Queue metrics
summary["avg_queue_time"]    # Thời gian chờ trung bình
summary["max_queue_time"]    # Thời gian chờ tối đa

# Slowdown
summary["avg_bsld"]          # Avg Bounded Slowdown (lý tưởng = 1.0)

# Throughput
summary["jobs_completed"]    # Tổng số job hoàn thành
summary["jobs_preempted"]    # Tổng số lần preempt

# Resource utilization
summary["gpu_utilization"]   # GPU Utilization trung bình (0–1)
summary["peak_gpu_util"]     # GPU Utilization đỉnh

# Energy
summary["renewable_energy_utilization"]  # ReUtil (0–1)
summary["total_energy_wh"]              # Tổng điện tiêu thụ (Wh)
summary["renewable_energy_wh"]          # Điện xanh dùng được (Wh)
summary["brown_energy_wh"]             # Điện thông thường (Wh)

# Fairness
summary["fairness_index"]    # Jain's fairness index (0–1, cao hơn = công bằng hơn)

# Deadline (nếu có)
summary["deadline_miss_rate"]  # Tỉ lệ job lỡ deadline
```

### 6.2 Công thức chi tiết

**AvgBSLD** (Average Bounded Slowdown):

```
AvgBSLD = (1/N) × Σ max((wait_i + run_i) / max(τ, run_i), 1)

τ = 10 giây (ngưỡng tối thiểu để tránh chia cho 0 với job rất ngắn)
N = tổng số job hoàn thành
```

**GPU Utilization**:

```
Util(t) = (số GPU đang chạy job tại thời điểm t) / (tổng GPU)
AvgUtil = (1/T) × ∫ Util(t) dt
```

**ReUtil** (Renewable Energy Utilization):

```
ReUtil = ∫ min(P_green(t), P_used(t)) dt  /  ∫ P_used(t) dt

P_green(t)  = công suất điện tái tạo khả dụng tại t (Watts)
P_used(t)   = công suất cluster đang tiêu thụ tại t (Watts)
```

**Jain's Fairness Index**:

```
F = (Σ x_i)² / (N × Σ x_i²)

x_i = JCT của user i (trung bình các job)
F ∈ [1/N, 1.0]; F = 1.0 là hoàn toàn công bằng
```

### 6.3 Xem time-series chi tiết

```python
# Lấy chuỗi thời gian GPU utilization theo thời gian
ts = metrics.gpu_utilization_series()
# → pd.DataFrame với columns: ['time', 'utilization']

# Công suất điện tái tạo theo thời gian
ts_energy = metrics.energy_series()
# → pd.DataFrame: ['time', 'renewable_w', 'consumed_w', 'brown_w']

# Vẽ biểu đồ
import matplotlib.pyplot as plt
ts.plot(x='time', y='utilization')
plt.xlabel("Simulation time (s)")
plt.ylabel("GPU Utilization")
plt.title("GPU Utilization over Time")
plt.savefig("gpu_util.png")
```

---

## 7. Benchmark Runner

### 7.1 Cấu hình BenchmarkConfig

```python
from hpcsim.benchmark.runner import BenchmarkConfig

cfg = BenchmarkConfig(
    schedulers=["fifo", "sjf", "tiresias", "gavel", "pollux"],
    cluster_config="medium_heterogeneous_gavel",

    num_runs=5,              # Số lần chạy để lấy trung bình
    sim_duration=3600,       # Thời gian mỗi run (giây)

    # Workload
    workload_duration=3600,
    arrival_rate=0.02,
    gpu_dist={1:0.3, 2:0.3, 4:0.3, 8:0.1},
    rng_seed=0,              # Seed cơ sở (mỗi run tăng thêm 1)

    # Output
    output_csv="results.csv",
    plot_file="benchmark.png",
    verbose=True,
)
```

### 7.2 Chạy và đọc kết quả

```python
from hpcsim.benchmark.runner import BenchmarkRunner

runner  = BenchmarkRunner(cfg)
results = runner.run()

# Mỗi entry là AggregatedResult
for scheduler_name, agg in results.items():
    print(f"\n{scheduler_name}:")
    print(f"  Avg JCT  : {agg.mean['avg_jct']:.1f} ± {agg.std['avg_jct']:.1f}s")
    print(f"  GPU Util : {agg.mean['gpu_utilization']:.1%}")
    print(f"  ReUtil   : {agg.mean['renewable_energy_utilization']:.1%}")

# In bảng so sánh chuẩn
runner.print_table(results)

# Xuất CSV
runner.save_csv(results, "results.csv")

# Vẽ biểu đồ so sánh
runner.plot(results, "benchmark.png")
```

### 7.3 Benchmark qua CLI

```bash
# Benchmark đơn giản
python -m hpcsim benchmark \
    --schedulers fifo,sjf,tiresias,gavel,pollux \
    --cluster medium_heterogeneous_gavel \
    --runs 5 --duration 3600

# Với output
python -m hpcsim benchmark \
    --schedulers fifo,tiresias,gavel,pollux \
    --cluster large_mixed \
    --runs 10 --duration 86400 \
    --output results.csv \
    --plot comparison.png

# Thêm RL schedulers vào so sánh
python -m hpcsim.rl.train compare \
    --classical fifo,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --model-dir models/ \
    --num-runs 5 --duration 3600 \
    --output comparison.csv --plot comparison.png
```

### 7.4 Đọc và phân tích kết quả CSV

```python
import pandas as pd

df = pd.read_csv("results.csv")
print(df.columns.tolist())
# ['scheduler', 'run', 'avg_jct', 'median_jct', 'p90_jct',
#  'avg_queue_time', 'avg_bsld', 'gpu_utilization',
#  'renewable_energy_utilization', 'fairness_index',
#  'deadline_miss_rate', 'total_energy_wh', 'jobs_completed', ...]

# Phân tích thống kê
grouped = df.groupby("scheduler")["avg_jct"].agg(["mean", "std", "min", "max"])
print(grouped)
```

---

## 8. Mô hình Năng lượng Tái tạo

### 8.1 Khởi tạo

```python
from hpcsim.energy.renewable import RenewableEnergyModule

re_module = RenewableEnergyModule(
    total_gpus=112,          # Tổng số GPU để scale công suất
    sim_duration=86400,      # Độ dài mô phỏng (giây)
    solar_area_m2=200,       # Diện tích tấm pin mặt trời
    wind_rated_kw=7.2,       # Công suất định mức turbine gió
    lat=37.5,                # Vĩ độ ảnh hưởng giờ nắng
    seed=42,
)

# Công suất xanh tại thời điểm bất kỳ
t = 36000   # 10 giờ sáng (giây từ 0h)
power = re_module.available_power_watts(t)
print(f"Green power at 10am: {power:.0f} W")

# Dự báo 24 giờ tới
forecast = re_module.get_forecast(current_time=0)
# → list of (solar_w, wind_w) cho 24 slot, mỗi slot 1 giờ
```

### 8.2 Tích hợp với Simulation Engine

```python
engine = SimulationEngine(
    cluster=cluster,
    scheduler=scheduler,
    jobs=jobs,
    metrics_collector=metrics,
    max_sim_time=86400,
    renewable_config={           # Truyền vào engine
        "total_gpus": 112,
        "solar_area_m2": 200,
        "wind_rated_kw": 7.2,
    }
)
```

Engine tự động gắn `renewable_power_watts` vào mỗi `ClusterSnapshot` → MetricsCollector dùng để tính `ReUtil`.

### 8.3 Visualize năng lượng

```python
import numpy as np
import matplotlib.pyplot as plt

hours  = np.arange(0, 24, 0.25)
times  = hours * 3600
powers = [re_module.available_power_watts(t) for t in times]

plt.figure(figsize=(12, 4))
plt.fill_between(hours, powers, alpha=0.4, color="green", label="Green Power")
plt.xlabel("Hour of day")
plt.ylabel("Available Power (W)")
plt.title("Renewable Energy Profile")
plt.legend()
plt.savefig("energy_profile.png")
```

---

## Phụ lục: Danh sách Scheduler có sẵn

```python
from hpcsim import create_scheduler, list_schedulers

print(list_schedulers())
# ['fifo', 'sjf', 'backfill', 'sjf_backfill', 'tiresias',
#  'gavel', 'pollux', 'chronus', 'maskable_ppo', 'gas_marl']

# Tạo scheduler
sched = create_scheduler("gavel", cluster)
```

| Tên | Mô tả | Phù hợp với |
|-----|-------|-------------|
| `fifo` | First-In-First-Out | Baseline |
| `sjf` | Shortest Job First | Giảm avg JCT |
| `backfill` | EASY Backfilling | Tăng utilization |
| `sjf_backfill` | SJF + Backfilling | Cân bằng tốt |
| `tiresias` | GPU-time preemptive | DL workload |
| `gavel` | Heterogeneity-aware | Cụm không đồng nhất |
| `pollux` | Adaptive allocation | Research DL jobs |
| `chronus` | Deadline-aware | SLA-critical workloads |
| `maskable_ppo` | RL + PPO | Cần train trước |
| `gas_marl` | RL + Green-aware | Cần train trước |
