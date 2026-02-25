# Hướng dẫn Simulation

## Kiến trúc Cluster

### NodeSpec — Khai báo node

Kể từ v0.3, cluster dùng `NodeSpec` thay vì tuple cũ:

```python
from hpcsim.cluster.cluster import NodeSpec, ClusterConfig
from hpcsim.cluster.hardware import GPUType, CPUType, NodeType, MIGProfile, InterconnectType

# Node GPU thuần (như cũ, backward-compatible)
NodeSpec(gpu_type=GPUType.V100, num_nodes=4, gpus_per_node=8)

# Node CPU-only (login/compute)
NodeSpec(
    node_type=NodeType.CPU_ONLY,
    cpu_type=CPUType.EPYC_7003,
    num_nodes=8, num_sockets=2, cores_per_socket=64, ram_gb=512,
    role="cpu_compute",
)

# Node Mixed: CPU + GPU (phổ biến nhất trong HPC thực tế)
NodeSpec(
    node_type=NodeType.MIXED,
    gpu_type=GPUType.A100, gpus_per_node=8,
    cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
    num_nodes=16, ram_gb=512,
)

# A100 với MIG enabled — 7 × 1g.10gb slices per GPU
NodeSpec(
    node_type=NodeType.MIXED,
    gpu_type=GPUType.A100, gpus_per_node=8,
    cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
    num_nodes=8, ram_gb=512,
    mig_profile=MIGProfile.G1_10GB,   # ← enable MIG
)

# GPU với MPS (multi-process sharing)
NodeSpec(
    gpu_type=GPUType.RTX4090, gpus_per_node=4,
    num_nodes=8,
    mps_max_jobs=4,   # ← 4 jobs share mỗi GPU
)
```

### Cluster Presets

| Tên | Nodes | GPUs | CPU cores | MIG slices | Mô tả |
|-----|-------|------|-----------|------------|-------|
| `tiny_test` | 2 | 8 | 0 | 0 | Smoke test |
| `small_v100` | 4 | 32 | 0 | 0 | Phát triển |
| `medium_heterogeneous_gavel` | 14 | 112 | 0 | 0 | Gavel benchmark |
| `large_mixed` | 16 | 128 | 0 | 0 | Hetero GPU |
| `gogh_hetero` | 18 | 72 | 0 | 0 | GOGH dataset |
| `hpc_realistic` | 28 | 128 | 1984 | 0 | **HPC thực tế** |
| `a100_mig_cluster` | 14 | 96 | 1024 | 448 | A100 + MIG |
| `cloud_mixed` | 28 | 160 | 2048 | 0 | Cloud mixed |
| `h100_supercluster` | 36 | 256 | 3072 | 0 | H100 + NDR |

### Loại tài nguyên và cách allocate

```python
from hpcsim.cluster.cluster import Cluster, CLUSTER_CONFIGS

c = Cluster(CLUSTER_CONFIGS["hpc_realistic"])

# GPU allocation (trả về list gpu_ids)
gpu_ids = c.find_best_placement(num_gpus=4, prefer_consolidated=True)
ok = c.allocate("job_001", gpu_ids, memory_per_gpu_gb=8.0)
c.deallocate("job_001", gpu_ids, memory_per_gpu_gb=8.0)

# CPU allocation (trả về list "cpu_id:N")
cpu_alloc = c.find_cpu_cores(num_cores=16, prefer_consolidated=True)
# → e.g. ["n_cpu_only_0_cpu0:16"]
ok = c.allocate_cpu("job_002", cpu_alloc)
c.deallocate_cpu("job_002", cpu_alloc)

# MIG allocation
mig_ids = c.find_mig_slices(num_slices=3, profile=MIGProfile.G1_10GB)
ok = c.allocate_mig("job_003", mig_ids)
c.deallocate_mig("job_003", mig_ids)

# Cluster snapshot
snap = c.snapshot()
# {total_gpus, busy_gpus, free_gpus, gpu_utilization,
#  total_mig_slices, free_mig_slices,
#  total_cpu_cores, free_cpu_cores, cpu_utilization, power_watts}
```

---

## Loại Job

### GPU jobs (mặc định)

```python
from hpcsim.workload.job import TrainingJob, InferenceJob, LLMJob, ModelArch, GPUType

job = TrainingJob(
    arch=ModelArch.RESNET50,
    num_gpus_requested=4,
    gpu_type_preference=GPUType.V100,  # None = bất kỳ
    num_iterations=50_000,
    scheduling_mode=SchedulingMode.GANG,
)
```

### CPU-only jobs

```python
from hpcsim.workload.job import CPUJob

job = CPUJob(
    num_cpus_requested=32,    # tổng cores cần
    min_cpus=8,               # tối thiểu (elastic)
    max_cpus=64,              # tối đa
    base_duration_sec=600.0,  # thời gian tại ref_cores
    ref_cores=4,              # parallelism tham chiếu
    memory_gb=64.0,
)

# Ước tính thời gian với số cores thực tế
print(job.effective_duration(actual_cores=32))  # nhanh hơn ~ 6×
```

Công thức thời gian (Amdahl, parallel_frac = 0.8):
```
speedup = 1 / ((1 - 0.8) + 0.8 × ref_cores / actual_cores)
duration = base_duration / speedup
```

### MIG jobs

```python
from hpcsim.workload.job import MIGJob
from hpcsim.cluster.hardware import MIGProfile

job = MIGJob(
    mig_profile=MIGProfile.G1_10GB,   # 1/7 A100, 10 GB
    num_mig_requested=2,               # 2 slices đồng thời
    base_duration_sec=120.0,
)
# memory_per_gpu_gb tự tính từ profile spec
```

**MIG profiles (A100-80GB):**

| Profile | Compute | Memory | Slots/GPU |
|---------|---------|--------|-----------|
| `1g.10gb` | 1/7 | 10 GB | 7 |
| `2g.20gb` | 2/7 | 20 GB | 3 |
| `3g.40gb` | 3/7 | 40 GB | 2 |
| `7g.80gb` | 7/7 | 80 GB | 1 |

### Hybrid CPU+GPU jobs

```python
from hpcsim.workload.job import HybridJob

job = HybridJob(
    num_gpus_requested=4,    # GPU cho training
    num_cpus_requested=16,   # CPU cores cho DataLoader
    arch=ModelArch.RESNET50,
    num_iterations=20_000,
    scheduling_mode=SchedulingMode.GANG,
)
# Scheduler PHẢI cấp phát CẢ GPU lẫn CPU trước khi job bắt đầu
```

---

## Workload Generator

```python
from hpcsim.workload.generator import WorkloadGenerator, WorkloadConfig

config = WorkloadConfig(
    duration=86400,           # 24 giờ
    arrival_process="poisson", # hoặc "pareto", "diurnal"
    mean_arrival_interval=30, # giây
    # Mix loại job (mặc định v0.3):
    training_fraction=0.62,
    inference_fraction=0.13,
    llm_fraction=0.08,
    hpo_fraction=0.04,
    cpu_fraction=0.08,        # CPU-only jobs
    mig_fraction=0.03,        # MIG slice jobs
    hybrid_fraction=0.02,     # CPU+GPU hybrid
    rng_seed=42,
)

jobs = WorkloadGenerator(config).generate()
```

Hoặc dùng preset trace:
```bash
hpcsim generate --trace philly --duration 86400 --output workload.json
hpcsim generate --trace alibaba --output workload.csv
```

---

## Metrics

```python
summary = sim.run()

# GPU metrics
summary["avg_gpu_util"]      # GPU utilization trung bình
summary["avg_jct_s"]         # Average Job Completion Time
summary["avg_bsld"]          # Average Bounded Slowdown

# CPU metrics (mới)
summary["avg_cpu_util"]      # CPU core utilization trung bình
summary["cpu_jobs_completed"]  # Số CPU-only jobs hoàn thành
summary["mig_jobs_completed"]  # Số MIG jobs hoàn thành
summary["hybrid_jobs_completed"]

# Energy metrics
summary["renewable_energy_utilization"]  # ReUtil (0..1)
summary["total_energy_kwh"]
```
