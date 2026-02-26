# HPCSim — HPC GPU/CPU Cluster Scheduler Simulator

Simulator nghiên cứu các thuật toán lập lịch cho HPC heterogeneous cluster, tích hợp năng lượng tái tạo và các thuật toán RL Scheduling.

## Tính năng nổi bật

- **Heterogeneous Cluster thực tế**: GPU-only, CPU-only, Mixed CPU+GPU, MIG (A100/H100), MPS
- **3 loại tài nguyên**: GPU vật lý, CPU cores, MIG slices — cấp phát độc lập
- **5 loại job**: TrainingJob, InferenceJob, LLMJob, HPOJob, CPUJob, MIGJob, HybridJob
- **14 scheduler**: FIFO, SJF, Tiresias, Gavel, Pollux, Themis, Chronus, ElasticFlow, MaxMin, Backfill + 2 RL (MaskablePPO, GAS-MARL)
- **RL state space 121 × 12**: Queue + Running + Green forecast + Cluster state
- **Năng lượng xanh**: Solar + Wind model, reward R = ReUtil − η × AvgBSLD

---

## Cài đặt (uv)

```bash
# Cài uv nếu chưa có
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# Clone và cài
git clone https://github.com/nhan2892005/hpcsim.git
cd hpcsim

uv sync               # core (không có PyTorch)
uv sync --extra rl    # với PyTorch (cần cho MaskablePPO, GAS-MARL)
uv sync --all-extras  # full (bao gồm dev tools)

# PyTorch CUDA version
uv pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU only

# Kiểm tra cài đặt
uv run hpcsim info
uv run hpcsim test
```

---

## Quick Start

```bash
# 1. Smoke test
uv run hpcsim test

# 2. Simulation đơn (cluster mặc định hpc_realistic — có GPU + CPU nodes)
uv run hpcsim simulate --scheduler gavel --plot example.png

# 3. So sánh schedulers (bao gồm CPU jobs)
uv run hpcsim benchmark --schedulers fifo,sjf,tiresias,gavel,pollux --plot example.png

# 4. Train RL (GPU + CPU + MIG jobs)
uv run hpcsim train --algo all --epochs 300

# 5. So sánh RL vs classical
uv run hpcsim compare \
    --classical fifo,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --plot example.png
```

---

## Cơ sở lý thuyết

### Cluster và tài nguyên

HPCSim mô phỏng HPC heterogeneous cluster gồm ba loại node:

**GPU-only node** — Node chuyên GPU, không schedule CPU:
```
NodeSpec(gpu_type=GPUType.V100, num_nodes=4, gpus_per_node=8)
```

**CPU-only node** — Login node, data preprocessing, parameter server:
```
NodeSpec(node_type=NodeType.CPU_ONLY, cpu_type=CPUType.EPYC_7003,
         num_nodes=8, num_sockets=2, cores_per_socket=64)
```

**Mixed node** — Node thực tế: CPU cores + GPU accelerators:
```
NodeSpec(node_type=NodeType.MIXED,
         gpu_type=GPUType.A100, gpus_per_node=8,
         cpu_type=CPUType.EPYC_7002, num_sockets=2,
         mig_profile=MIGProfile.G1_10GB)  # tùy chọn: bật MIG
```

### MIG (Multi-Instance GPU)

A100 và H100 hỗ trợ phân vùng phần cứng cứng (MIG):

| Profile | Compute | Memory | Slots/GPU | Use case |
|---------|---------|--------|-----------|---------|
| `1g.10gb` | 1/7 | 10 GB | 7 | Inference nhỏ, fine-tuning |
| `2g.20gb` | 2/7 | 20 GB | 3 | Medium inference |
| `3g.40gb` | 3/7 | 40 GB | 2 | Larger models |
| `7g.80gb` | 7/7 | 80 GB | 1 | Full GPU (= không MIG) |

Khác với MPS (time-sharing), MIG đảm bảo **cách ly hoàn toàn** — mỗi slice có compute engine, memory controller, và cache riêng.

### Job Types

| Loại | Class | Tài nguyên | Ví dụ |
|------|-------|-----------|-------|
| GPU training | `TrainingJob` | GPU(s) | ResNet, BERT fine-tuning |
| LLM | `LLMJob` | nhiều GPU | GPT-2, LLaMA pre-training |
| Inference | `InferenceJob` | GPU | Serving, batch scoring |
| HPO | `HPOJob` | nhiều GPU | Hyperparameter search |
| CPU-only | `CPUJob` | CPU cores | Data prep, feature eng. |
| MIG | `MIGJob` | MIG slice | Light inference, sharing |
| Hybrid | `HybridJob` | GPU + CPU | Training + DataLoader |

### Scheduling Algorithms

| # | Tên | Loại | Nguồn |
|---|-----|------|-------|
| 1 | FIFO | Classical | — |
| 2 | SJF | Classical | — |
| 3 | Tiresias (LAS) | Classical | Gu et al., NSDI'19 |
| 4 | E-LAS | Classical | Sultana et al., ICPP'20 |
| 5 | MLFQ | Classical | HPC standard |
| 6 | Gavel | Classical | Narayanan et al., OSDI'20 |
| 7 | Pollux | Classical | Qiao et al., OSDI'21 |
| 8 | Themis | Classical | Mahajan et al., NSDI'20 |
| 9 | Chronus | Classical | Gao et al., SoCC'21 |
| 10 | ElasticFlow | Classical | Gu et al., ASPLOS'23 |
| 11 | MaxMinFairness | Classical | Ghodsi et al., NSDI'11 |
| 12 | Backfill | Classical | HPC standard (EASY) |
| 13 | **MaskablePPO** | RL | PPO + action masking |
| 14 | **GAS-MARL** | RL | Chen et al., FGCS'25 |

### Reward Function (RL)

```
R = ReUtil − η × AvgBSLD

ReUtil  = ∫ P_renewable(t) dt / ∫ P_total(t) dt
AvgBSLD = mean[ (wait_time + exec_time) / max(τ, estimated_runtime) ]
η       = 0.002  (điều chỉnh bằng --eta)
```

---

## Cấu trúc dự án

```
hpcsim/
├── src/hpcsim/
│   ├── cluster/
│   │   ├── hardware.py      # GPUSpec, CPUSpec, MIGProfile, ServerNode
│   │   └── cluster.py       # NodeSpec, Cluster, CLUSTER_CONFIGS
│   ├── workload/
│   │   ├── job.py           # TrainingJob, CPUJob, MIGJob, HybridJob, ...
│   │   └── generator.py     # WorkloadGenerator, WorkloadConfig
│   ├── scheduler/
│   │   └── schedulers.py    # 14 schedulers (BaseScheduler, ...)
│   ├── simulator/
│   │   └── engine.py        # Discrete-event simulation engine
│   ├── metrics/
│   │   └── collector.py     # MetricsCollector (GPU + CPU + energy)
│   ├── energy/
│   │   └── renewable.py     # Solar + Wind power model
│   ├── rl/
│   │   ├── env.py           # HPCGreenEnv (121×12 obs, CPU/MIG/Hybrid)
│   │   ├── networks.py      # MaskablePPOActor/Critic, GASMARLActor/Critic
│   │   ├── maskable_ppo.py  # MaskablePPOAgent + train loop
│   │   ├── gas_marl.py      # GASMARLAgent + train loop
│   │   └── train.py         # CLI train entry point
│   ├── benchmark/
│   │   └── runner.py        # BenchmarkRunner
│   └── cli.py               # 11-command CLI
├── docs/
│   ├── simulation-guide.md  # Cluster, Job, Metrics API
│   ├── rl-training.md       # RL state space, action, reward, training
│   └── custom-scheduler.md  # Viết scheduler mới
├── pyproject.toml
└── README.md
```

---

## CLI — Tất cả lệnh

```
hpcsim info         Thông tin môi trường, packages, GPU
hpcsim list         Liệt kê schedulers, clusters, traces
hpcsim test         Smoke tests (30-60 giây)

hpcsim simulate     Chạy một simulation
hpcsim benchmark    So sánh nhiều schedulers
hpcsim generate     Tạo workload trace
hpcsim replay       Chạy lại workload đã lưu

hpcsim train        Train RL (maskable_ppo / gas_marl / all)
hpcsim eval         Đánh giá model đã train
hpcsim compare      So sánh RL vs classical
hpcsim plot         Vẽ biểu đồ từ CSV
```

### Ví dụ chi tiết

```bash
# --- Simulation ---

# Cluster hpc_realistic: có GPU + CPU nodes, workload hỗn hợp
uv run hpcsim simulate \
    --scheduler gavel \
    --cluster hpc_realistic \
    --duration 86400 \
    --arrival-rate 30 \
    --seed 42 \
    --plot example.png \
    --output-json results_gavel.json

# Cluster A100 với MIG
uv run hpcsim simulate \
    --scheduler fifo \
    --cluster a100_mig_cluster \
    --plot example.png

# --- Benchmark ---

# So sánh 5 runs, lưu CSV
uv run hpcsim benchmark \
    --schedulers fifo,sjf,tiresias,gavel,pollux \
    --cluster hpc_realistic \
    --runs 5 \
    --duration 86400 \
    --output-csv bench.csv \
    --plot example.png

# --- Workload ---

# Tạo workload có CPU jobs (mặc định từ v0.3)
uv run hpcsim generate \
    --duration 86400 \
    --arrival-rate 25 \
    --seed 42 \
    --output workload.json

# Replay workload đã lưu với các schedulers khác nhau
uv run hpcsim replay --workload workload.json --scheduler fifo  --output-json r_fifo.json
uv run hpcsim replay --workload workload.json --scheduler gavel --output-json r_gavel.json

# --- RL Training ---

# Train với cluster có CPU nodes (mặc định từ v0.3)
uv run hpcsim train --algo all --epochs 300 --traj 200 --cluster hpc_realistic

# Train với A100 MIG cluster
uv run hpcsim train --algo all --epochs 300 --cluster a100_mig_cluster

# Resume từ checkpoint
uv run hpcsim train --algo maskable_ppo --resume auto --epochs 200

# Điều chỉnh trade-off xanh vs latency
uv run hpcsim train --algo all --eta 0.001   # ưu tiên renewable energy
uv run hpcsim train --algo all --eta 0.02    # ưu tiên latency

# Đánh giá
uv run hpcsim eval \
    --model-dir models/ \
    --algo all \
    --episodes 20 \
    --output-csv eval.csv

# So sánh RL vs classical (5 runs mỗi scheduler)
uv run hpcsim compare \
    --classical fifo,sjf,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --cluster hpc_realistic \
    --runs 5 \
    --output-csv compare.csv \
    --plot example.png

# --- Plots ---
uv run hpcsim plot --type learning-curve --input models/maskable_ppo/train_log.csv
uv run hpcsim plot --type benchmark      --input bench.csv --output bench_plot.png
```

---

## API nhanh

```python
import sys; sys.path.insert(0, "src")

# ── Cluster ──────────────────────────────────────────────────────────────────
from hpcsim.cluster.cluster import Cluster, CLUSTER_CONFIGS
cluster = Cluster(CLUSTER_CONFIGS["hpc_realistic"])
print(cluster.describe())
# → GPUs=128, CPU cores=1984, 28 nodes

# ── Workload ─────────────────────────────────────────────────────────────────
from hpcsim.workload.generator import WorkloadGenerator, WorkloadConfig
from hpcsim.workload.job import CPUJob, MIGJob, HybridJob, ResourceType

jobs = WorkloadGenerator(WorkloadConfig(duration=3600, rng_seed=42)).generate()
print(f"Generated {len(jobs)} jobs")
# Mix mặc định: ~62% GPU, 8% CPU, 3% MIG, 2% Hybrid, ...

# ── Simulation ───────────────────────────────────────────────────────────────
from hpcsim.simulator.engine import SimulationEngine
from hpcsim.scheduler.schedulers import SCHEDULER_REGISTRY
from hpcsim.energy.renewable import RenewableEnergyModule

scheduler = SCHEDULER_REGISTRY["gavel"](cluster)
engine = SimulationEngine(cluster, scheduler, jobs,
                          energy_module=RenewableEnergyModule())
summary = engine.run()
print(summary)
# avg_jct_s, avg_bsld, avg_gpu_util, avg_cpu_util,
# renewable_energy_utilization, total_energy_kwh, ...

# ── RL Environment ───────────────────────────────────────────────────────────
from hpcsim.rl.env import HPCGreenEnv, EnvConfig
import numpy as np

env = HPCGreenEnv(EnvConfig(cluster_config="hpc_realistic"))
obs = env.reset()              # (1452,) = 121 × 12
mask = env.action_mask1()      # (64,) — 1 = schedulable
action = int(np.argmax(mask))
obs, reward, done, *_ = env.step(action)
```

---

## Workflow nghiên cứu điển hình

### 1. Benchmark schedulers

```bash
uv run hpcsim generate --duration 86400 --seed 42 --output wl.json
for sched in fifo sjf tiresias gavel pollux; do
    uv run hpcsim replay --workload wl.json --scheduler $sched \
        --output-json results_$sched.json
done
uv run hpcsim plot --type benchmark --input bench.csv
```

### 2. Nghiên cứu ảnh hưởng CPU fraction

```python
# Thay đổi tỉ lệ CPU jobs và so sánh kết quả
for cpu_frac in [0.0, 0.05, 0.10, 0.20]:
    cfg = WorkloadConfig(cpu_fraction=cpu_frac, rng_seed=42)
    jobs = WorkloadGenerator(cfg).generate()
    # ... chạy simulation với từng scheduler
```

### 3. Pipeline RL đầy đủ

```bash
# Train
uv run hpcsim train --algo all --epochs 500 --cluster hpc_realistic

# Evaluate
uv run hpcsim eval --model-dir models/ --episodes 30 --output-csv eval.csv

# Compare
uv run hpcsim compare \
    --classical fifo,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --runs 5 --plot example.png --output-csv compare.csv

# Plot learning curves
uv run hpcsim plot --type learning-curve \
    --input models/maskable_ppo/train_log.csv
uv run hpcsim plot --type learning-curve \
    --input models/gas_marl/train_log.csv
```

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|----------|---------|
| [docs/simulation-guide.md](docs/simulation-guide.md) | Cluster, NodeSpec, Job API, Metrics |
| [docs/rl-training.md](docs/rl-training.md) | State space, Action masking CPU/MIG, Training, Eval |
| [docs/custom-scheduler.md](docs/custom-scheduler.md) | Viết scheduler mới kế thừa BaseScheduler |

---

## Tham khảo

```
@article{10.1145/3638757,
author = {Ye, Zhisheng and Gao, Wei and Hu, Qinghao and Sun, Peng and Wang, Xiaolin and Luo, Yingwei and Zhang, Tianwei and Wen, Yonggang},
title = {Deep Learning Workload Scheduling in GPU Datacenters: A Survey},
year = {2024},
issue_date = {June 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {56},
number = {6},
issn = {0360-0300},
url = {https://doi.org/10.1145/3638757},
doi = {10.1145/3638757},
journal = {ACM Comput. Surv.},
month = jan,
articleno = {146},
numpages = {38},
keywords = {Deep learning systems, datacenter scheduling}
}

@manual{nvidia_mig_guide,
  title        = {Multi-Instance GPU (MIG)},
  author       = {{NVIDIA Corporation}},
  year         = {2023},
  url          = {https://docs.nvidia.com/dgx/dgxa100-user-guide/using-mig.html},
  note         = {Accessed: February 25, 2026},
  organization = {NVIDIA Corporation},
  howpublished = {\url{https://docs.nvidia.com/dgx/dgxa100-user-guide/using-mig.html}}
}

@article{CHEN2025107760,
title = {GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning},
journal = {Future Generation Computer Systems},
volume = {167},
pages = {107760},
year = {2025},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2025.107760},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X2500055X},
author = {Rui Chen and Weiwei Lin and Huikang Huang and Xiaoying Ye and Zhiping Peng},
keywords = {Job scheduling, High-performance computing, Deep Reinforcement Learning, Renewable energy, Green computing},
}

@Article{a18070385,
AUTHOR = {Chab, Robert and Li, Fei and Setia, Sanjeev},
TITLE = {Algorithmic Techniques for GPU Scheduling: A Comprehensive Survey},
JOURNAL = {Algorithms},
VOLUME = {18},
YEAR = {2025},
NUMBER = {7},
ARTICLE-NUMBER = {385},
URL = {https://www.mdpi.com/1999-4893/18/7/385},
ISSN = {1999-4893},
}

@inproceedings{10.1145/3748273.3749212,
author = {Kumar, Sumit and Temura, Arjun and Sharma, Naman and Singh, Ramanjeet and Dadhania, Meet and Tammana, Praveen and Burla, Satananda and Kamaluddin, Abed Mohammad and Shah, Rinku},
title = {Simulating LLM training workloads for heterogeneous compute and network infrastructure},
year = {2025},
isbn = {9798400720826},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3748273.3749212},
doi = {10.1145/3748273.3749212},
booktitle = {Proceedings of the 2nd Workshop on Networks for AI Computing},
pages = {105–107},
numpages = {3},
keywords = {Distributed Training, Heterogeneous GPU Cluster, LLM Simulator},
location = {Coimbra, Portugal},
series = {NAIC '25}
}

```
