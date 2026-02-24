# Hướng dẫn Huấn luyện RL Scheduler

Tài liệu này giải thích chi tiết cách hoạt động, cách chuẩn bị, quy trình huấn luyện, và cách sử dụng hai RL scheduler có sẵn trong HPCSim: **MaskablePPO** và **GAS-MARL**.

---

## Mục lục

1. [Tổng quan Kiến trúc RL](#1-tổng-quan-kiến-trúc-rl)
2. [Môi trường HPCGreenEnv](#2-môi-trường-hpcgreenenv)
3. [Kiến trúc Neural Network](#3-kiến-trúc-neural-network)
4. [Thuật toán PPO và GAS-MARL](#4-thuật-toán-ppo-và-gas-marl)
5. [Chuẩn bị Huấn luyện](#5-chuẩn-bị-huấn-luyện)
6. [Quy trình Huấn luyện](#6-quy-trình-huấn-luyện)
7. [Checkpoint và Logging](#7-checkpoint-và-logging)
8. [CLI Đầy đủ](#8-cli-đầy-đủ)
9. [Đánh giá và So sánh](#9-đánh-giá-và-so-sánh)
10. [Điều chỉnh Hyperparameter](#10-điều-chỉnh-hyperparameter)
11. [Phát triển RL Scheduler Tùy chỉnh](#11-phát-triển-rl-scheduler-tùy-chỉnh)

---

## 1. Tổng quan Kiến trúc RL

### Formulation bài toán

HPCSim sử dụng framework **Proximal Policy Optimization (PPO)** để huấn luyện agent lập lịch. Bài toán được đặt ra như sau:

```
Agent (Scheduler) ↔ Environment (Simulation)

Agent nhận: State s_t  →  chọn Action a_t  →  nhận Reward r_t  →  lặp
```

**Đặc điểm quan trọng:**

| Thuộc tính | Giá trị |
|------------|---------|
| Reward | Sparse (chỉ ở cuối episode) |
| Episode | Một đoạn mô phỏng = seq_len quyết định |
| Action space | Discrete + masked (chỉ job khả dụng) |
| State space | Continuous, 64×8 + 32×4 + 24×2 = 624 features |

### MaskablePPO vs GAS-MARL

| Tính năng | MaskablePPO | GAS-MARL |
|-----------|-------------|----------|
| Số hành động | 1 (chọn job) | 2 (chọn job + delay) |
| Green-Backfilling | Không | Có |
| Complexity | Thấp hơn | Cao hơn |
| Training time | ~30 min (GPU) | ~60 min (GPU) |
| ReUtil tiêu biểu | 88–92% | 93–97% |

---

## 2. Môi trường HPCGreenEnv

### 2.1 State Space

State là một tensor hình `[total_slots × JOB_FEATURES]` = `[120 × 8]`:

```
State tensor (120 × 8):
  ┌─────────────────────────────────────────────────────┐
  │  Slots 0-63:   Job Queue (pending jobs)             │
  │    Feature 0: wait_time / 3600 (h)                  │
  │    Feature 1: num_gpus / 16                         │
  │    Feature 2: runtime_estimate / 7200               │
  │    Feature 3: power_w / 1000                        │
  │    Feature 4: power_per_gpu / 500                   │
  │    Feature 5: uses_brown_energy (0/1)               │
  │    Feature 6: brown_ratio (0-1)                     │
  │    Feature 7: is_schedulable (0/1)                  │
  ├─────────────────────────────────────────────────────┤
  │  Slots 64-95:  Running Window (đang chạy)           │
  │    Feature 0: time_remaining / 3600                 │
  │    Feature 1: num_gpus / 16                         │
  │    Feature 2: power_w / 1000                        │
  │    Feature 3: finish_time_normalized                │
  │    Features 4-7: padding zeros                      │
  ├─────────────────────────────────────────────────────┤
  │  Slots 96-119: Green Energy Forecast (24h)          │
  │    Feature 0: solar_power / max_solar               │
  │    Feature 1: wind_power / max_wind                 │
  │    Features 2-7: padding zeros                      │
  └─────────────────────────────────────────────────────┘
```

### 2.2 Action Space

**MaskablePPO** — 1 hành động:
```
Action 1: job_idx ∈ {0, 1, ..., 63}
          Chọn job trong queue để schedule tiếp theo
          Masked: job_idx invalid nếu không đủ GPU
```

**GAS-MARL** — 2 hành động:
```
Action 1: job_idx ∈ {0, 1, ..., 63}
          Chọn job

Action 2: delay_type ∈ {0, 1, ..., 12}
  Type 0:     Chạy ngay (immediate)
  Type 1-5:   Delay cho đến khi N job hoàn thành (N=1..5)
  Type 6-12:  Fixed delay: [300, 600, 1200, 1800, 2400, 3000, 3600]s
```

### 2.3 Reward Function

```python
# Sparse reward — chỉ tính cuối episode (seq_len quyết định)
reward = ReUtil - η × AvgBSLD

# ReUtil = tỉ lệ điện xanh thực sự được sử dụng
# AvgBSLD = trung bình bounded slowdown (1.0 là tối ưu)
# η = penalty factor (default=0.005) để cân bằng 2 mục tiêu

# Trong quá trình episode, r = 0 (intermediate steps không có reward)
# Chỉ khi done=True: r = ReUtil - η × AvgBSLD
```

### 2.4 Cấu hình EnvConfig

```python
from hpcsim.rl.env import HPCGreenEnv, EnvConfig
from hpcsim.workload.generator import WorkloadConfig

env_config = EnvConfig(
    # Workload
    workload_config=WorkloadConfig(
        duration=86400,
        arrival_rate=0.015,
        gpu_dist={1:0.3, 2:0.3, 4:0.25, 8:0.1, 16:0.05},
        rng_seed=42,
    ),

    # Cluster
    cluster_config="medium_heterogeneous_gavel",

    # Simulation
    sim_duration_sec=86400,      # 24h mô phỏng mỗi episode

    # Reward
    eta=0.005,                   # Penalty factor cho AvgBSLD

    # Episode
    seq_len=256,                 # Số quyết định mỗi episode

    # Renewable
    seed=42,
)

env = HPCGreenEnv(env_config)
obs = env.reset()                         # obs.shape = (120 × 8,) = (960,)
print(f"Obs shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

---

## 3. Kiến trúc Neural Network

### 3.1 MaskablePPO Actor-Critic

```
                    State Input (120 × 8)
                           │
                    [Job Encoder]
                    Linear(8 → 128) + ReLU
                    Linear(128 → 128) + ReLU
                           │
              ┌────────────┴─────────────────┐
              │                              │
         [Actor Head]                  [Critic Head]
         Linear(128 → 1)               Linear(128 × 120 → 256)
         squeeze → logits (120)        Linear(256 → 1) = V(s)
              │
         Masked Softmax
              │
         Action (job_idx)
```

Tham số actor: ~35K  
Tham số critic: ~50K

### 3.2 GAS-MARL Actor-Critic (Dual-Head)

```
                    State Input (120 × 8)
                           │
                    [Shared Encoder]
                    Linear(8 → 128) + ReLU per slot
                           │
              ┌────────────┴──────────────────────────┐
              │                                       │
     [Job Selection Head]                    [Delay Decision Head]
     Attention over slots 0-63               Conditioned on selected job embedding
     → logits (64) + mask                   → logits (13) + mask2
              │                                       │
     action_job (0-63)                       action_delay (0-12)
```

```python
# Composite probability ratio (GAS-MARL Eq. 5)
r_composite = (π(a_job | s) × π(a_delay | s)) / (π_old(a_job | s) × π_old(a_delay | s))

# PPO clip áp dụng trên r_composite
L_clip = E[min(r_composite × A, clip(r_composite, 1-ε, 1+ε) × A)]
```

---

## 4. Thuật toán PPO và GAS-MARL

### 4.1 MaskablePPO Training Loop

```
Cho mỗi epoch (300 epochs):
  1. Thu thập traj_num (100) trajectories từ môi trường
     - Mỗi trajectory = seq_len (256) quyết định
     - Lưu (state, action, log_prob, value, reward, mask)

  2. Tính GAE advantage:
     A_t = Σ (γλ)^k × δ_{t+k}
     δ_t = r_t + γ V(s_{t+1}) - V(s_t)
     (γ=1.0, λ=0.97 — khuyến nghị từ GAS-MARL paper)

  3. PPO update (8 epochs, batch_size=256):
     L = L_clip(θ) + c1 × L_VF - c2 × H(π)
     L_clip = E[min(r(θ) × A, clip(r(θ), 0.8, 1.2) × A)]
     L_VF   = (V(s) - V_target)²  (MSE)
     H(π)   = Entropy (khuyến khích exploration)

  4. Lưu checkpoint (nếu epoch % interval == 0)
```

### 4.2 GAS-MARL: Green-Backfilling During Delay

```
Khi agent chọn delay_type > 0:
  1. Job bị trì hoãn đến thời điểm delay_until
  2. Trong thời gian chờ, thực hiện "Green Backfilling":
     - Duyệt qua pending queue
     - Tìm job có brown_energy < BROWN_ENERGY_MAX (50 kJ)
     - Nếu có GPU rảnh → schedule ngay (greedy fill)
  3. Khi delay_until tới → schedule job đã delay
```

### 4.3 Hyperparameters mặc định

| Hyperparameter | Giá trị | Mô tả |
|----------------|---------|-------|
| `epochs` | 300 | Số vòng huấn luyện |
| `traj_num` | 100 | Trajectories mỗi epoch |
| `seq_len` | 256 | Quyết định mỗi trajectory |
| `batch_size` | 256 | Mini-batch size trong PPO update |
| `ppo_epochs` | 8 | Số lần update mỗi rollout |
| `clip_eps` | 0.2 | PPO clip ratio |
| `gamma` | 1.0 | Discount factor |
| `gae_lambda` | 0.97 | GAE lambda |
| `lr_actor` | 3e-4 | Learning rate actor |
| `lr_critic` | 1e-3 | Learning rate critic |
| `eta` | 0.005 | Penalty factor cho AvgBSLD |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |

---

## 5. Chuẩn bị Huấn luyện

### 5.1 Yêu cầu

```bash
# Cài đặt PyTorch (chọn 1 trong 2)
# CUDA (khuyến nghị):
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch

# Cài hpcsim với RL support
pip install -e .[rl]

# Kiểm tra
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from hpcsim.rl.env import HPCGreenEnv; print('RL env OK')"
```

### 5.2 Chọn cluster và workload phù hợp

```python
from hpcsim.rl.env import EnvConfig
from hpcsim.workload.generator import WorkloadConfig

# Cấu hình cho nghiên cứu (khuyến nghị)
env_cfg = EnvConfig(
    workload_config=WorkloadConfig(
        duration=86400,          # 1 ngày mô phỏng mỗi episode
        arrival_rate=0.015,      # ~54 job/giờ (realistic)
        rng_seed=42,
    ),
    cluster_config="medium_heterogeneous_gavel",  # 112 GPUs, mixed
    sim_duration_sec=86400,
    eta=0.005,
    seq_len=256,
)
```

**Gợi ý chọn cluster:**

| Mục đích | Cluster | Lý do |
|----------|---------|-------|
| Debug / test nhanh | `tiny_test` | Ít GPU, episode ngắn |
| Nghiên cứu chuẩn | `medium_heterogeneous_gavel` | Giống paper GAS-MARL |
| Cụm lớn thực tế | `large_mixed` | Cần nhiều GPU hơn |

### 5.3 Tính toán thời gian huấn luyện

```
Thời gian ≈ epochs × traj_num × (seq_len / decisions_per_second)

CPU (i7):  ~0.5 decisions/second → 300 × 100 × 256 / 0.5 / 3600 ≈ 4.3 giờ
GPU (V100): ~15 decisions/second → 300 × 100 × 256 / 15 / 3600 ≈ 0.7 giờ

Khuyến nghị bắt đầu với epochs=50, traj=50 để kiểm tra (5–10 phút)
```

---

## 6. Quy trình Huấn luyện

### 6.1 API Python

```python
from hpcsim.rl.train import run_training

# Huấn luyện cơ bản
agents = run_training(
    algo="all",              # "maskable_ppo" | "gas_marl" | "all"
    epochs=300,
    traj_num=100,
    cluster_config="medium_heterogeneous_gavel",
    sim_duration=86400.0,
    eta=0.005,
    save_dir="models/",
    device="auto",           # auto-detect GPU
    seed=42,
)

# Với checkpoint và logging
agents = run_training(
    algo="all",
    epochs=300,
    checkpoint_interval=50,  # Lưu checkpoint mỗi 50 epoch
    save_best=True,          # Lưu model tốt nhất
    log_interval=10,         # In log mỗi 10 epoch
    save_dir="models/",
)

# Chỉ train MaskablePPO với cấu hình tùy chỉnh
from hpcsim.rl.maskable_ppo import train_maskable_ppo
from hpcsim.rl.env import EnvConfig
from hpcsim.workload.generator import WorkloadConfig

env_cfg = EnvConfig(
    workload_config=WorkloadConfig(duration=86400, rng_seed=42),
    cluster_config="medium_heterogeneous_gavel",
    eta=0.003,         # Thử giảm penalty AvgBSLD
    seq_len=512,       # Episode dài hơn
)

agent = train_maskable_ppo(
    env_config=env_cfg,
    save_dir="models/my_ppo",
    epochs=200,
    traj_num=150,
    device="cuda",
    checkpoint_interval=25,
    log_interval=5,
    save_best=True,
)
```

### 6.2 Output Training

```
========================================================
[MaskablePPO] Device=CUDA  epochs=300  traj/epoch=100
checkpoint_interval=50  save_best=True  log_interval=10
========================================================
 Epoch      Reward    ReUtil   AvgBSLD       ETA
──────────────────────────────────────────────────────
[░░░░░░░░░░░░░░░░] ep=   1  reward=-0.0023  green=0.7812  bsld=0.1643  ETA=45.2m
[█░░░░░░░░░░░░░░░] ep=  10  reward=+0.0234  green=0.8021  bsld=0.1432  ETA=41.3m
[██░░░░░░░░░░░░░░] ep=  20  reward=+0.0841  green=0.8234  bsld=0.1201  ETA=38.1m
[████░░░░░░░░░░░░] ep=  50  reward=+0.1423  green=0.8901  bsld=0.0982  ETA=28.3m
✓ Checkpoint → models/maskable_ppo/checkpoints/epoch_0050
[███████░░░░░░░░░] ep= 100  reward=+0.1687  green=0.9112  bsld=0.0834  ETA=19.5m
✓ Checkpoint → models/maskable_ppo/checkpoints/epoch_0100
[██████████░░░░░░] ep= 150  reward=+0.1723  green=0.9203  bsld=0.0801  ETA=12.4m
✓ Checkpoint → models/maskable_ppo/checkpoints/epoch_0150
[████████████████] ep= 300  reward=+0.1834  green=0.9341  bsld=0.0756  ETA=0s

[MaskablePPO] Done  (42.3 min)
Final model  → models/maskable_ppo
Training log → models/maskable_ppo/train_log.csv
Best model   → models/maskable_ppo/checkpoints/best  (reward=0.1834)
Checkpoints  → models/maskable_ppo/checkpoints  (6 periodic + 1 best)
```

### 6.3 Cấu trúc thư mục sau khi train

```
models/
├── train_config.json               ← Tham số training đã dùng
├── maskable_ppo/
│   ├── actor.pt                    ← Model cuối cùng
│   ├── critic.pt
│   ├── train_log.csv               ← Lịch sử training đầy đủ
│   └── checkpoints/
│       ├── epoch_0050/
│       │   ├── actor.pt
│       │   ├── critic.pt
│       │   └── meta.json           ← {epoch, reward, green, bsld, elapsed}
│       ├── epoch_0100/
│       ├── epoch_0150/
│       ├── epoch_0200/
│       ├── epoch_0250/
│       ├── epoch_0300/
│       └── best/                   ← Model có reward cao nhất
│           ├── actor.pt
│           ├── critic.pt
│           └── meta.json
└── gas_marl/
    ├── actor.pt
    ├── critic.pt
    ├── train_log.csv
    └── checkpoints/
        └── ...
```

### 6.4 Đọc training log

```python
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("models/maskable_ppo/train_log.csv")
print(log.columns.tolist())
# ['epoch', 'avg_reward', 'avg_green', 'avg_bsld', 'elapsed_sec', 'device']

# Vẽ learning curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

log.plot("epoch", "avg_reward", ax=axes[0], title="Reward")
log.plot("epoch", "avg_green",  ax=axes[1], title="ReUtil")
log.plot("epoch", "avg_bsld",   ax=axes[2], title="AvgBSLD")

axes[0].set_ylabel("Avg Reward")
axes[1].set_ylabel("ReUtil")
axes[2].set_ylabel("AvgBSLD")

plt.tight_layout()
plt.savefig("learning_curve.png")
print("Saved learning_curve.png")
```

---

## 7. Checkpoint và Logging

### 7.1 Lưu và Load Checkpoint

```python
# Lưu checkpoint thủ công
agent.save("models/my_checkpoint")
# → Tạo: models/my_checkpoint/actor.pt, critic.pt

# Load checkpoint
from hpcsim.rl.maskable_ppo import MaskablePPOAgent
agent = MaskablePPOAgent(device="cpu")
agent.load("models/maskable_ppo/checkpoints/epoch_0150")
```

### 7.2 Resume Training

```python
# Tiếp tục từ checkpoint cụ thể
agents = run_training(
    algo="maskable_ppo",
    epochs=300,               # Tổng số epoch (không phải epoch còn lại)
    resume_from="models/maskable_ppo/checkpoints/epoch_0150",
    save_dir="models/",
    checkpoint_interval=50,
)
# Training sẽ tiếp tục từ epoch 151 → 300
```

```bash
# Qua CLI
python -m hpcsim.rl.train train \
    --algo maskable_ppo \
    --epochs 300 \
    --resume models/maskable_ppo/checkpoints/epoch_0150 \
    --ckpt-interval 50
```

### 7.3 Kiểm tra metadata checkpoint

```python
import json
from pathlib import Path

# Xem thông tin checkpoint
ckpt_dir = Path("models/maskable_ppo/checkpoints")

for ckpt in sorted(ckpt_dir.glob("epoch_*")):
    meta = json.loads((ckpt / "meta.json").read_text())
    print(f"{ckpt.name:15s}  reward={meta['avg_reward']:.4f}  "
          f"green={meta['avg_green']:.4f}  "
          f"bsld={meta['avg_bsld']:.4f}  "
          f"time={meta['elapsed_sec']/60:.1f}m")

# Best checkpoint
best_meta = json.loads((ckpt_dir / "best" / "meta.json").read_text())
print(f"\nBest: epoch={best_meta['epoch']}  reward={best_meta['avg_reward']:.4f}")
```

**Ví dụ output:**
```
epoch_0050    reward=+0.1423  green=0.8901  bsld=0.0982  time=7.1m
epoch_0100    reward=+0.1687  green=0.9112  bsld=0.0834  time=14.2m
epoch_0150    reward=+0.1723  green=0.9203  bsld=0.0801  time=21.3m
epoch_0200    reward=+0.1798  green=0.9289  bsld=0.0771  time=28.4m
epoch_0250    reward=+0.1821  green=0.9318  bsld=0.0761  time=35.5m
epoch_0300    reward=+0.1834  green=0.9341  bsld=0.0756  time=42.3m

Best: epoch=299  reward=0.1834
```

---

## 8. CLI Đầy đủ

### 8.1 Lệnh `train`

```bash
python -m hpcsim.rl.train train [options]

# Tham số bắt buộc (đều có default):
--algo           all | maskable_ppo | gas_marl    (default: all)

# Tham số training
--epochs         Số epoch huấn luyện               (default: 300)
--traj           Trajectories mỗi epoch             (default: 100)
--cluster        Tên cluster preset                 (default: medium_heterogeneous_gavel)
--duration       Thời gian mỗi episode (giây)       (default: 86400.0)
--eta            Penalty factor η                   (default: 0.005)
--device         cpu | cuda | auto                  (default: auto)
--seed           Random seed                        (default: 42)
--save-dir       Thư mục lưu model                  (default: models)

# Tham số checkpoint & logging
--ckpt-interval  Lưu checkpoint mỗi N epoch (0=tắt) (default: 50)
--resume         Đường dẫn checkpoint để tiếp tục   (default: None)
--log-interval   In log mỗi N epoch                 (default: 10)
--no-save-best   Không lưu best model               (flag)
```

**Ví dụ sử dụng:**

```bash
# Train cơ bản
python -m hpcsim.rl.train train --algo all --epochs 300

# Train với log dày đặc hơn
python -m hpcsim.rl.train train \
    --algo maskable_ppo \
    --epochs 300 --traj 100 \
    --log-interval 5 \
    --ckpt-interval 25 \
    --save-dir models/exp1

# Train nhanh để test
python -m hpcsim.rl.train train \
    --algo maskable_ppo \
    --epochs 50 --traj 50 \
    --cluster tiny_test \
    --log-interval 1

# Resume sau khi bị interrupt
python -m hpcsim.rl.train train \
    --algo gas_marl \
    --epochs 300 \
    --resume models/gas_marl/checkpoints/epoch_0150

# Thay đổi eta để ưu tiên ReUtil hơn
python -m hpcsim.rl.train train \
    --algo gas_marl \
    --epochs 300 \
    --eta 0.001 \
    --save-dir models/green_heavy

# Chạy trên CPU
python -m hpcsim.rl.train train \
    --algo maskable_ppo \
    --epochs 100 --traj 50 \
    --device cpu
```

### 8.2 Lệnh `eval`

```bash
python -m hpcsim.rl.train eval [options]

--model-dir      Thư mục chứa model đã train        (default: models)
--algo           all | maskable_ppo | gas_marl       (default: all)
--episodes       Số episode đánh giá                 (default: 10)
--cluster        Cluster để đánh giá                 (default: medium_heterogeneous_gavel)
--duration       Thời gian mỗi episode               (default: 86400.0)
--output-csv     Lưu kết quả ra CSV                  (default: None)
```

```bash
# Đánh giá cả 2 model
python -m hpcsim.rl.train eval \
    --model-dir models/ \
    --episodes 20 \
    --output-csv eval_results.csv

# Đánh giá trên cluster lớn hơn
python -m hpcsim.rl.train eval \
    --model-dir models/ \
    --algo gas_marl \
    --cluster large_mixed \
    --episodes 10

# Đánh giá model best (không phải final)
python -m hpcsim.rl.train eval \
    --model-dir models/maskable_ppo/checkpoints/best \
    --algo maskable_ppo
```

### 8.3 Lệnh `compare`

```bash
python -m hpcsim.rl.train compare [options]

--model-dir      Thư mục model                       (default: models)
--classical      Danh sách scheduler cổ điển (CSV)   (default: fifo,tiresias,gavel,pollux,chronus)
--rl             Danh sách RL scheduler (CSV)         (default: maskable_ppo,gas_marl)
--cluster        Cluster để benchmark                (default: medium_heterogeneous_gavel)
--num-runs       Số lần chạy mỗi scheduler           (default: 3)
--duration       Thời gian mỗi run                   (default: 3600.0)
--output-csv     Lưu kết quả                         (default: None)
--plot           Lưu biểu đồ                         (default: None)
```

```bash
# So sánh đầy đủ
python -m hpcsim.rl.train compare \
    --classical fifo,sjf,tiresias,gavel,pollux \
    --rl maskable_ppo,gas_marl \
    --model-dir models/ \
    --num-runs 5 \
    --duration 3600 \
    --output-csv comparison.csv \
    --plot comparison.png

# So sánh nhanh
python -m hpcsim.rl.train compare \
    --classical fifo,gavel \
    --rl gas_marl \
    --num-runs 3
```

**Kết quả compare:**

```
Running comparison: ['fifo', 'tiresias', 'gavel', 'pollux', 'maskable_ppo', 'gas_marl'] × 5 runs

Scheduler       | Jobs | AvgJCT  | MedianJCT | p90JCT | AvgBSLD | GPUUtil% | ReUtil% | Energy(Wh)
----------------+------+---------+-----------+--------+---------+----------+---------+-----------
fifo            |   87 |  420.2s |    385.1s | 891.3s |    1.00 |    48.3% |  78.1%  |    45230
tiresias        |  118 |  198.4s |    165.2s | 445.8s |    1.00 |    65.1% |  87.6%  |    52140
gavel           |  134 |  259.3s |    221.0s | 583.2s |    1.00 |    68.9% |  94.6%  |    54890
pollux          |  141 |  201.7s |    178.3s | 412.1s |    1.00 |    71.2% |  95.7%  |    56120
maskable_ppo    |  138 |  215.3s |    192.1s | 431.4s |    1.02 |    70.1% |  92.4%  |    55340
gas_marl        |  143 |  223.8s |    198.7s | 452.3s |    1.03 |    72.3% |  96.8%  |    57210

Renewable Energy Utilization:
  fifo            :  78.1%
  tiresias        :  87.6%
  gavel           :  94.6%
  pollux          :  95.7%
  maskable_ppo    :  92.4%
  gas_marl        :  96.8%  ← Tốt nhất!
```

---

## 9. Đánh giá và So sánh

### 9.1 Đánh giá chi tiết qua API

```python
from hpcsim.rl.train import run_evaluation

results = run_evaluation(
    model_dir="models/",
    algo="all",
    n_episodes=20,
    cluster_config="medium_heterogeneous_gavel",
    sim_duration=86400.0,
    seed=0,
    output_csv="eval.csv",
    verbose=True,
)

for algo, r in results.items():
    print(f"\n{algo}:")
    print(f"  ReUtil  : {r['mean_green']:.4f} ± {r['std_green']:.4f}")
    print(f"  AvgBSLD : {r['mean_bsld']:.4f} ± {r['std_bsld']:.4f}")
    print(f"  Reward  : {r['mean_reward']:.4f} ± {r['std_reward']:.4f}")
```

### 9.2 So sánh các checkpoint

```python
from hpcsim.rl.maskable_ppo import MaskablePPOAgent, MaskablePPOScheduler
from pathlib import Path
import json

# So sánh các checkpoint để chọn epoch tốt nhất
ckpt_dir = Path("models/maskable_ppo/checkpoints")

for ckpt in sorted(ckpt_dir.glob("epoch_*")):
    meta = json.loads((ckpt / "meta.json").read_text())

    # Nhanh chóng đánh giá trên 3 episode
    cluster   = Cluster(CLUSTER_CONFIGS["medium_heterogeneous_gavel"])
    scheduler = MaskablePPOScheduler(cluster, model_dir=str(ckpt))
    # ... run 3 episodes và tính metrics
    print(f"{ckpt.name}: reward={meta['avg_reward']:.4f}")
```

---

## 10. Điều chỉnh Hyperparameter

### 10.1 Cân bằng ReUtil và AvgBSLD (eta)

`eta` là tham số quan trọng nhất ảnh hưởng đến cân bằng mục tiêu:

```
Reward = ReUtil - eta × AvgBSLD

eta = 0.001 → Rất ưu tiên ReUtil, chấp nhận BSLD cao
eta = 0.005 → Cân bằng (default, khuyến nghị từ paper)
eta = 0.01  → Ưu tiên giảm BSLD, ReUtil có thể giảm
eta = 0.05  → Rất ưu tiên giảm BSLD
```

```bash
# Thử nhiều giá trị eta
for eta in 0.001 0.003 0.005 0.01; do
    python -m hpcsim.rl.train train \
        --algo maskable_ppo \
        --epochs 100 --eta $eta \
        --save-dir "models/eta_${eta}" \
        --log-interval 20
done
```

### 10.2 Traj num và Episode length

```
traj_num lớn hơn → gradient estimate chính xác hơn → stable training
                 → nhưng mỗi epoch chậm hơn

seq_len lớn hơn  → agent học trên horizon dài hơn
                 → nhưng reward sparser hơn (khó học hơn)

Khuyến nghị bắt đầu: traj_num=50, seq_len=128 → kiểm tra hội tụ
Sau đó tăng dần:    traj_num=100, seq_len=256
```

### 10.3 Learning rate

```bash
# Nếu training không ổn định (reward dao động mạnh):
#   → Giảm learning rate
# Nếu training quá chậm hội tụ:
#   → Tăng learning rate hoặc traj_num

# Hiện tại lr được hard-code trong networks.py
# Để thay đổi, sửa trực tiếp trong MaskablePPOAgent.__init__():
# self.actor_optim  = Adam(lr=3e-4)
# self.critic_optim = Adam(lr=1e-3)
```

### 10.4 Điều chỉnh theo cluster size

| Cluster size | Khuyến nghị |
|-------------|-------------|
| Nhỏ (<32 GPU) | epochs=150, traj=50, seq_len=128 |
| Vừa (32–128 GPU) | epochs=300, traj=100, seq_len=256 *(default)* |
| Lớn (>128 GPU) | epochs=500, traj=150, seq_len=512 |

---

## 11. Phát triển RL Scheduler Tùy chỉnh

### 11.1 Mở rộng state space

```python
from hpcsim.rl.env import HPCGreenEnv, EnvConfig

class ExtendedGreenEnv(HPCGreenEnv):
    """
    Mở rộng môi trường với thêm thông tin người dùng (fairness).
    """

    def _get_job_features(self, job, current_time):
        # Feature cơ bản từ parent
        base = super()._get_job_features(job, current_time)

        # Thêm fairness feature: job của user đã chờ lâu → ưu tiên
        user_total_wait = self._user_wait_times.get(job.user_id, 0.0)
        fairness_score  = min(user_total_wait / 3600, 1.0)   # Normalize

        # Mở rộng feature vector
        return base + [fairness_score]

    def reset(self):
        self._user_wait_times = {}
        return super().reset()
```

### 11.2 Custom reward

```python
class GreenFairnessEnv(HPCGreenEnv):
    """
    Reward kết hợp 3 mục tiêu: ReUtil, AvgBSLD, và Fairness.
    """

    def __init__(self, config, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__(config)
        self.alpha = alpha   # Trọng số ReUtil
        self.beta  = beta    # Trọng số AvgBSLD
        self.gamma = gamma   # Trọng số Fairness

    def _compute_reward(self, metrics):
        reutil   = metrics.renewable_energy_utilization()
        avg_bsld = metrics.avg_bsld()
        fairness = metrics.fairness_index()

        # Multi-objective reward
        reward = (self.alpha * reutil
                  - self.beta * (avg_bsld - 1.0)   # BSLD >= 1, perfect = 1
                  + self.gamma * fairness)
        return reward
```

### 11.3 Wrapper cho thuật toán RL bên ngoài (Stable Baselines 3)

```python
import gym
import numpy as np

class HPCGymWrapper(gym.Env):
    """
    Wrapper biến HPCGreenEnv thành Gym environment chuẩn
    để dùng với Stable Baselines 3.
    """

    def __init__(self, env_config=None):
        super().__init__()
        from hpcsim.rl.env import HPCGreenEnv, EnvConfig
        self._env = HPCGreenEnv(env_config or EnvConfig())

        n_features = self._env.obs_size
        n_actions  = self._env.action_space_size

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        obs, reward, done, bsld, *_ = self._env.step(action, 0)
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass


# Sử dụng với PPO từ Stable Baselines 3
from stable_baselines3 import PPO

env   = HPCGymWrapper()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("sb3_ppo_hpc")
```

---

## Tóm tắt Quick Reference

| Tác vụ | Lệnh |
|--------|------|
| Train cơ bản (cả 2 algo) | `python -m hpcsim.rl.train train --algo all --epochs 300` |
| Train nhanh để test | `python -m hpcsim.rl.train train --algo maskable_ppo --epochs 50 --traj 50 --log-interval 1` |
| Resume từ checkpoint | `python -m hpcsim.rl.train train --resume models/maskable_ppo/checkpoints/epoch_0150` |
| Đánh giá model | `python -m hpcsim.rl.train eval --model-dir models/ --episodes 20` |
| So sánh tổng thể | `python -m hpcsim.rl.train compare --classical fifo,gavel,pollux --rl maskable_ppo,gas_marl` |
| Xem learning curve | `pandas.read_csv("models/maskable_ppo/train_log.csv").plot(...)` |
