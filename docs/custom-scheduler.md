# Hướng dẫn Phát triển Custom Scheduler

Tài liệu này hướng dẫn cách viết một bộ lập lịch tùy chỉnh trong HPCSim để phục vụ nghiên cứu — từ scheduler đơn giản nhất đến các thuật toán phức tạp như Backfilling, Green Backfilling, và RL-based scheduling.

---

## Mục lục

1. [Interface BaseScheduler](#1-interface-basescheduler)
2. [Viết Scheduler Đầu tiên](#2-viết-scheduler-đầu-tiên)
3. [Thuật toán EASY Backfilling](#3-thuật-toán-easy-backfilling)
4. [Green Backfilling](#4-green-backfilling)
5. [Tích hợp RL Scheduler](#5-tích-hợp-rl-scheduler)
6. [Đăng ký và So sánh](#6-đăng-ký-và-so-sánh)
7. [Debugging và Testing](#7-debugging-và-testing)
8. [Ví dụ Nâng cao](#8-ví-dụ-nâng-cao)

---

## 1. Interface BaseScheduler

Tất cả scheduler phải kế thừa `BaseScheduler` và implement phương thức `schedule()`:

```python
from hpcsim.scheduler.schedulers import BaseScheduler, SchedulingDecision
from hpcsim.cluster.cluster import Cluster

class BaseScheduler:
    """
    Interface cơ sở cho tất cả scheduler trong HPCSim.
    Scheduler chỉ cần implement một phương thức: schedule().
    """

    def __init__(self, name: str, cluster: Cluster):
        self.name    = name
        self.cluster = cluster

    def schedule(
        self,
        pending: list,        # Danh sách Job đang chờ trong queue
        running: list,        # Danh sách Job đang chạy
        current_time: float,  # Thời gian hiện tại (giây)
    ) -> SchedulingDecision:
        """
        Quyết định lập lịch: job nào chạy trên GPU nào.

        Returns:
            SchedulingDecision chứa danh sách (job, gpu_ids) để launch
        """
        raise NotImplementedError
```

### SchedulingDecision

```python
from hpcsim.scheduler.schedulers import SchedulingDecision

decision = SchedulingDecision()

# Thêm một job vào lịch
decision.add(job, gpu_ids=[0, 1, 2, 3])

# Yêu cầu preempt (dừng) một job đang chạy
decision.preempt(job_id=5)

# Trì hoãn một job (cho GAS-MARL)
decision.delay(job, until=current_time + 300)

# Kiểm tra
len(decision.assignments)   # Số job sẽ launch
len(decision.preemptions)   # Số job sẽ bị dừng
```

### GPU Helper Methods

```python
class BaseScheduler:
    # Tìm GPU phù hợp cho job
    def _find_gpus(self, job, prefer_consolidated=True) -> list[int]:
        """Tìm danh sách GPU khả dụng phù hợp với job.num_gpus và gpu_type_pref."""

    def _gpu_count_for_job(self, job) -> int:
        """Số GPU cần cấp phát cho job."""

    def _estimate_job_power(self, job) -> float:
        """Ước tính công suất tiêu thụ (W) của job."""
```

---

## 2. Viết Scheduler Đầu tiên

### 2.1 Scheduler Tối giản (FIFO)

```python
# my_schedulers.py
from hpcsim.scheduler.schedulers import BaseScheduler, SchedulingDecision

class MyFIFOScheduler(BaseScheduler):
    """
    First-In-First-Out: chạy job theo thứ tự đến.
    Đây là scheduler đơn giản nhất, dùng làm baseline.
    """

    def __init__(self, cluster):
        super().__init__("MyFIFO", cluster)

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()

        # Sắp xếp theo thời gian submit (FIFO)
        sorted_jobs = sorted(pending, key=lambda j: j.submit_time)

        for job in sorted_jobs:
            gpu_ids = self._find_gpus(job)
            if gpu_ids:
                decision.add(job, gpu_ids)
            # Nếu không tìm được GPU → bỏ qua (không block jobs sau)

        return decision
```

### 2.2 Chạy thử

```python
from hpcsim import Cluster, CLUSTER_CONFIGS, SimulationEngine, MetricsCollector
from hpcsim.workload.generator import WorkloadGenerator, WorkloadConfig
from my_schedulers import MyFIFOScheduler

cluster   = Cluster(CLUSTER_CONFIGS["tiny_test"])
jobs      = WorkloadGenerator(WorkloadConfig(duration=3600)).generate()
scheduler = MyFIFOScheduler(cluster)
metrics   = MetricsCollector()

engine = SimulationEngine(cluster, scheduler, jobs, metrics, max_sim_time=3600)
engine.run()

s = metrics.summary()
print(f"Jobs done: {s['jobs_completed']}, Avg JCT: {s['avg_jct']:.1f}s")
```

### 2.3 Scheduler Ưu tiên (Priority)

```python
class PriorityScheduler(BaseScheduler):
    """
    Lập lịch theo priority field của job.
    priority cao hơn = chạy trước.
    """

    def __init__(self, cluster, starvation_threshold=600.0):
        super().__init__("Priority", cluster)
        self.starvation_threshold = starvation_threshold  # giây

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()

        # Áp dụng anti-starvation: job chờ quá lâu được tăng priority
        def effective_priority(job):
            wait = current_time - job.submit_time
            starvation_boost = wait / self.starvation_threshold
            return job.priority + starvation_boost

        sorted_jobs = sorted(pending, key=effective_priority, reverse=True)

        for job in sorted_jobs:
            gpu_ids = self._find_gpus(job)
            if gpu_ids:
                decision.add(job, gpu_ids)

        return decision
```

---

## 3. Thuật toán EASY Backfilling

**Backfilling** lấp đầy GPU khi job ưu tiên nhất đang chờ tài nguyên. Ý tưởng: chạy các job nhỏ/ngắn hơn trong khoảng thời gian trống mà **không làm trễ** job đứng đầu hàng.

### 3.1 Nguyên lý EASY Backfilling

```
Queue (theo FIFO):  Job-A(8 GPU, 2h), Job-B(2 GPU, 30m), Job-C(4 GPU, 1h)
Hiện tại:          6 GPU đang bận; 2 GPU rảnh

EASY Backfilling:
  1. Job-A cần 8 GPU → không chạy được ngay
  2. Tính "reservation time" cho Job-A:
     = thời điểm sớm nhất 8 GPU có thể giải phóng
  3. Backfill: Job-B cần 2 GPU, có thể xong trước reservation_time
     → chạy Job-B ngay với 2 GPU rảnh
  4. Job-C cần 4 GPU → không đủ GPU rảnh → bỏ qua
```

### 3.2 Implementation

```python
import heapq

class EASYBackfillScheduler(BaseScheduler):
    """
    EASY Backfilling scheduler.
    Bảo đảm job đứng đầu queue không bị trễ do backfilling.
    """

    def __init__(self, cluster):
        super().__init__("EASY-Backfill", cluster)
        self._reservation = None   # (job, earliest_start_time)

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        if not pending:
            return decision

        free_gpus = self.cluster.free_gpu_count()

        # === Bước 1: Cố chạy job đầu hàng ===
        head_job = pending[0]
        head_gpus = self._find_gpus(head_job)

        if head_gpus:
            # Có đủ GPU → chạy ngay
            decision.add(head_job, head_gpus)
            free_gpus -= len(head_gpus)
            remaining = pending[1:]
        else:
            # Không đủ → tính reservation time
            self._reservation = self._compute_reservation(
                head_job, running, current_time
            )
            remaining = pending[1:]

        # === Bước 2: Backfill các job còn lại ===
        if self._reservation:
            reserved_job, reserved_start = self._reservation
            for job in remaining:
                if job is reserved_job:
                    continue
                if self._can_backfill(job, reserved_start, current_time):
                    gpus = self._find_gpus(job)
                    if gpus and len(gpus) <= free_gpus:
                        decision.add(job, gpus)
                        free_gpus -= len(gpus)
        else:
            # Không có reservation → greedy fill
            for job in remaining:
                gpus = self._find_gpus(job)
                if gpus and len(gpus) <= free_gpus:
                    decision.add(job, gpus)
                    free_gpus -= len(gpus)

        return decision

    def _compute_reservation(self, head_job, running, current_time):
        """
        Tính thời điểm sớm nhất có đủ GPU cho head_job.
        Dựa trên runtime estimate của các job đang chạy.
        """
        needed = head_job.num_gpus
        free   = self.cluster.free_gpu_count()

        if free >= needed:
            return (head_job, current_time)

        # Sắp xếp job đang chạy theo thời điểm kết thúc ước tính
        finishing_order = sorted(
            running,
            key=lambda j: (j.start_time or current_time)
                          + getattr(j, 'runtime_estimate', j.runtime)
        )

        accumulated_gpus = free
        for job in finishing_order:
            accumulated_gpus += len(getattr(job, 'allocated_gpus', []) or [1])
            finish_est = ((job.start_time or current_time)
                          + getattr(job, 'runtime_estimate', job.runtime))
            if accumulated_gpus >= needed:
                return (head_job, finish_est)

        # Worst case: sau khi tất cả job xong
        if running:
            last = max(running,
                       key=lambda j: (j.start_time or current_time)
                                     + getattr(j, 'runtime_estimate', j.runtime))
            return (head_job, (last.start_time or current_time)
                              + getattr(last, 'runtime_estimate', last.runtime))
        return None

    def _can_backfill(self, job, reserved_start, current_time):
        """
        Backfill job nếu nó có thể xong trước reserved_start.
        (Không làm trễ job đứng đầu.)
        """
        if reserved_start is None:
            return True
        runtime_est = getattr(job, 'runtime_estimate', job.runtime)
        expected_finish = current_time + runtime_est
        return expected_finish <= reserved_start
```

---

## 4. Green Backfilling

**Green Backfilling** mở rộng Backfilling bằng cách ưu tiên chạy job khi năng lượng tái tạo đang dồi dào, đồng thời trì hoãn job tiêu tốn nhiều điện đến khi có điện xanh.

### 4.1 Nguyên lý

```
Năng lượng xanh dự báo (kW):
  Hiện tại (10h): 15 kW  ← Dồi dào!
  11h:            18 kW
  12h:            20 kW  ← Đỉnh
  13h:             8 kW  ← Giảm
  14h:             2 kW

Green Backfilling quyết định:
  → Job tiêu thụ nhiều điện (8 GPU × 300W = 2400W)
     → trì hoãn đến 12h khi có 20 kW xanh
  → Job nhỏ (1 GPU × 150W = 150W)
     → chạy ngay (không cần chờ)
  → Tính toán "green score" = green_available / job_power
     → ưu tiên job có green score cao
```

### 4.2 Implementation

```python
from hpcsim.energy.renewable import RenewableEnergyModule

class GreenBackfillScheduler(BaseScheduler):
    """
    Green Backfilling: kết hợp backfilling với ưu tiên điện tái tạo.

    Chiến lược:
    1. Luôn đảm bảo job đứng đầu không bị delay
    2. Backfill: ưu tiên job có green_score cao (ít brown energy)
    3. Delay job tốn điện khi sắp có peak điện xanh
    """

    DELAY_THRESHOLD_W = 500.0    # Trì hoãn job tiêu > 500W nếu green sắp có
    BROWN_ENERGY_MAX  = 50_000   # Joule — ngưỡng để xét "green job"

    def __init__(self, cluster, renewable_module=None):
        super().__init__("GreenBackfill", cluster)
        self.re = renewable_module    # RenewableEnergyModule (tùy chọn)
        self._delayed_jobs = {}       # job_id → delay_until

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        if not pending:
            return decision

        # Lấy công suất xanh hiện tại
        green_w = (self.re.available_power_watts(current_time)
                   if self.re else 0.0)

        # Lọc bỏ job đang trong delay
        active_pending = [
            j for j in pending
            if current_time >= self._delayed_jobs.get(j.job_id, 0.0)
        ]

        # === Job đầu hàng ===
        head_job  = active_pending[0] if active_pending else pending[0]
        head_gpus = self._find_gpus(head_job)
        free_gpus = self.cluster.free_gpu_count()

        if head_gpus:
            decision.add(head_job, head_gpus)
            free_gpus -= len(head_gpus)
            active_remaining = [j for j in active_pending[1:] if j is not head_job]
        else:
            reservation = self._compute_reservation(head_job, running, current_time)
            active_remaining = active_pending[1:]

            # === Green Backfill ===
            # Ưu tiên theo: (1) có điện xanh đủ, (2) job nhỏ, (3) job ngắn
            def green_score(job):
                power    = self._estimate_job_power(job)
                runtime  = getattr(job, 'runtime_estimate', job.runtime)
                brown_j  = max(0.0, power - green_w) * runtime    # Brown energy (J)
                if brown_j < self.BROWN_ENERGY_MAX:
                    return (2, -power)   # Tier 1: low brown energy
                elif green_w > power * 0.5:
                    return (1, -power)   # Tier 2: 50%+ covered by green
                else:
                    return (0, -power)   # Tier 3: mostly brown

            sorted_remaining = sorted(active_remaining, key=green_score, reverse=True)

            for job in sorted_remaining:
                # Kiểm tra có backfill được không (không làm trễ head)
                if reservation and not self._can_backfill(job, reservation[1], current_time):
                    # Nếu job tốn nhiều điện và sắp có peak xanh → delay
                    job_power = self._estimate_job_power(job)
                    if job_power > self.DELAY_THRESHOLD_W:
                        peak_time = self._find_green_peak(current_time, window_h=4)
                        if peak_time:
                            self._delayed_jobs[job.job_id] = peak_time
                    continue

                gpus = self._find_gpus(job)
                if gpus and len(gpus) <= free_gpus:
                    decision.add(job, gpus)
                    free_gpus -= len(gpus)

        return decision

    def _find_green_peak(self, current_time, window_h=4):
        """Tìm thời điểm điện xanh đạt đỉnh trong cửa sổ time_window."""
        if not self.re:
            return None

        best_t     = None
        best_power = 0.0
        step       = 900   # Kiểm tra mỗi 15 phút

        for dt in range(0, int(window_h * 3600), step):
            t = current_time + dt
            p = self.re.available_power_watts(t)
            if p > best_power:
                best_power = p
                best_t     = t

        # Chỉ delay nếu đỉnh xanh cao hơn hiện tại đáng kể
        current_power = self.re.available_power_watts(current_time)
        if best_power > current_power * 1.3:
            return best_t
        return None

    def _can_backfill(self, job, reserved_start, current_time):
        runtime_est = getattr(job, 'runtime_estimate', job.runtime)
        return current_time + runtime_est <= reserved_start

    def _compute_reservation(self, head_job, running, current_time):
        needed = head_job.num_gpus
        free   = self.cluster.free_gpu_count()
        if free >= needed:
            return (head_job, current_time)
        finishing = sorted(
            running,
            key=lambda j: (j.start_time or current_time)
                          + getattr(j, 'runtime_estimate', j.runtime)
        )
        acc = free
        for job in finishing:
            acc += len(getattr(job, 'allocated_gpus', []) or [1])
            t   = ((job.start_time or current_time)
                   + getattr(job, 'runtime_estimate', job.runtime))
            if acc >= needed:
                return (head_job, t)
        return None
```

### 4.3 Sử dụng GreenBackfillScheduler

```python
from hpcsim.energy.renewable import RenewableEnergyModule
from my_schedulers import GreenBackfillScheduler

# Tạo renewable module
re_module = RenewableEnergyModule(
    total_gpus=112,
    sim_duration=86400,
    seed=42,
)

# Tạo scheduler với renewable module
scheduler = GreenBackfillScheduler(cluster, renewable_module=re_module)

# Chạy
engine = SimulationEngine(
    cluster, scheduler, jobs, metrics,
    max_sim_time=86400,
    renewable_config={"total_gpus": 112},
)
engine.run()

s = metrics.summary()
print(f"ReUtil   : {s['renewable_energy_utilization']:.1%}")
print(f"AvgBSLD  : {s['avg_bsld']:.3f}")
```

---

## 5. Tích hợp RL Scheduler

Các RL scheduler (MaskablePPO, GAS-MARL) cũng implement `BaseScheduler`, chỉ khác ở chỗ họ dùng neural network để chọn hành động.

### 5.1 Khung xương RL Scheduler

```python
import torch
import numpy as np
from hpcsim.scheduler.schedulers import BaseScheduler, SchedulingDecision

class MyRLScheduler(BaseScheduler):
    """
    Khung xương cho RL-based scheduler.
    Neural network quyết định job nào chạy tiếp theo.
    """

    def __init__(self, cluster, model_path=None):
        super().__init__("MyRL", cluster)
        self.model = self._build_model()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def _build_model(self):
        """Xây dựng policy network."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(64 * 8, 256),   # 64 jobs × 8 features
            nn.ReLU(),
            nn.Linear(256, 64),       # Output: score cho mỗi job
        )

    def _get_state(self, pending, running, current_time):
        """Chuyển trạng thái simulation thành tensor đầu vào cho model."""
        features = np.zeros((64, 8))  # Max 64 jobs, 8 features mỗi job
        for i, job in enumerate(pending[:64]):
            features[i] = [
                (current_time - job.submit_time) / 3600,    # Wait time (h)
                job.num_gpus / 16,                           # GPU normalized
                getattr(job, 'runtime_estimate', job.runtime) / 7200,
                self._estimate_job_power(job) / 1000,
                1.0 if job.gpu_type_pref else 0.0,
                getattr(job, 'priority', 0.0) / 10,
                1.0,                                         # Is pending
                0.0,
            ]
        return torch.FloatTensor(features.flatten()).unsqueeze(0)

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()
        if not pending:
            return decision

        # Lấy state tensor
        state = self._get_state(pending, running, current_time)

        # Action mask: chỉ chọn job có GPU khả dụng
        mask = np.zeros(64)
        for i, job in enumerate(pending[:64]):
            if self._find_gpus(job):
                mask[i] = 1.0

        # Inference
        with torch.no_grad():
            scores = self.model(state).squeeze(0).numpy()
            scores[mask == 0] = float('-inf')   # Mask invalid
            action = int(np.argmax(scores[:len(pending)]))

        # Apply action
        job     = pending[action]
        gpu_ids = self._find_gpus(job)
        if gpu_ids:
            decision.add(job, gpu_ids)

        # Greedy fill phần còn lại
        for j in pending:
            if j is job:
                continue
            gids = self._find_gpus(j)
            if gids:
                decision.add(j, gids)

        return decision
```

### 5.2 Sử dụng MaskablePPO và GAS-MARL có sẵn

```python
from hpcsim.rl.maskable_ppo import MaskablePPOScheduler
from hpcsim.rl.gas_marl import GASMARLScheduler

# Load model đã train
ppo_sched = MaskablePPOScheduler(
    cluster,
    model_dir="models/maskable_ppo",  # Thư mục chứa actor.pt, critic.pt
    device="cpu",
)

gas_sched = GASMARLScheduler(
    cluster,
    model_dir="models/gas_marl",
    device="cpu",
)

# Dùng như mọi scheduler khác
engine = SimulationEngine(cluster, ppo_sched, jobs, metrics, max_sim_time=3600)
engine.run()
```

---

## 6. Đăng ký và So sánh

### 6.1 Đăng ký scheduler

```python
from hpcsim.scheduler.schedulers import register, BaseScheduler, SchedulingDecision

@register                          # Decorator này đăng ký vào registry
class MyCustomScheduler(BaseScheduler):
    name = "my_custom"             # Tên để gọi qua create_scheduler()

    def __init__(self, cluster):
        super().__init__("MyCustom", cluster)

    def schedule(self, pending, running, current_time):
        ...
```

Sau khi đăng ký:

```python
from hpcsim import create_scheduler

# Import module để trigger đăng ký
import my_schedulers

sched = create_scheduler("my_custom", cluster)
```

### 6.2 So sánh với BenchmarkRunner

```python
from hpcsim.benchmark.runner import BenchmarkRunner, BenchmarkConfig
import my_schedulers  # Trigger registration

cfg = BenchmarkConfig(
    schedulers=[
        "fifo",
        "tiresias",
        "my_custom",       # Scheduler của bạn
        "green_backfill",  # Nếu đã register
    ],
    cluster_config="medium_heterogeneous_gavel",
    num_runs=5,
    sim_duration=3600,
)

runner = BenchmarkRunner(cfg)
runner.print_table(runner.run())
```

### 6.3 Đăng ký scheduler với factory function

```python
from hpcsim.scheduler.schedulers import register_factory

# Đăng ký scheduler cần parameters đặc biệt
register_factory(
    "green_backfill_solar200",
    lambda cluster: GreenBackfillScheduler(
        cluster,
        renewable_module=RenewableEnergyModule(
            total_gpus=cluster.total_gpu_count(),
            sim_duration=86400,
            solar_area_m2=200,
        )
    )
)
```

---

## 7. Debugging và Testing

### 7.1 Unit test scheduler

```python
import pytest
from hpcsim import Cluster, CLUSTER_CONFIGS
from hpcsim.workload.generator import Job
from my_schedulers import MyCustomScheduler

def make_job(job_id, submit_time, num_gpus, runtime):
    return Job(
        job_id=job_id,
        submit_time=submit_time,
        num_gpus=num_gpus,
        runtime=runtime,
        runtime_estimate=runtime * 1.1,
    )

def test_basic_scheduling():
    cluster   = Cluster(CLUSTER_CONFIGS["tiny_test"])  # 8 GPUs
    scheduler = MyCustomScheduler(cluster)

    jobs = [
        make_job(0, 0.0, 2, 300),
        make_job(1, 1.0, 4, 600),
        make_job(2, 2.0, 2, 150),
    ]

    decision = scheduler.schedule(jobs, [], current_time=5.0)

    # Phải schedule ít nhất 1 job
    assert len(decision.assignments) > 0

    # Mỗi job phải có đủ GPU
    for job, gpu_ids in decision.assignments:
        assert len(gpu_ids) == job.num_gpus

    # Không được schedule quá nhiều job (vượt quá GPU)
    total_gpus_used = sum(len(g) for _, g in decision.assignments)
    assert total_gpus_used <= cluster.total_gpu_count()

def test_no_double_allocation():
    """Kiểm tra không cấp phát cùng GPU cho 2 job."""
    cluster   = Cluster(CLUSTER_CONFIGS["tiny_test"])
    scheduler = MyCustomScheduler(cluster)
    jobs      = [make_job(i, 0.0, 4, 300) for i in range(4)]

    decision = scheduler.schedule(jobs, [], current_time=0.0)

    all_gpu_ids = []
    for _, gpu_ids in decision.assignments:
        all_gpu_ids.extend(gpu_ids)

    # Không có GPU nào bị cấp phát 2 lần
    assert len(all_gpu_ids) == len(set(all_gpu_ids)), "Duplicate GPU allocation!"
```

### 7.2 Verbose logging trong scheduler

```python
class DebugScheduler(BaseScheduler):
    def __init__(self, cluster, debug=False):
        super().__init__("Debug", cluster)
        self.debug = debug

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()

        if self.debug:
            print(f"\n[t={current_time:.0f}s] Scheduling called")
            print(f"  Pending: {len(pending)} jobs")
            print(f"  Running: {len(running)} jobs")
            print(f"  Free GPU: {self.cluster.free_gpu_count()}")
            for j in pending[:5]:
                wait = current_time - j.submit_time
                print(f"    Job {j.job_id}: {j.num_gpus} GPUs, "
                      f"wait={wait:.0f}s, runtime={j.runtime:.0f}s")

        # ... scheduling logic ...

        if self.debug:
            print(f"  Decision: schedule {len(decision.assignments)} jobs")
            for job, gpus in decision.assignments:
                print(f"    → Job {job.job_id} on GPUs {gpus}")

        return decision
```

### 7.3 Trace file

```python
import csv

class TracingScheduler(BaseScheduler):
    """Scheduler bao ngoài để ghi trace scheduling decisions."""

    def __init__(self, inner_scheduler, trace_file="trace.csv"):
        super().__init__(f"Tracing({inner_scheduler.name})",
                         inner_scheduler.cluster)
        self.inner    = inner_scheduler
        self._file    = open(trace_file, "w", newline="")
        self._writer  = csv.writer(self._file)
        self._writer.writerow(["time", "action", "job_id",
                                "num_gpus", "gpu_ids", "wait_time"])

    def schedule(self, pending, running, current_time):
        decision = self.inner.schedule(pending, running, current_time)
        for job, gpu_ids in decision.assignments:
            wait = current_time - job.submit_time
            self._writer.writerow([
                f"{current_time:.1f}", "SCHEDULE", job.job_id,
                job.num_gpus, str(gpu_ids), f"{wait:.1f}"
            ])
        return decision

    def close(self):
        self._file.close()
```

---

## 8. Ví dụ Nâng cao

### 8.1 Scheduler nhận biết heterogeneity

```python
class HeterogeneityAwareScheduler(BaseScheduler):
    """
    Ưu tiên cấp phát GPU phù hợp nhất với yêu cầu của job.
    V100 cho job FP16/training; T4 cho inference; K80 cho jobs nhỏ.
    """

    # Điểm hiệu năng tương đối (V100 = 1.0)
    GPU_PERFORMANCE = {"A100": 2.5, "V100": 1.0, "T4": 0.65, "K80": 0.30}

    def __init__(self, cluster):
        super().__init__("HeteroAware", cluster)

    def _score_gpu_type(self, job, gpu_type):
        """Tính điểm phù hợp giữa job và loại GPU."""
        perf = self.GPU_PERFORMANCE.get(gpu_type, 0.5)

        if job.gpu_type_pref:
            if gpu_type in job.gpu_type_pref:
                return perf * 2.0   # Khớp yêu cầu → bonus
            else:
                return perf * 0.5   # Không khớp → penalty
        return perf                 # Không có preference → dùng perf

    def schedule(self, pending, running, current_time):
        decision  = SchedulingDecision()
        free_gpus = self.cluster.free_gpu_count()

        # Sắp xếp job theo priority
        for job in sorted(pending, key=lambda j: j.priority, reverse=True):
            if free_gpus <= 0:
                break

            # Tìm GPU tốt nhất cho job này
            best_gpus  = None
            best_score = -1.0

            for gpu_type in self.cluster.available_gpu_types():
                gpus = self._find_gpus_of_type(job, gpu_type)
                if gpus:
                    score = self._score_gpu_type(job, gpu_type)
                    if score > best_score:
                        best_score = score
                        best_gpus  = gpus

            if best_gpus:
                decision.add(job, best_gpus)
                free_gpus -= len(best_gpus)

        return decision

    def _find_gpus_of_type(self, job, gpu_type):
        """Tìm GPU của loại cụ thể cho job."""
        available = [
            gid for gid, g in self.cluster.gpus.items()
            if g.gpu_type == gpu_type and g.is_free()
        ]
        if len(available) >= job.num_gpus:
            return available[:job.num_gpus]
        return []
```

### 8.2 Scheduler thực nghiệm: Kết hợp nhiều chiến lược

```python
class HybridScheduler(BaseScheduler):
    """
    Hybrid scheduler: chọn chiến lược dựa trên trạng thái cluster.
    - Cluster tải cao (util > 80%): dùng FIFO strict
    - Cluster tải trung bình: dùng Backfill + Green priority
    - Cluster nhàn rỗi: dùng SJF để clear queue
    """

    HIGH_LOAD_THRESHOLD  = 0.80
    LOW_LOAD_THRESHOLD   = 0.30

    def __init__(self, cluster):
        super().__init__("Hybrid", cluster)

    def schedule(self, pending, running, current_time):
        total = self.cluster.total_gpu_count()
        free  = self.cluster.free_gpu_count()
        util  = 1.0 - free / total if total > 0 else 0.0

        if util > self.HIGH_LOAD_THRESHOLD:
            return self._fifo_schedule(pending, current_time)
        elif util > self.LOW_LOAD_THRESHOLD:
            return self._backfill_schedule(pending, running, current_time)
        else:
            return self._sjf_schedule(pending, current_time)

    def _fifo_schedule(self, pending, current_time):
        decision  = SchedulingDecision()
        free_gpus = self.cluster.free_gpu_count()
        for job in sorted(pending, key=lambda j: j.submit_time):
            gpus = self._find_gpus(job)
            if gpus and len(gpus) <= free_gpus:
                decision.add(job, gpus)
                free_gpus -= len(gpus)
        return decision

    def _sjf_schedule(self, pending, current_time):
        decision  = SchedulingDecision()
        free_gpus = self.cluster.free_gpu_count()
        for job in sorted(pending, key=lambda j: getattr(j, 'runtime_estimate', j.runtime)):
            gpus = self._find_gpus(job)
            if gpus and len(gpus) <= free_gpus:
                decision.add(job, gpus)
                free_gpus -= len(gpus)
        return decision

    def _backfill_schedule(self, pending, running, current_time):
        # Tái sử dụng logic EASYBackfill
        from my_schedulers import EASYBackfillScheduler
        tmp = EASYBackfillScheduler(self.cluster)
        return tmp.schedule(pending, running, current_time)
```

---

## Checklist khi viết Custom Scheduler

Trước khi chạy benchmark, kiểm tra:

- [ ] Kế thừa `BaseScheduler` và implement `schedule()`
- [ ] Trả về `SchedulingDecision` hợp lệ
- [ ] Không cấp phát cùng GPU cho 2 job (dùng `_find_gpus()` helper)
- [ ] Không sửa trực tiếp danh sách `pending` hay `running`
- [ ] Xử lý trường hợp `pending` rỗng
- [ ] Xử lý trường hợp không đủ GPU (không crash)
- [ ] Đã viết unit test cơ bản
- [ ] Đã đăng ký với `@register` hoặc `register_factory`
