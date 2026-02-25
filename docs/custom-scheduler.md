# Viết Scheduler Tùy Chỉnh

## Kế thừa BaseScheduler

```python
from hpcsim.scheduler.schedulers import (
    BaseScheduler, SchedulingDecision, register,
)
from hpcsim.workload.job import AnyJob, ResourceType

@register   # ← tự động đăng ký vào SCHEDULER_REGISTRY
class MyScheduler(BaseScheduler):
    name = "my_scheduler"

    def schedule(
        self,
        pending: list[AnyJob],
        running: list[AnyJob],
        current_time: float,
    ) -> SchedulingDecision:
        decision = SchedulingDecision()

        for job in pending:
            # Dùng _find_resources() — tự động xử lý GPU/CPU/MIG/Hybrid
            res = self._find_resources(job)
            if res:
                gpus, migs, cpus = res
                decision.add(job, gpus, migs, cpus)

        return decision
```

## Xử lý từng loại tài nguyên riêng

```python
def schedule(self, pending, running, current_time):
    decision = SchedulingDecision()

    for job in pending:
        rt = getattr(job, "resource_type", ResourceType.GPU)

        if rt == ResourceType.CPU:
            # Chỉ cần CPU cores
            cpu_alloc = self._find_cpus(job)
            if cpu_alloc:
                decision.add_cpu(job, cpu_alloc)

        elif rt == ResourceType.MIG:
            # Chỉ cần MIG slices
            mig_ids = self._find_mig(job)
            if mig_ids:
                decision.add_mig(job, mig_ids)

        elif rt == ResourceType.CPU_GPU:
            # Cần cả GPU lẫn CPU — phải có cả hai mới schedule
            gpu_ids = self._find_gpus(job)
            cpu_alloc = self._find_cpus(job)
            if gpu_ids and cpu_alloc:
                decision.add_hybrid(job, gpu_ids, cpu_alloc)

        else:
            # GPU job thông thường
            gpu_ids = self._find_gpus(job)
            if gpu_ids:
                decision.add(job, gpu_ids)

    return decision
```

## Helper methods có sẵn

```python
# Tìm GPU tốt nhất (consolidated trước, scattered fallback)
gpu_ids = self._find_gpus(job, prefer_consolidated=True)

# Tìm MIG slices theo profile
mig_ids = self._find_mig(job, profile=MIGProfile.G1_10GB)

# Tìm CPU cores (NUMA-friendly by default)
cpu_alloc = self._find_cpus(job, prefer_consolidated=True)
# Trả về: ["n_cpu_compute_0_cpu0:16", "n_cpu_compute_0_cpu1:16"]

# Tìm tất cả tài nguyên phù hợp với job type
res = self._find_resources(job)   # → (gpu_ids, mig_ids, cpu_alloc) | None
```

## Sử dụng trong CLI

```bash
# Sau khi đăng ký với @register, dùng ngay:
uv run hpcsim simulate --scheduler my_scheduler
uv run hpcsim benchmark --schedulers fifo,gavel,my_scheduler
```
