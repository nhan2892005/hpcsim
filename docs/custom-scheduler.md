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
            cpu_alloc = self._find_cpus(job)
            if cpu_alloc:
                decision.add_cpu(job, cpu_alloc)

        elif rt == ResourceType.MIG:
            mig_ids = self._find_mig(job)
            if mig_ids:
                decision.add_mig(job, mig_ids)

        elif rt == ResourceType.CPU_GPU:
            # Phải có cả GPU lẫn CPU mới schedule
            gpu_ids   = self._find_gpus(job)
            cpu_alloc = self._find_cpus(job)
            if gpu_ids and cpu_alloc:
                decision.add_hybrid(job, gpu_ids, cpu_alloc)

        else:
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

---

## Kết hợp với BackfillWrapper

`BackfillWrapper` hoạt động với **bất kỳ** custom scheduler nào — không cần sửa code scheduler. Chỉ cần wrap khi tạo instance:

### EASY-Backfilling (không cần năng lượng xanh)

```python
from hpcsim.scheduler.backfill import BackfillWrapper, EASYBackfillPolicy
from hpcsim.scheduler.schedulers import create_scheduler

# Cách 1: wrap trực tiếp
sched = BackfillWrapper(
    primary = MyScheduler(cluster),
    policy  = EASYBackfillPolicy(),
)

# Cách 2: dùng factory helper
from hpcsim import wrap_with_backfill
sched = wrap_with_backfill(MyScheduler(cluster), "easy")
```

### Green-Backfilling (với năng lượng xanh)

```python
from hpcsim.scheduler.backfill import BackfillWrapper, GreenBackfillPolicy
from hpcsim.energy.renewable import RenewableEnergyModule
from hpcsim import wrap_with_backfill

re    = RenewableEnergyModule(total_gpus=cluster.total_gpus(), sim_duration=86400)

# Cách 1: wrap trực tiếp
sched = BackfillWrapper(
    primary = MyScheduler(cluster),
    policy  = GreenBackfillPolicy(re),
)

# Cách 2: factory helper
sched = wrap_with_backfill(MyScheduler(cluster), "green", renewable=re)
```

### Sử dụng trong CLI

Sau khi đăng ký với `@register`, custom scheduler dùng được ngay với `--backfill`:

```bash
# Không backfilling (mặc định)
uv run hpcsim simulate --scheduler my_scheduler

# Với EASY-Backfilling
uv run hpcsim simulate --scheduler my_scheduler --backfill easy

# Với Green-Backfilling
uv run hpcsim simulate --scheduler my_scheduler --backfill green

# Benchmark với backfilling
uv run hpcsim benchmark \
    --schedulers fifo,gavel,my_scheduler \
    --backfill easy
```

---

## Scheduler có cơ chế Delay (tích hợp Green-Backfilling tối ưu)

Nếu scheduler của bạn có cơ chế **delay job** (như GAS-MARL), hãy đặt `decision.delay_info` để `BackfillWrapper` tính đúng backfill window theo thuật toán paper §4.3. Nếu không set, wrapper dùng shadow time mặc định.

```python
from hpcsim.scheduler.schedulers import BaseScheduler, SchedulingDecision, register

@register
class MyDelayScheduler(BaseScheduler):
    name = "my_delay_scheduler"

    def schedule(self, pending, running, current_time):
        decision = SchedulingDecision()

        selected_job = self._pick_job(pending)   # logic của bạn
        should_delay, delay_seconds = self._check_delay(selected_job, current_time)

        if should_delay:
            # Ghi delay metadata — BackfillWrapper sẽ đọc điều này
            # để mở backfill window đúng thời gian
            decision.delay_info = {
                "delay_type":    6,                              # type 6 = delay DT giây
                "release_time":  current_time + delay_seconds,  # khi nào job được chạy
                "head_job_id":   selected_job.job_id,
                "head_req_gpus": getattr(selected_job, "num_gpus_requested", 1),
            }
            # Không add job vào decision — job bị delay
        else:
            res = self._find_resources(selected_job)
            if res:
                gpus, migs, cpus = res
                decision.add(selected_job, gpus, migs, cpus)

        return decision
```

### Cấu trúc `delay_info`

| Key | Type | Mô tả |
|-----|------|-------|
| `delay_type` | `int` | 1–5 = chờ N jobs xong; 6–12 = chờ DT giây |
| `release_time` | `float` | Timestamp khi job được phép chạy |
| `head_job_id` | `str` | job_id của job đang bị delay |
| `head_req_gpus` | `int` | Số GPU job cần (để tính shadow time) |

`BackfillWrapper` tính `max_finish_time` (giới hạn backfill window) như sau:
- `delay_type` 1–5: `max(release_time, shadow_time)` capped tại `current_time + 3600`
- `delay_type` 6–12: `max(shadow_time, current_time + DT)`
- Nếu không có `delay_info`: dùng `shadow_time` chuẩn (EASY logic)

---

## Đăng ký ngoài package (dynamic registration)

```python
# Dùng khi custom scheduler ở file riêng, không muốn sửa package
from hpcsim.scheduler.schedulers import register_factory

register_factory(
    "my_scheduler",
    lambda cluster: MyScheduler(cluster, param=42),
)

# Sau đó dùng bình thường
from hpcsim import create_scheduler, wrap_with_backfill
sched = wrap_with_backfill(create_scheduler("my_scheduler", cluster), "green", renewable=re)
```