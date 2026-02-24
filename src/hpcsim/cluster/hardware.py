"""
Hardware models for HPC GPU cluster simulation.

Based on:
- Ye et al. "Deep Learning Workload Scheduling in GPU Datacenters: A Survey" (ACM CSUR 2024)
  T1: Heterogeneous affinity, T2: Placement sensitivity
- Raeisi et al. "GOGH: Correlation-Guided Orchestration of GPUs" (2025)
  GOGH §2.4 power model γ_a(x)
- Gavel benchmark dataset [Narayanan et al., OSDI'20]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────
# GPU Type Definitions  (Survey T1 – heterogeneous hardware)
# ─────────────────────────────────────────────────────────────

class GPUType(str, Enum):
    K80           = "k80"
    P100          = "p100"
    V100          = "v100"
    A100          = "a100"
    RTX3090       = "rtx3090"
    # Fragmented / unconsolidated variants (Gavel dataset GOGH paper)
    K80_UNCONSOL  = "k80_unconsolidated"
    P100_UNCONSOL = "p100_unconsolidated"
    V100_UNCONSOL = "v100_unconsolidated"


class InterconnectType(str, Enum):
    NVLINK     = "nvlink"
    PCIE       = "pcie"
    INFINIBAND = "infiniband"
    ETHERNET   = "ethernet"


@dataclass(frozen=True)
class GPUSpec:
    """
    Hardware specification of a GPU type.
    Calibrated from official NVIDIA datasheets.
    """
    gpu_type: GPUType
    compute_capability: float    # relative, K80 = 1.0
    memory_gb: float
    memory_bandwidth_gbps: float
    fp32_tflops: float
    fp16_tflops: float
    tdp_watts: float
    idle_watts: float
    nvlink_bandwidth_gbps: Optional[float] = None
    pcie_bandwidth_gbps: float = 16.0


GPU_SPECS: dict[GPUType, GPUSpec] = {
    GPUType.K80: GPUSpec(
        gpu_type=GPUType.K80, compute_capability=1.0,
        memory_gb=24.0, memory_bandwidth_gbps=480.0,
        fp32_tflops=8.73, fp16_tflops=8.73,
        tdp_watts=300.0, idle_watts=25.0,
    ),
    GPUType.P100: GPUSpec(
        gpu_type=GPUType.P100, compute_capability=3.5,
        memory_gb=16.0, memory_bandwidth_gbps=720.0,
        fp32_tflops=9.3, fp16_tflops=18.7,
        tdp_watts=250.0, idle_watts=20.0,
        nvlink_bandwidth_gbps=160.0,
    ),
    GPUType.V100: GPUSpec(
        gpu_type=GPUType.V100, compute_capability=6.5,
        memory_gb=32.0, memory_bandwidth_gbps=900.0,
        fp32_tflops=14.0, fp16_tflops=112.0,
        tdp_watts=300.0, idle_watts=25.0,
        nvlink_bandwidth_gbps=300.0, pcie_bandwidth_gbps=32.0,
    ),
    GPUType.A100: GPUSpec(
        gpu_type=GPUType.A100, compute_capability=19.5,
        memory_gb=80.0, memory_bandwidth_gbps=2000.0,
        fp32_tflops=19.5, fp16_tflops=312.0,
        tdp_watts=400.0, idle_watts=30.0,
        nvlink_bandwidth_gbps=600.0, pcie_bandwidth_gbps=64.0,
    ),
    GPUType.RTX3090: GPUSpec(
        gpu_type=GPUType.RTX3090, compute_capability=8.0,
        memory_gb=24.0, memory_bandwidth_gbps=936.0,
        fp32_tflops=35.6, fp16_tflops=71.0,
        tdp_watts=350.0, idle_watts=20.0, pcie_bandwidth_gbps=32.0,
    ),
    GPUType.K80_UNCONSOL: GPUSpec(
        gpu_type=GPUType.K80_UNCONSOL, compute_capability=0.5,
        memory_gb=12.0, memory_bandwidth_gbps=240.0,
        fp32_tflops=4.36, fp16_tflops=4.36,
        tdp_watts=150.0, idle_watts=12.0, pcie_bandwidth_gbps=8.0,
    ),
    GPUType.P100_UNCONSOL: GPUSpec(
        gpu_type=GPUType.P100_UNCONSOL, compute_capability=1.75,
        memory_gb=8.0, memory_bandwidth_gbps=360.0,
        fp32_tflops=4.65, fp16_tflops=9.35,
        tdp_watts=125.0, idle_watts=10.0, pcie_bandwidth_gbps=8.0,
    ),
    GPUType.V100_UNCONSOL: GPUSpec(
        gpu_type=GPUType.V100_UNCONSOL, compute_capability=3.25,
        memory_gb=16.0, memory_bandwidth_gbps=450.0,
        fp32_tflops=7.0, fp16_tflops=56.0,
        tdp_watts=150.0, idle_watts=12.0, pcie_bandwidth_gbps=16.0,
    ),
}


@dataclass(frozen=True)
class InterconnectSpec:
    bw_gbps: float
    latency_us: float
    name: str


INTERCONNECT_SPECS: dict[InterconnectType, InterconnectSpec] = {
    InterconnectType.NVLINK:     InterconnectSpec(300.0,  1.0, "NVLink"),
    InterconnectType.PCIE:       InterconnectSpec(16.0,   5.0, "PCIe"),
    InterconnectType.INFINIBAND: InterconnectSpec(12.5,   2.0, "InfiniBand HDR"),
    InterconnectType.ETHERNET:   InterconnectSpec(1.25,  10.0, "Ethernet 10GbE"),
}


# ─────────────────────────────────────────────────────────────
# GPU Instance (runtime state)
# ─────────────────────────────────────────────────────────────

@dataclass
class GPUInstance:
    """Runtime state of a single physical GPU."""
    gpu_id: str
    node_id: str
    gpu_type: GPUType
    spec: GPUSpec
    allocated_jobs: list = field(default_factory=list)
    utilization: float = 0.0
    memory_used_gb: float = 0.0
    capacity: int = 2   # max co-located jobs (θ_a in GOGH)

    def is_free(self) -> bool:
        return len(self.allocated_jobs) == 0

    def has_capacity(self, slots: int = 1) -> bool:
        return len(self.allocated_jobs) + slots <= self.capacity

    def current_power_watts(self) -> float:
        """
        GOGH §2.4 power model: γ_a(u) = P_idle + (P_tdp - P_idle)·(0.3u + 0.7u²)
        """
        u = self.utilization
        rng = self.spec.tdp_watts - self.spec.idle_watts
        return self.spec.idle_watts + rng * (0.3 * u + 0.7 * u * u)

    @property
    def memory_free_gb(self) -> float:
        return self.spec.memory_gb - self.memory_used_gb


@dataclass
class ServerNode:
    """A server node containing one GPU type."""
    node_id: str
    gpu_type: GPUType
    num_gpus: int
    cpu_cores: int = 64
    ram_gb: float = 512.0
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND
    gpus: list = field(default_factory=list)

    def build_gpus(self) -> None:
        spec = GPU_SPECS[self.gpu_type]
        self.gpus = [
            GPUInstance(
                gpu_id=f"{self.node_id}_g{i}",
                node_id=self.node_id,
                gpu_type=self.gpu_type,
                spec=spec,
            )
            for i in range(self.num_gpus)
        ]

    def free_gpu_count(self) -> int:
        return sum(1 for g in self.gpus if g.is_free())

    def total_power_watts(self) -> float:
        return sum(g.current_power_watts() for g in self.gpus)


# ─────────────────────────────────────────────────────────────
# Communication Time Models  (Survey T2 – placement sensitivity)
# ─────────────────────────────────────────────────────────────

def ring_allreduce_time_sec(
    model_params: int,
    num_gpus: int,
    bandwidth_gbps: float,
    dtype_bytes: int = 4,
) -> float:
    """
    Ring-AllReduce communication time per iteration.
    Formula: t = 2·(N-1)/N · M / B   [Horovod, Sergeev & Del Balso 2018]
    Survey T2 – placement sensitivity model.
    """
    if num_gpus <= 1:
        return 0.0
    model_bytes = model_params * dtype_bytes
    bw_bytes = bandwidth_gbps * 1e9
    return 2.0 * (num_gpus - 1) / num_gpus * model_bytes / bw_bytes


def ps_communication_time_sec(
    model_params: int,
    num_workers: int,
    bandwidth_gbps: float,
    dtype_bytes: int = 4,
) -> float:
    """Parameter-Server communication time per iteration."""
    if num_workers <= 1:
        return 0.0
    model_bytes = model_params * dtype_bytes
    bw_bytes = bandwidth_gbps * 1e9
    return 2.0 * model_bytes * num_workers / bw_bytes
