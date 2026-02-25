"""
Hardware models for HPC cluster simulation.

Supports:
  - GPU nodes (homogeneous or heterogeneous GPU types)
  - CPU-only nodes (login nodes, data-prep, parameter servers)
  - Mixed CPU+GPU nodes (most modern HPC nodes)
  - MIG (Multi-Instance GPU) partitioning on A100/H100
  - MPS (Multi-Process Service) GPU time-sharing

References:
  - Ye et al. "Deep Learning Workload Scheduling in GPU Datacenters: A Survey" (CSUR 2024)
  - NVIDIA A100 MIG User Guide (v23.09)
  - Chen et al. GAS-MARL (FGCS 2025)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


# ─── GPU Type ─────────────────────────────────────────────────────────────────

class GPUType(str, Enum):
    K80           = "k80"
    P100          = "p100"
    V100          = "v100"
    A100          = "a100"
    H100          = "h100"
    RTX3090       = "rtx3090"
    RTX4090       = "rtx4090"
    # Fragmented / unconsolidated variants (Gavel / GOGH dataset)
    K80_UNCONSOL  = "k80_unconsolidated"
    P100_UNCONSOL = "p100_unconsolidated"
    V100_UNCONSOL = "v100_unconsolidated"


# ─── CPU Type ─────────────────────────────────────────────────────────────────

class CPUType(str, Enum):
    """Common HPC CPU families."""
    XEON_E5    = "xeon_e5"       # Intel Xeon E5 — older HPC workhorse
    XEON_GOLD  = "xeon_gold"     # Intel Xeon Gold Scalable
    XEON_PLAT  = "xeon_platinum" # Intel Xeon Platinum — high-end
    EPYC_7002  = "epyc_7002"     # AMD EPYC Rome
    EPYC_7003  = "epyc_7003"     # AMD EPYC Milan
    EPYC_9004  = "epyc_9004"     # AMD EPYC Genoa
    GRACE      = "grace"         # NVIDIA Grace (ARM, paired with H100)
    GENERIC    = "generic"       # Unspecified / simulation placeholder


# ─── Interconnect ─────────────────────────────────────────────────────────────

class InterconnectType(str, Enum):
    NVLINK     = "nvlink"
    NVLINK4    = "nvlink4"       # NVLink 4.0 (H100)
    PCIE       = "pcie"
    PCIE5      = "pcie5"         # PCIe 5.0
    INFINIBAND = "infiniband"    # IB HDR 100 Gb/s
    INFINIBAND_NDR = "infiniband_ndr"  # IB NDR 400 Gb/s (H100 clusters)
    ETHERNET   = "ethernet"
    NVSWITCH   = "nvswitch"      # NVSwitch fabric (DGX SuperPOD)


# ─── MIG (Multi-Instance GPU) ─────────────────────────────────────────────────

class MIGProfile(str, Enum):
    """
    NVIDIA A100 MIG profiles (also available on H100).
    Format: <compute_slices>g.<memory_gb>gb
    A100-80GB supports up to 7 instances.
    """
    G1_10GB  = "1g.10gb"   # 1/7 compute, 10 GB  — up to 7 instances
    G2_20GB  = "2g.20gb"   # 2/7 compute, 20 GB  — up to 3 instances
    G3_40GB  = "3g.40gb"   # 3/7 compute, 40 GB  — up to 2 instances
    G7_80GB  = "7g.80gb"   # Full GPU, 80 GB     — 1 instance (= no MIG)
    # H100 profiles
    G1_10GB_H = "1g.10gb.h100"
    G2_20GB_H = "2g.20gb.h100"
    G4_40GB_H = "4g.40gb.h100"
    # Disable MIG (whole GPU)
    NONE     = "none"


MIG_PROFILE_SPECS: dict[MIGProfile, dict] = {
    MIGProfile.G1_10GB:  {"compute_fraction": 1/7, "memory_gb": 10,  "max_per_gpu": 7},
    MIGProfile.G2_20GB:  {"compute_fraction": 2/7, "memory_gb": 20,  "max_per_gpu": 3},
    MIGProfile.G3_40GB:  {"compute_fraction": 3/7, "memory_gb": 40,  "max_per_gpu": 2},
    MIGProfile.G7_80GB:  {"compute_fraction": 7/7, "memory_gb": 80,  "max_per_gpu": 1},
    MIGProfile.G1_10GB_H:{"compute_fraction": 1/7, "memory_gb": 10,  "max_per_gpu": 7},
    MIGProfile.G2_20GB_H:{"compute_fraction": 2/7, "memory_gb": 20,  "max_per_gpu": 3},
    MIGProfile.G4_40GB_H:{"compute_fraction": 4/7, "memory_gb": 40,  "max_per_gpu": 2},
    MIGProfile.NONE:     {"compute_fraction": 1.0, "memory_gb": None, "max_per_gpu": 1},
}


# ─── GPU Spec ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPUSpec:
    """
    Hardware specification of a GPU type.
    compute_capability: throughput relative to K80 = 1.0.
    mig_capable: whether this GPU supports MIG partitioning.
    """
    gpu_type: GPUType
    compute_capability: float
    memory_gb: float
    memory_bandwidth_gbps: float
    fp32_tflops: float
    fp16_tflops: float
    tdp_watts: float
    idle_watts: float
    nvlink_bandwidth_gbps: Optional[float] = None
    pcie_bandwidth_gbps: float = 16.0
    mig_capable: bool = False
    mps_capable: bool = True         # all modern GPUs support MPS
    default_mig_profile: MIGProfile = MIGProfile.NONE


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
        mig_capable=True, default_mig_profile=MIGProfile.NONE,
    ),
    GPUType.H100: GPUSpec(
        gpu_type=GPUType.H100, compute_capability=60.0,
        memory_gb=80.0, memory_bandwidth_gbps=3350.0,
        fp32_tflops=67.0, fp16_tflops=989.0,
        tdp_watts=700.0, idle_watts=50.0,
        nvlink_bandwidth_gbps=900.0, pcie_bandwidth_gbps=128.0,
        mig_capable=True, default_mig_profile=MIGProfile.NONE,
    ),
    GPUType.RTX3090: GPUSpec(
        gpu_type=GPUType.RTX3090, compute_capability=8.0,
        memory_gb=24.0, memory_bandwidth_gbps=936.0,
        fp32_tflops=35.6, fp16_tflops=71.0,
        tdp_watts=350.0, idle_watts=20.0, pcie_bandwidth_gbps=32.0,
    ),
    GPUType.RTX4090: GPUSpec(
        gpu_type=GPUType.RTX4090, compute_capability=30.0,
        memory_gb=24.0, memory_bandwidth_gbps=1008.0,
        fp32_tflops=82.6, fp16_tflops=165.2,
        tdp_watts=450.0, idle_watts=25.0, pcie_bandwidth_gbps=64.0,
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


# ─── CPU Spec ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CPUSpec:
    """
    Hardware specification of a CPU type.
    Single-threaded performance relative to Xeon E5 = 1.0.
    """
    cpu_type: CPUType
    cores_per_socket: int
    base_clock_ghz: float
    boost_clock_ghz: float
    memory_channels: int
    memory_bandwidth_gbps: float
    tdp_watts: float
    idle_watts: float
    st_perf: float          # single-thread performance (relative, Xeon E5 = 1.0)
    mt_perf: float          # multi-thread performance (relative)
    avx512: bool = False    # AVX-512 support (important for ML inference on CPU)
    numa_nodes: int = 1


CPU_SPECS: dict[CPUType, CPUSpec] = {
    CPUType.XEON_E5: CPUSpec(
        cpu_type=CPUType.XEON_E5, cores_per_socket=14,
        base_clock_ghz=2.4, boost_clock_ghz=3.2,
        memory_channels=4, memory_bandwidth_gbps=68.0,
        tdp_watts=120.0, idle_watts=15.0,
        st_perf=1.0, mt_perf=1.0,
    ),
    CPUType.XEON_GOLD: CPUSpec(
        cpu_type=CPUType.XEON_GOLD, cores_per_socket=24,
        base_clock_ghz=2.5, boost_clock_ghz=3.8,
        memory_channels=6, memory_bandwidth_gbps=141.0,
        tdp_watts=150.0, idle_watts=20.0,
        st_perf=1.4, mt_perf=2.8, avx512=True,
    ),
    CPUType.XEON_PLAT: CPUSpec(
        cpu_type=CPUType.XEON_PLAT, cores_per_socket=32,
        base_clock_ghz=2.6, boost_clock_ghz=4.0,
        memory_channels=8, memory_bandwidth_gbps=204.8,
        tdp_watts=205.0, idle_watts=30.0,
        st_perf=1.6, mt_perf=4.5, avx512=True, numa_nodes=2,
    ),
    CPUType.EPYC_7002: CPUSpec(
        cpu_type=CPUType.EPYC_7002, cores_per_socket=32,
        base_clock_ghz=2.6, boost_clock_ghz=3.35,
        memory_channels=8, memory_bandwidth_gbps=204.8,
        tdp_watts=180.0, idle_watts=25.0,
        st_perf=1.3, mt_perf=4.2, avx512=False, numa_nodes=4,
    ),
    CPUType.EPYC_7003: CPUSpec(
        cpu_type=CPUType.EPYC_7003, cores_per_socket=64,
        base_clock_ghz=2.65, boost_clock_ghz=3.7,
        memory_channels=8, memory_bandwidth_gbps=204.8,
        tdp_watts=280.0, idle_watts=35.0,
        st_perf=1.6, mt_perf=8.0, avx512=False, numa_nodes=4,
    ),
    CPUType.EPYC_9004: CPUSpec(
        cpu_type=CPUType.EPYC_9004, cores_per_socket=96,
        base_clock_ghz=2.4, boost_clock_ghz=3.7,
        memory_channels=12, memory_bandwidth_gbps=460.8,
        tdp_watts=360.0, idle_watts=45.0,
        st_perf=1.7, mt_perf=12.0, avx512=True, numa_nodes=4,
    ),
    CPUType.GRACE: CPUSpec(
        cpu_type=CPUType.GRACE, cores_per_socket=72,
        base_clock_ghz=3.1, boost_clock_ghz=3.6,
        memory_channels=16, memory_bandwidth_gbps=512.0,
        tdp_watts=500.0, idle_watts=60.0,  # includes NVLink to H100
        st_perf=1.5, mt_perf=11.0, avx512=False, numa_nodes=1,
    ),
    CPUType.GENERIC: CPUSpec(
        cpu_type=CPUType.GENERIC, cores_per_socket=16,
        base_clock_ghz=2.5, boost_clock_ghz=3.0,
        memory_channels=4, memory_bandwidth_gbps=50.0,
        tdp_watts=100.0, idle_watts=10.0,
        st_perf=1.0, mt_perf=1.5,
    ),
}


# ─── Interconnect ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InterconnectSpec:
    bw_gbps: float
    latency_us: float
    name: str


INTERCONNECT_SPECS: dict[InterconnectType, InterconnectSpec] = {
    InterconnectType.NVLINK:         InterconnectSpec(300.0,   1.0, "NVLink 3.0"),
    InterconnectType.NVLINK4:        InterconnectSpec(900.0,   0.5, "NVLink 4.0"),
    InterconnectType.PCIE:           InterconnectSpec(16.0,    5.0, "PCIe 4.0"),
    InterconnectType.PCIE5:          InterconnectSpec(32.0,    3.0, "PCIe 5.0"),
    InterconnectType.INFINIBAND:     InterconnectSpec(12.5,    2.0, "InfiniBand HDR"),
    InterconnectType.INFINIBAND_NDR: InterconnectSpec(50.0,    1.0, "InfiniBand NDR"),
    InterconnectType.ETHERNET:       InterconnectSpec(1.25,   10.0, "Ethernet 10GbE"),
    InterconnectType.NVSWITCH:       InterconnectSpec(900.0,   0.3, "NVSwitch"),
}


# ─── GPU Instance (runtime state) ─────────────────────────────────────────────

@dataclass
class MIGInstance:
    """
    A single MIG slice — a hardware partition of a physical GPU.
    Created when a GPUInstance is partitioned via MIG.
    """
    mig_id: str                     # e.g. "n_v100_0_g0_mig0"
    parent_gpu_id: str
    node_id: str
    gpu_type: GPUType
    profile: MIGProfile
    spec: GPUSpec                   # parent GPU spec
    allocated_jobs: list = field(default_factory=list)
    utilization: float = 0.0
    memory_used_gb: float = 0.0

    @property
    def memory_gb(self) -> float:
        s = MIG_PROFILE_SPECS.get(self.profile, {})
        return s.get("memory_gb") or self.spec.memory_gb

    @property
    def compute_fraction(self) -> float:
        return MIG_PROFILE_SPECS.get(self.profile, {}).get("compute_fraction", 1.0)

    def is_free(self) -> bool:
        return len(self.allocated_jobs) == 0

    def current_power_watts(self) -> float:
        """MIG slice power = fraction × full GPU power."""
        u   = self.utilization
        rng = self.spec.tdp_watts - self.spec.idle_watts
        full_power = self.spec.idle_watts + rng * (0.3 * u + 0.7 * u * u)
        return full_power * self.compute_fraction


@dataclass
class GPUInstance:
    """
    Runtime state of a single physical GPU.
    May be partitioned into MIG slices or shared via MPS.
    """
    gpu_id: str
    node_id: str
    gpu_type: GPUType
    spec: GPUSpec
    allocated_jobs: list = field(default_factory=list)
    utilization: float = 0.0
    memory_used_gb: float = 0.0
    # MIG partitioning
    mig_mode: bool = False
    mig_profile: MIGProfile = MIGProfile.NONE
    mig_instances: list = field(default_factory=list)   # list[MIGInstance]
    # MPS time-sharing (capacity = max concurrent jobs)
    mps_mode: bool = False
    capacity: int = 1        # 1 = exclusive; >1 = MPS sharing

    def is_free(self) -> bool:
        if self.mig_mode:
            return all(m.is_free() for m in self.mig_instances)
        return len(self.allocated_jobs) == 0

    def has_capacity(self, slots: int = 1) -> bool:
        return len(self.allocated_jobs) + slots <= self.capacity

    def free_mig_instances(self) -> list:
        return [m for m in self.mig_instances if m.is_free()]

    def enable_mig(self, profile: MIGProfile) -> None:
        """Partition this GPU into MIG instances of given profile."""
        if not self.spec.mig_capable:
            raise ValueError(f"{self.gpu_type} does not support MIG")
        spec_data  = MIG_PROFILE_SPECS[profile]
        n_slices   = spec_data["max_per_gpu"]
        self.mig_mode    = True
        self.mig_profile = profile
        self.mig_instances = [
            MIGInstance(
                mig_id=f"{self.gpu_id}_mig{i}",
                parent_gpu_id=self.gpu_id,
                node_id=self.node_id,
                gpu_type=self.gpu_type,
                profile=profile,
                spec=self.spec,
            )
            for i in range(n_slices)
        ]

    def enable_mps(self, max_jobs: int = 4) -> None:
        """Enable MPS time-sharing with up to max_jobs concurrent jobs."""
        self.mps_mode = True
        self.capacity = max_jobs

    def disable_mig(self) -> None:
        self.mig_mode      = False
        self.mig_profile   = MIGProfile.NONE
        self.mig_instances = []

    def current_power_watts(self) -> float:
        """GOGH §2.4 power model: P_idle + (P_tdp - P_idle)·(0.3u + 0.7u²)"""
        u   = self.utilization
        rng = self.spec.tdp_watts - self.spec.idle_watts
        return self.spec.idle_watts + rng * (0.3 * u + 0.7 * u * u)

    @property
    def memory_free_gb(self) -> float:
        return self.spec.memory_gb - self.memory_used_gb


# ─── CPU Instance (runtime state) ─────────────────────────────────────────────

@dataclass
class CPUInstance:
    """
    Runtime state of a CPU (one socket/chip).
    CPU resources are tracked at core granularity.
    """
    cpu_id: str          # e.g. "n_login_0_cpu0"
    node_id: str
    cpu_type: CPUType
    spec: CPUSpec
    total_cores: int
    allocated_cores: int = 0
    utilization: float = 0.0
    allocated_jobs: list = field(default_factory=list)   # list of job_ids

    @property
    def free_cores(self) -> int:
        return self.total_cores - self.allocated_cores

    def is_free(self) -> bool:
        return self.allocated_cores == 0

    def can_allocate(self, cores: int) -> bool:
        return self.free_cores >= cores

    def current_power_watts(self) -> float:
        """Linear power model for CPU."""
        u   = self.utilization
        rng = self.spec.tdp_watts - self.spec.idle_watts
        return self.spec.idle_watts + rng * u


# ─── Server Node ──────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    GPU_ONLY  = "gpu_only"   # Only GPUs (no significant CPU compute role)
    CPU_ONLY  = "cpu_only"   # Login nodes, pre/post-processing, parameter servers
    MIXED     = "mixed"      # Standard HPC node: CPUs + GPUs


@dataclass
class ServerNode:
    """
    A server node in the cluster.
    Supports three configurations:
      - GPU-only:   Traditional HPC accelerator node
      - CPU-only:   Login/compute/preprocessing node
      - Mixed:      CPU cores + GPU accelerators (most common in practice)
    """
    node_id: str
    node_type: NodeType = NodeType.GPU_ONLY

    # GPU configuration (for GPU_ONLY and MIXED nodes)
    gpu_type: Optional[GPUType] = None
    num_gpus: int = 0

    # CPU configuration (for CPU_ONLY and MIXED nodes)
    cpu_type: CPUType = CPUType.XEON_GOLD
    num_sockets: int = 2
    cores_per_socket: int = 0    # 0 = use CPUSpec default

    # Memory
    ram_gb: float = 256.0

    # Interconnects
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND

    # Runtime state — populated by build()
    gpus:  list = field(default_factory=list)   # list[GPUInstance]
    cpus:  list = field(default_factory=list)   # list[CPUInstance]

    def build(self) -> None:
        """Create GPU and CPU instance objects."""
        self._build_gpus()
        self._build_cpus()

    def _build_gpus(self) -> None:
        if self.num_gpus == 0 or self.gpu_type is None:
            self.gpus = []
            return
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

    def _build_cpus(self) -> None:
        if self.node_type == NodeType.GPU_ONLY:
            # GPU-only nodes still have CPUs but not scheduled
            self.cpus = []
            return
        spec = CPU_SPECS.get(self.cpu_type, CPU_SPECS[CPUType.GENERIC])
        cores = self.cores_per_socket or spec.cores_per_socket
        self.cpus = [
            CPUInstance(
                cpu_id=f"{self.node_id}_cpu{s}",
                node_id=self.node_id,
                cpu_type=self.cpu_type,
                spec=spec,
                total_cores=cores,
            )
            for s in range(self.num_sockets)
        ]

    # ── Legacy compatibility ──────────────────────────────────────────────────

    def build_gpus(self) -> None:
        """Backward-compatible alias for build()."""
        self.build()

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def total_cpu_cores(self) -> int:
        if self.cpus:
            return sum(c.total_cores for c in self.cpus)
        spec = CPU_SPECS.get(self.cpu_type, CPU_SPECS[CPUType.GENERIC])
        return (self.cores_per_socket or spec.cores_per_socket) * self.num_sockets

    @property
    def free_cpu_cores(self) -> int:
        return sum(c.free_cores for c in self.cpus)

    def free_gpu_count(self) -> int:
        return sum(1 for g in self.gpus if g.is_free())

    def total_power_watts(self) -> float:
        gpu_power = sum(g.current_power_watts() for g in self.gpus)
        cpu_power = sum(c.current_power_watts() for c in self.cpus)
        # Base system power (cooling, networking, memory)
        base_power = 50.0 + (self.ram_gb / 1024) * 20.0
        return gpu_power + cpu_power + base_power


# ─── Communication Time Models ────────────────────────────────────────────────

def ring_allreduce_time_sec(
    model_params: int,
    num_gpus: int,
    bandwidth_gbps: float,
    dtype_bytes: int = 4,
) -> float:
    """Ring-AllReduce communication time per iteration."""
    if num_gpus <= 1:
        return 0.0
    model_bytes = model_params * dtype_bytes
    bw_bytes    = bandwidth_gbps * 1e9
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
    bw_bytes    = bandwidth_gbps * 1e9
    return 2.0 * model_bytes * num_workers / bw_bytes
