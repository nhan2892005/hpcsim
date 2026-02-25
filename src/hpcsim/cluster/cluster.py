"""
Cluster resource manager.

Supports GPU-only, CPU-only, and mixed GPU+CPU nodes.
Placement-aware allocation for heterogeneous clusters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .hardware import (
    GPUType, CPUType, NodeType,
    GPU_SPECS, CPU_SPECS, CPUSpec,
    GPUInstance, CPUInstance, MIGInstance, MIGProfile,
    ServerNode,
    InterconnectType, INTERCONNECT_SPECS,
)


# ─── NodeSpec ─────────────────────────────────────────────────────────────────

@dataclass
class NodeSpec:
    """
    Declarative specification of a node type in the cluster.
    Replaces the old (GPUType, num_nodes, gpus_per_node) tuple.

    Examples:
        # Standard GPU node
        NodeSpec(gpu_type=GPUType.V100, num_nodes=4, gpus_per_node=8)

        # CPU-only login/preprocessing node
        NodeSpec(node_type=NodeType.CPU_ONLY, cpu_type=CPUType.EPYC_7003,
                 num_nodes=4, num_sockets=2, cores_per_socket=64, ram_gb=512)

        # Mixed node: CPU cores + GPUs (common in modern HPC)
        NodeSpec(node_type=NodeType.MIXED,
                 cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                 gpu_type=GPUType.A100, gpus_per_node=4,
                 num_nodes=8, ram_gb=256)

        # A100 with MIG enabled (all GPUs partitioned into 1g.10gb slices)
        NodeSpec(gpu_type=GPUType.A100, num_nodes=2, gpus_per_node=8,
                 mig_profile=MIGProfile.G1_10GB)
    """
    # Node multiplicity
    num_nodes:          int         = 1

    # Node type
    node_type:          NodeType    = NodeType.GPU_ONLY

    # GPU config (GPU_ONLY / MIXED)
    gpu_type:           Optional[GPUType] = None
    gpus_per_node:      int         = 0

    # CPU config (CPU_ONLY / MIXED)
    cpu_type:           CPUType     = CPUType.XEON_GOLD
    num_sockets:        int         = 2
    cores_per_socket:   int         = 0   # 0 = use CPUSpec default

    # Memory
    ram_gb:             float       = 256.0

    # Interconnects
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND

    # MIG / MPS
    mig_profile:        MIGProfile  = MIGProfile.NONE
    mps_max_jobs:       int         = 1   # 1 = exclusive

    # Metadata
    role:               str         = ""  # e.g. "login", "compute", "storage"

    def __post_init__(self):
        # Auto-infer node_type if not explicit
        has_gpu = self.gpu_type is not None and self.gpus_per_node > 0
        has_cpu_alloc = self.node_type in (NodeType.CPU_ONLY, NodeType.MIXED)
        if has_gpu and not has_cpu_alloc:
            if self.node_type != NodeType.MIXED:
                pass  # Keep GPU_ONLY as default
        if not has_gpu and self.node_type == NodeType.GPU_ONLY:
            self.node_type = NodeType.CPU_ONLY

    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    def total_cpu_cores(self) -> int:
        if self.node_type == NodeType.GPU_ONLY:
            return 0  # CPU not scheduled on GPU-only nodes
        spec = CPU_SPECS.get(self.cpu_type, CPU_SPECS[CPUType.XEON_GOLD])
        cpp  = self.cores_per_socket or spec.cores_per_socket
        return self.num_nodes * self.num_sockets * cpp


# ─── ClusterConfig ────────────────────────────────────────────────────────────

@dataclass
class ClusterConfig:
    """
    Cluster configuration.
    nodes: list of NodeSpec (new) OR legacy list of (GPUType, num_nodes, gpus_per_node) tuples.
    """
    name: str
    nodes: list                                         # list[NodeSpec | tuple]
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND
    # Legacy GPU-only defaults (used when nodes are plain tuples)
    cpu_cores_per_node: int   = 64
    ram_gb_per_node:    float = 512.0

    def _normalise_nodes(self) -> list:
        """Convert legacy tuples to NodeSpec objects."""
        result = []
        for n in self.nodes:
            if isinstance(n, NodeSpec):
                result.append(n)
            elif isinstance(n, tuple):
                # Old format: (GPUType, num_nodes, gpus_per_node)
                gpu_type, num_nodes, gpus_per_node = n
                result.append(NodeSpec(
                    num_nodes=num_nodes,
                    node_type=NodeType.GPU_ONLY,
                    gpu_type=gpu_type,
                    gpus_per_node=gpus_per_node,
                    ram_gb=self.ram_gb_per_node,
                    intra_interconnect=self.intra_interconnect,
                    inter_interconnect=self.inter_interconnect,
                ))
            else:
                raise ValueError(f"Unknown node spec type: {type(n)}")
        return result


# ─── Preset Cluster Configurations ───────────────────────────────────────────

CLUSTER_CONFIGS: dict[str, ClusterConfig] = {

    # ── Minimal test cluster ──────────────────────────────────────────────────
    "tiny_test": ClusterConfig(
        "tiny_test",
        nodes=[NodeSpec(gpu_type=GPUType.V100, num_nodes=2, gpus_per_node=4)],
    ),

    # ── GPU-only clusters ─────────────────────────────────────────────────────
    "small_v100": ClusterConfig(
        "small_v100",
        nodes=[NodeSpec(gpu_type=GPUType.V100, num_nodes=4, gpus_per_node=8)],
    ),

    "medium_heterogeneous_gavel": ClusterConfig(
        "medium_heterogeneous_gavel",
        nodes=[
            NodeSpec(gpu_type=GPUType.K80,  num_nodes=6, gpus_per_node=8),
            NodeSpec(gpu_type=GPUType.P100, num_nodes=4, gpus_per_node=8),
            NodeSpec(gpu_type=GPUType.V100, num_nodes=4, gpus_per_node=8),
        ],
    ),

    "large_mixed": ClusterConfig(
        "large_mixed",
        nodes=[
            NodeSpec(gpu_type=GPUType.V100,   num_nodes=8,  gpus_per_node=8),
            NodeSpec(gpu_type=GPUType.A100,   num_nodes=4,  gpus_per_node=8),
            NodeSpec(gpu_type=GPUType.RTX3090, num_nodes=4, gpus_per_node=8),
        ],
    ),

    "gogh_hetero": ClusterConfig(
        "gogh_hetero",
        nodes=[
            NodeSpec(gpu_type=GPUType.K80,           num_nodes=4, gpus_per_node=4),
            NodeSpec(gpu_type=GPUType.P100,          num_nodes=4, gpus_per_node=4),
            NodeSpec(gpu_type=GPUType.V100,          num_nodes=4, gpus_per_node=4),
            NodeSpec(gpu_type=GPUType.K80_UNCONSOL,  num_nodes=2, gpus_per_node=4),
            NodeSpec(gpu_type=GPUType.P100_UNCONSOL, num_nodes=2, gpus_per_node=4),
            NodeSpec(gpu_type=GPUType.V100_UNCONSOL, num_nodes=2, gpus_per_node=4),
        ],
    ),

    # ── HPC clusters with CPU nodes (realistic) ───────────────────────────────

    "hpc_realistic": ClusterConfig(
        "hpc_realistic",
        nodes=[
            # Login nodes — CPU only, user-facing
            NodeSpec(
                node_type=NodeType.CPU_ONLY, role="login",
                num_nodes=4, cpu_type=CPUType.XEON_GOLD,
                num_sockets=2, cores_per_socket=24, ram_gb=256,
            ),
            # CPU compute nodes — data prep, pre/post-processing
            NodeSpec(
                node_type=NodeType.CPU_ONLY, role="cpu_compute",
                num_nodes=8, cpu_type=CPUType.EPYC_7003,
                num_sockets=2, cores_per_socket=64, ram_gb=512,
            ),
            # Mixed GPU+CPU training nodes
            NodeSpec(
                node_type=NodeType.MIXED, role="gpu_compute",
                num_nodes=16, gpu_type=GPUType.V100, gpus_per_node=8,
                cpu_type=CPUType.XEON_GOLD, num_sockets=2, cores_per_socket=24,
                ram_gb=384,
            ),
        ],
    ),

    "a100_mig_cluster": ClusterConfig(
        "a100_mig_cluster",
        nodes=[
            # Login nodes
            NodeSpec(
                node_type=NodeType.CPU_ONLY, role="login",
                num_nodes=2, cpu_type=CPUType.EPYC_7003,
                num_sockets=2, cores_per_socket=64, ram_gb=512,
            ),
            # A100 nodes with MIG enabled — 7 × 1g.10gb slices each
            NodeSpec(
                node_type=NodeType.MIXED, role="mig_compute",
                num_nodes=8, gpu_type=GPUType.A100, gpus_per_node=8,
                cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                ram_gb=512, mig_profile=MIGProfile.G1_10GB,
            ),
            # Some A100 nodes in full-GPU mode for large jobs
            NodeSpec(
                node_type=NodeType.MIXED, role="full_gpu",
                num_nodes=4, gpu_type=GPUType.A100, gpus_per_node=8,
                cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                ram_gb=512,
            ),
        ],
    ),

    "cloud_mixed": ClusterConfig(
        "cloud_mixed",
        nodes=[
            # CPU-heavy parameter server nodes
            NodeSpec(
                node_type=NodeType.CPU_ONLY, role="param_server",
                num_nodes=4, cpu_type=CPUType.EPYC_9004,
                num_sockets=2, cores_per_socket=96, ram_gb=768,
            ),
            # GPU inference nodes with MPS (4 concurrent jobs per GPU)
            NodeSpec(
                node_type=NodeType.MIXED, role="inference",
                num_nodes=8, gpu_type=GPUType.RTX4090, gpus_per_node=4,
                cpu_type=CPUType.EPYC_7003, num_sockets=1, cores_per_socket=32,
                ram_gb=128, mps_max_jobs=4,
            ),
            # GPU training nodes (A100)
            NodeSpec(
                node_type=NodeType.MIXED, role="training",
                num_nodes=16, gpu_type=GPUType.A100, gpus_per_node=8,
                cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                ram_gb=512,
            ),
        ],
    ),

    "h100_supercluster": ClusterConfig(
        "h100_supercluster",
        inter_interconnect=InterconnectType.INFINIBAND_NDR,
        intra_interconnect=InterconnectType.NVLINK4,
        nodes=[
            # CPU login/orchestration nodes
            NodeSpec(
                node_type=NodeType.CPU_ONLY, role="login",
                num_nodes=4, cpu_type=CPUType.EPYC_9004,
                num_sockets=2, cores_per_socket=96, ram_gb=768,
            ),
            # Grace+H100 compute nodes
            NodeSpec(
                node_type=NodeType.MIXED, role="compute",
                num_nodes=32, gpu_type=GPUType.H100, gpus_per_node=8,
                cpu_type=CPUType.GRACE, num_sockets=1, cores_per_socket=72,
                ram_gb=480, intra_interconnect=InterconnectType.NVLINK4,
            ),
        ],
    ),
}


# ─── Cluster ──────────────────────────────────────────────────────────────────

class Cluster:
    """
    HPC cluster resource manager.

    Maintains runtime state of all nodes, GPUs, and CPUs.
    Provides placement-aware allocation for all resource types.
    """

    def __init__(self, config: ClusterConfig):
        self.config    = config
        self.nodes:     dict[str, ServerNode] = {}
        self.gpus:      dict[str, GPUInstance] = {}
        self.cpus:      dict[str, CPUInstance] = {}
        self.mig_slices: dict[str, MIGInstance] = {}
        self.node_map:  dict[str, str] = {}   # gpu_id / cpu_id → node_id
        self._build()

    def _build(self) -> None:
        node_specs = self.config._normalise_nodes()
        gpu_counters: dict[str, int] = {}
        cpu_counters: dict[str, int] = {}

        for spec in node_specs:
            key = f"{spec.node_type.value}_{spec.gpu_type.value if spec.gpu_type else 'cpu'}"
            gpu_counters.setdefault(key, 0)
            cpu_counters.setdefault(spec.cpu_type.value, 0)

            for _ in range(spec.num_nodes):
                gi  = gpu_counters[key]
                nid = f"n_{key}_{gi}"

                node = ServerNode(
                    node_id=nid,
                    node_type=spec.node_type,
                    gpu_type=spec.gpu_type,
                    num_gpus=spec.gpus_per_node,
                    cpu_type=spec.cpu_type,
                    num_sockets=spec.num_sockets,
                    cores_per_socket=spec.cores_per_socket,
                    ram_gb=spec.ram_gb,
                    intra_interconnect=spec.intra_interconnect,
                    inter_interconnect=spec.inter_interconnect,
                )
                node.build()
                self.nodes[nid] = node

                # Register GPUs + optionally enable MIG/MPS
                for g in node.gpus:
                    if spec.mig_profile not in (MIGProfile.NONE, None):
                        if g.spec.mig_capable:
                            g.enable_mig(spec.mig_profile)
                            for m in g.mig_instances:
                                self.mig_slices[m.mig_id] = m
                                self.node_map[m.mig_id] = nid
                    if spec.mps_max_jobs > 1:
                        g.enable_mps(spec.mps_max_jobs)
                    self.gpus[g.gpu_id] = g
                    self.node_map[g.gpu_id] = nid

                # Register CPUs
                for c in node.cpus:
                    self.cpus[c.cpu_id] = c
                    self.node_map[c.cpu_id] = nid

                gpu_counters[key] += 1

    # ── Queries ───────────────────────────────────────────────────────────────

    def total_gpu_count(self) -> int:
        """Total physical GPUs (legacy alias: total_gpus)."""
        return len(self.gpus)

    def total_gpus(self) -> int:   # legacy alias
        return self.total_gpu_count()

    def free_gpu_count(self) -> int:
        return sum(1 for g in self.gpus.values() if g.is_free())

    def total_mig_slices(self) -> int:
        return len(self.mig_slices)

    def free_mig_slices(self) -> int:
        return sum(1 for m in self.mig_slices.values() if m.is_free())

    def total_cpu_cores(self) -> int:
        return sum(c.total_cores for c in self.cpus.values())

    def free_cpu_cores(self) -> int:
        return sum(c.free_cores for c in self.cpus.values())

    def total_cpu_nodes(self) -> int:
        return sum(
            1 for n in self.nodes.values()
            if n.node_type in (NodeType.CPU_ONLY, NodeType.MIXED)
        )

    def has_cpu_nodes(self) -> bool:
        return bool(self.cpus)

    def has_mig(self) -> bool:
        return bool(self.mig_slices)

    def free_gpus_by_type(self) -> dict:
        d: dict[GPUType, list[GPUInstance]] = {}
        for g in self.gpus.values():
            if g.is_free():
                d.setdefault(g.gpu_type, []).append(g)
        return d

    def gpus_by_type(self) -> dict:
        d: dict[GPUType, list[GPUInstance]] = {}
        for g in self.gpus.values():
            d.setdefault(g.gpu_type, []).append(g)
        return d

    def total_gpu_utilization(self) -> float:
        if not self.gpus:
            return 0.0
        return sum(g.utilization for g in self.gpus.values()) / len(self.gpus)

    def total_cpu_utilization(self) -> float:
        if not self.cpus:
            return 0.0
        return sum(c.utilization for c in self.cpus.values()) / len(self.cpus)

    def total_power_watts(self) -> float:
        return sum(n.total_power_watts() for n in self.nodes.values())

    # ── GPU Placement ─────────────────────────────────────────────────────────

    def find_consolidated_gpus(
        self, num_gpus: int, gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Prefer GPUs on the same node (NVLink locality)."""
        for node in self.nodes.values():
            if gpu_type and node.gpu_type != gpu_type:
                continue
            free = [g for g in node.gpus if g.is_free()]
            if len(free) >= num_gpus:
                return [g.gpu_id for g in free[:num_gpus]]
        return None

    def find_scattered_gpus(
        self, num_gpus: int, gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Any free GPUs across nodes."""
        candidates = [
            g for g in self.gpus.values()
            if g.is_free() and (gpu_type is None or g.gpu_type == gpu_type)
        ]
        candidates.sort(key=lambda g: (self.node_map.get(g.gpu_id, ""), g.gpu_id))
        if len(candidates) >= num_gpus:
            return [g.gpu_id for g in candidates[:num_gpus]]
        return None

    def find_best_placement(
        self, num_gpus: int, gpu_type: Optional[GPUType] = None,
        prefer_consolidated: bool = True,
    ) -> Optional[list[str]]:
        if prefer_consolidated:
            result = self.find_consolidated_gpus(num_gpus, gpu_type)
            if result:
                return result
        return self.find_scattered_gpus(num_gpus, gpu_type)

    # ── MIG Placement ─────────────────────────────────────────────────────────

    def find_mig_slices(
        self, num_slices: int, profile: Optional[MIGProfile] = None,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Find free MIG slices matching the requested profile."""
        candidates = [
            m for m in self.mig_slices.values()
            if m.is_free()
            and (profile is None or m.profile == profile)
            and (gpu_type is None or m.gpu_type == gpu_type)
        ]
        if len(candidates) >= num_slices:
            # Prefer slices from the same physical GPU first
            candidates.sort(key=lambda m: (m.parent_gpu_id, m.mig_id))
            return [m.mig_id for m in candidates[:num_slices]]
        return None

    # ── CPU Placement ─────────────────────────────────────────────────────────

    def find_cpu_cores(
        self, num_cores: int, cpu_type: Optional[CPUType] = None,
        prefer_consolidated: bool = True,
    ) -> Optional[list[str]]:
        """
        Find CPU instances that can provide num_cores total.
        Returns list of (cpu_id, cores_to_use) tuples encoded as "cpu_id:N".
        """
        candidates = [
            c for c in self.cpus.values()
            if c.can_allocate(1)
            and (cpu_type is None or c.cpu_type == cpu_type)
        ]
        if prefer_consolidated:
            # Try to fit on as few CPUs as possible (NUMA-friendly)
            candidates.sort(key=lambda c: -c.free_cores)
            allocation = []
            remaining  = num_cores
            for cpu in candidates:
                if remaining <= 0:
                    break
                take = min(cpu.free_cores, remaining)
                allocation.append(f"{cpu.cpu_id}:{take}")
                remaining -= take
            if remaining <= 0:
                return allocation
            return None
        else:
            # Spread across CPUs
            total_free = sum(c.free_cores for c in candidates)
            if total_free < num_cores:
                return None
            allocation = []
            remaining  = num_cores
            for cpu in candidates:
                if remaining <= 0:
                    break
                take = min(cpu.free_cores, remaining)
                allocation.append(f"{cpu.cpu_id}:{take}")
                remaining -= take
            return allocation

    # ── Allocation / Deallocation ─────────────────────────────────────────────

    def allocate(
        self, job_id: str, gpu_ids: list[str], memory_per_gpu_gb: float,
    ) -> bool:
        """Allocate physical GPUs to a job."""
        for gid in gpu_ids:
            if gid not in self.gpus:
                return False
            if self.gpus[gid].memory_free_gb < memory_per_gpu_gb:
                return False
        for gid in gpu_ids:
            g = self.gpus[gid]
            g.allocated_jobs.append(job_id)
            g.memory_used_gb += memory_per_gpu_gb
            g.utilization = min(1.0, 1.0 / len(g.allocated_jobs))
        return True

    def deallocate(
        self, job_id: str, gpu_ids: list[str], memory_per_gpu_gb: float,
    ) -> None:
        """Release physical GPUs."""
        for gid in gpu_ids:
            if gid not in self.gpus:
                continue
            g = self.gpus[gid]
            if job_id in g.allocated_jobs:
                g.allocated_jobs.remove(job_id)
                g.memory_used_gb = max(0.0, g.memory_used_gb - memory_per_gpu_gb)
                g.utilization = 0.0 if not g.allocated_jobs else min(
                    1.0, 1.0 / len(g.allocated_jobs)
                )

    def allocate_mig(
        self, job_id: str, mig_ids: list[str], memory_gb: float = 0.0,
    ) -> bool:
        """Allocate MIG slices to a job."""
        for mid in mig_ids:
            if mid not in self.mig_slices:
                return False
            if not self.mig_slices[mid].is_free():
                return False
        for mid in mig_ids:
            m = self.mig_slices[mid]
            m.allocated_jobs.append(job_id)
            m.memory_used_gb = m.memory_gb
            m.utilization    = 1.0
        return True

    def deallocate_mig(self, job_id: str, mig_ids: list[str]) -> None:
        for mid in mig_ids:
            if mid not in self.mig_slices:
                continue
            m = self.mig_slices[mid]
            if job_id in m.allocated_jobs:
                m.allocated_jobs.remove(job_id)
                m.memory_used_gb = 0.0
                m.utilization    = 0.0

    def allocate_cpu(
        self, job_id: str, cpu_allocation: list[str],
    ) -> bool:
        """
        Allocate CPU cores. cpu_allocation is list of "cpu_id:N" strings
        from find_cpu_cores().
        """
        # Validate first
        parsed = []
        for entry in cpu_allocation:
            cpu_id, cores_str = entry.rsplit(":", 1)
            cores = int(cores_str)
            if cpu_id not in self.cpus:
                return False
            if not self.cpus[cpu_id].can_allocate(cores):
                return False
            parsed.append((cpu_id, cores))
        # Commit
        for cpu_id, cores in parsed:
            c = self.cpus[cpu_id]
            c.allocated_cores += cores
            c.allocated_jobs.append(job_id)
            c.utilization = c.allocated_cores / c.total_cores
        return True

    def deallocate_cpu(
        self, job_id: str, cpu_allocation: list[str],
    ) -> None:
        for entry in cpu_allocation:
            cpu_id, cores_str = entry.rsplit(":", 1)
            cores = int(cores_str)
            if cpu_id not in self.cpus:
                continue
            c = self.cpus[cpu_id]
            if job_id in c.allocated_jobs:
                c.allocated_jobs.remove(job_id)
                c.allocated_cores = max(0, c.allocated_cores - cores)
                c.utilization = c.allocated_cores / max(1, c.total_cores)

    # ── Connectivity ──────────────────────────────────────────────────────────

    def is_consolidated(self, gpu_ids: list[str]) -> bool:
        nodes = {self.node_map[gid] for gid in gpu_ids if gid in self.node_map}
        return len(nodes) <= 1

    def effective_bandwidth_gbps(self, gpu_ids: list[str]) -> float:
        nodes = {self.node_map[gid] for gid in gpu_ids if gid in self.node_map}
        if len(nodes) <= 1:
            nid = next(iter(nodes))
            return INTERCONNECT_SPECS[self.nodes[nid].intra_interconnect].bw_gbps
        return min(
            INTERCONNECT_SPECS[self.nodes[nid].inter_interconnect].bw_gbps
            for nid in nodes if nid in self.nodes
        )

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        total_gpu = len(self.gpus)
        busy_gpu  = sum(1 for g in self.gpus.values() if g.allocated_jobs)
        total_cpu = self.total_cpu_cores()
        free_cpu  = self.free_cpu_cores()

        return {
            "total_gpus":     total_gpu,
            "busy_gpus":      busy_gpu,
            "free_gpus":      total_gpu - busy_gpu,
            "gpu_utilization": self.total_gpu_utilization(),
            "total_mig_slices": self.total_mig_slices(),
            "free_mig_slices":  self.free_mig_slices(),
            "total_cpu_cores":  total_cpu,
            "free_cpu_cores":   free_cpu,
            "cpu_utilization":  self.total_cpu_utilization(),
            "power_watts":      self.total_power_watts(),
        }

    def describe(self) -> str:
        """Human-readable cluster summary."""
        lines = [f"Cluster: {self.config.name}"]
        lines.append(f"  Nodes: {len(self.nodes)}")

        gpu_types = {}
        for g in self.gpus.values():
            gpu_types[g.gpu_type.value] = gpu_types.get(g.gpu_type.value, 0) + 1
        if gpu_types:
            lines.append(f"  GPUs: {sum(gpu_types.values())} total")
            for t, n in sorted(gpu_types.items()):
                lines.append(f"    {t}: {n}")

        if self.mig_slices:
            lines.append(f"  MIG slices: {len(self.mig_slices)}")

        if self.cpus:
            lines.append(f"  CPU cores: {self.total_cpu_cores()} schedulable")
            cpu_types = {}
            for c in self.cpus.values():
                cpu_types[c.cpu_type.value] = cpu_types.get(c.cpu_type.value, 0) + c.total_cores
            for t, n in sorted(cpu_types.items()):
                lines.append(f"    {t}: {n} cores")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Cluster({self.config.name}: "
                f"{self.total_gpu_count()} GPUs, "
                f"{self.total_cpu_cores()} CPU cores, "
                f"{len(self.nodes)} nodes)")
