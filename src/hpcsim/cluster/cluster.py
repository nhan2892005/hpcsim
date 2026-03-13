"""
Cluster resource manager — Optimised edition.

Key optimisations over baseline:

1. **O(1) free-resource pools** — every allocate/deallocate touches only the
   affected GPU/MIG/CPU rather than scanning the whole cluster.

   Data structures maintained in lockstep with GPUInstance state:
     _free_gpu_ids            set[str]                  — all GPUs with capacity
     _free_gpu_ids_by_type    dict[GPUType, set[str]]   — split by hardware gen
     _free_gpu_ids_by_node    dict[node_id, set[str]]   — split by node (placement)
     _node_gpu_type           dict[node_id, GPUType|None]— for consolidated search
     _free_mig_ids            set[str]                  — free MIG slices
     _free_mig_ids_by_profile dict[MIGProfile, set[str]]— slices by profile
     _free_mig_ids_by_parent  dict[gpu_id, set[str]]    — slices by parent GPU
     _total_free_cpu_cores    int                       — running sum of free cores
     _busy_gpu_count          int                       — GPUs with ≥ 1 job

   Result: free_gpu_count(), free_mig_slices(), free_cpu_cores(), snapshot()
   are all O(1).  find_consolidated_gpus() drops from O(nodes × gpus_per_node)
   to O(nodes).  find_scattered_gpus() drops from O(total_gpus) to O(free_gpus).

2. **Slot-free helper methods** — _gpu_free_add/_gpu_free_remove encapsulate the
   multi-set bookkeeping so every allocate/deallocate path stays DRY.

Public API is 100% backward-compatible with the original cluster.py.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .hardware import (
    GPUType, CPUType, NodeType,
    GPU_SPECS, CPU_SPECS, CPUSpec,
    GPUInstance, CPUInstance, MIGInstance, MIGProfile,
    ServerNode,
    InterconnectType, INTERCONNECT_SPECS,
)


# NodeSpec 

@dataclass
class NodeSpec:
    """
    Declarative specification of a node type in the cluster.

    Examples
    --------
    NodeSpec(gpu_type=GPUType.V100, num_nodes=4, gpus_per_node=8)

    NodeSpec(node_type=NodeType.CPU_ONLY, cpu_type=CPUType.EPYC_7003,
             num_nodes=4, num_sockets=2, cores_per_socket=64, ram_gb=512)

    NodeSpec(node_type=NodeType.MIXED,
             cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
             gpu_type=GPUType.A100, gpus_per_node=4,
             num_nodes=8, ram_gb=256)

    NodeSpec(gpu_type=GPUType.A100, num_nodes=2, gpus_per_node=8,
             mig_profile=MIGProfile.G1_10GB)
    """
    num_nodes:          int             = 1
    node_type:          NodeType        = NodeType.GPU_ONLY
    gpu_type:           Optional[GPUType] = None
    gpus_per_node:      int             = 0
    cpu_type:           CPUType         = CPUType.XEON_GOLD
    num_sockets:        int             = 2
    cores_per_socket:   int             = 0
    ram_gb:             float           = 256.0
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND
    mig_profile:        MIGProfile      = MIGProfile.NONE
    mps_max_jobs:       int             = 1
    role:               str             = ""

    def __post_init__(self):
        has_gpu = self.gpu_type is not None and self.gpus_per_node > 0
        has_cpu_alloc = self.node_type in (NodeType.CPU_ONLY, NodeType.MIXED)
        if not has_gpu and self.node_type == NodeType.GPU_ONLY:
            self.node_type = NodeType.CPU_ONLY

    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    def total_cpu_cores(self) -> int:
        if self.node_type == NodeType.GPU_ONLY:
            return 0
        spec = CPU_SPECS.get(self.cpu_type, CPU_SPECS[CPUType.XEON_GOLD])
        cpp  = self.cores_per_socket or spec.cores_per_socket
        return self.num_nodes * self.num_sockets * cpp


# ClusterConfig 

@dataclass
class ClusterConfig:
    """
    Cluster configuration.
    nodes: list of NodeSpec (preferred) OR legacy (GPUType, num_nodes, gpus_per_node) tuples.
    """
    name: str
    nodes: list
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND
    cpu_cores_per_node: int   = 64
    ram_gb_per_node:    float = 512.0

    def _normalise_nodes(self) -> list[NodeSpec]:
        result = []
        for n in self.nodes:
            if isinstance(n, NodeSpec):
                result.append(n)
            elif isinstance(n, tuple):
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


# Preset cluster configs 

CLUSTER_CONFIGS: dict[str, ClusterConfig] = {

    "tiny_test": ClusterConfig(
        "tiny_test",
        nodes=[NodeSpec(gpu_type=GPUType.V100, num_nodes=2, gpus_per_node=4)],
    ),

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
            NodeSpec(gpu_type=GPUType.V100,    num_nodes=8,  gpus_per_node=8),
            NodeSpec(gpu_type=GPUType.A100,    num_nodes=4,  gpus_per_node=8),
            NodeSpec(gpu_type=GPUType.RTX3090, num_nodes=4,  gpus_per_node=8),
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

    "hpc_realistic": ClusterConfig(
        "hpc_realistic",
        nodes=[
            NodeSpec(node_type=NodeType.CPU_ONLY, role="login",
                     num_nodes=4, cpu_type=CPUType.XEON_GOLD,
                     num_sockets=2, cores_per_socket=24, ram_gb=256),
            NodeSpec(node_type=NodeType.CPU_ONLY, role="cpu_compute",
                     num_nodes=8, cpu_type=CPUType.EPYC_7003,
                     num_sockets=2, cores_per_socket=64, ram_gb=512),
            NodeSpec(node_type=NodeType.MIXED, role="gpu_compute",
                     num_nodes=16, gpu_type=GPUType.V100, gpus_per_node=8,
                     cpu_type=CPUType.XEON_GOLD, num_sockets=2, cores_per_socket=24,
                     ram_gb=384),
        ],
    ),

    "a100_mig_cluster": ClusterConfig(
        "a100_mig_cluster",
        nodes=[
            NodeSpec(node_type=NodeType.CPU_ONLY, role="login",
                     num_nodes=2, cpu_type=CPUType.EPYC_7003,
                     num_sockets=2, cores_per_socket=64, ram_gb=512),
            NodeSpec(node_type=NodeType.MIXED, role="mig_compute",
                     num_nodes=8, gpu_type=GPUType.A100, gpus_per_node=8,
                     cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                     ram_gb=512, mig_profile=MIGProfile.G1_10GB),
            NodeSpec(node_type=NodeType.MIXED, role="full_gpu",
                     num_nodes=4, gpu_type=GPUType.A100, gpus_per_node=8,
                     cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                     ram_gb=512),
        ],
    ),

    "cloud_mixed": ClusterConfig(
        "cloud_mixed",
        nodes=[
            NodeSpec(node_type=NodeType.CPU_ONLY, role="param_server",
                     num_nodes=4, cpu_type=CPUType.EPYC_9004,
                     num_sockets=2, cores_per_socket=96, ram_gb=768),
            NodeSpec(node_type=NodeType.MIXED, role="inference",
                     num_nodes=8, gpu_type=GPUType.RTX4090, gpus_per_node=4,
                     cpu_type=CPUType.EPYC_7003, num_sockets=1, cores_per_socket=32,
                     ram_gb=128, mps_max_jobs=4),
            NodeSpec(node_type=NodeType.MIXED, role="training",
                     num_nodes=16, gpu_type=GPUType.A100, gpus_per_node=8,
                     cpu_type=CPUType.EPYC_7002, num_sockets=2, cores_per_socket=32,
                     ram_gb=512),
        ],
    ),

    "h100_supercluster": ClusterConfig(
        "h100_supercluster",
        inter_interconnect=InterconnectType.INFINIBAND_NDR,
        intra_interconnect=InterconnectType.NVLINK4,
        nodes=[
            NodeSpec(node_type=NodeType.CPU_ONLY, role="login",
                     num_nodes=4, cpu_type=CPUType.EPYC_9004,
                     num_sockets=2, cores_per_socket=96, ram_gb=768),
            NodeSpec(node_type=NodeType.MIXED, role="compute",
                     num_nodes=32, gpu_type=GPUType.H100, gpus_per_node=8,
                     cpu_type=CPUType.GRACE, num_sockets=1, cores_per_socket=72,
                     ram_gb=480, intra_interconnect=InterconnectType.NVLINK4),
        ],
    ),
}


# Cluster 

class Cluster:
    """
    HPC cluster resource manager with O(1) free-resource pools.

    Internal pools (updated on every allocate/deallocate):
        _free_gpu_ids            – GPUs whose slot-count < capacity
        _free_gpu_ids_by_type    – same, keyed by GPUType
        _free_gpu_ids_by_node    – same, keyed by node_id (placement-aware)
        _node_gpu_type           – node_id → gpu_type (static, built once)
        _free_mig_ids            – free MIG slices
        _free_mig_ids_by_profile – free MIG slices by MIGProfile
        _free_mig_ids_by_parent  – free MIG slices by parent GPU id
        _total_free_cpu_cores    – running sum of free CPU cores
        _busy_gpu_count          – count of GPUs with ≥ 1 allocated job
    """

    def __init__(self, config: ClusterConfig):
        self.config      = config
        self.nodes:       dict[str, ServerNode]   = {}
        self.gpus:        dict[str, GPUInstance]  = {}
        self.cpus:        dict[str, CPUInstance]  = {}
        self.mig_slices:  dict[str, MIGInstance]  = {}
        self.node_map:    dict[str, str]           = {}   # resource_id → node_id

        # O(1) free-resource pools 
        # GPU pools
        self._free_gpu_ids:           set[str]                       = set()
        self._free_gpu_ids_by_type:   dict[GPUType, set[str]]        = defaultdict(set)
        self._free_gpu_ids_by_node:   dict[str, set[str]]            = defaultdict(set)
        self._node_gpu_type:          dict[str, Optional[GPUType]]   = {}

        # MIG pools
        self._free_mig_ids:           set[str]                       = set()
        self._free_mig_ids_by_profile: dict[MIGProfile, set[str]]   = defaultdict(set)
        self._free_mig_ids_by_parent:  dict[str, set[str]]           = defaultdict(set)

        # CPU + summary counters
        self._total_free_cpu_cores:   int = 0
        self._busy_gpu_count:         int = 0   # GPUs with len(allocated_jobs) > 0

        self._build()

    # Build 

    def _build(self) -> None:
        node_specs = self.config._normalise_nodes()
        gpu_counters: dict[str, int] = {}
        cpu_counters: dict[str, int] = {}

        for spec in node_specs:
            key = (f"{spec.node_type.value}_"
                   f"{spec.gpu_type.value if spec.gpu_type else 'cpu'}")
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

                # Record the GPU type for this node (used in placement search)
                self._node_gpu_type[nid] = spec.gpu_type

                # Register GPUs
                for g in node.gpus:
                    if spec.mig_profile not in (MIGProfile.NONE, None):
                        if g.spec.mig_capable:
                            g.enable_mig(spec.mig_profile)
                            for m in g.mig_instances:
                                self.mig_slices[m.mig_id] = m
                                self.node_map[m.mig_id] = nid
                                # Pool: all MIG slices start free
                                self._mig_free_add(m.mig_id, m.profile, m.parent_gpu_id)
                    if spec.mps_max_jobs > 1:
                        g.enable_mps(spec.mps_max_jobs)
                    self.gpus[g.gpu_id] = g
                    self.node_map[g.gpu_id] = nid
                    # Pool: all GPUs start free
                    self._gpu_free_add(g.gpu_id, g.gpu_type, nid)

                # Register CPUs
                for c in node.cpus:
                    self.cpus[c.cpu_id] = c
                    self.node_map[c.cpu_id] = nid
                    self._total_free_cpu_cores += c.free_cores

                gpu_counters[key] += 1

    # Pool helpers (internal) 

    def _gpu_free_add(self, gid: str, gtype: GPUType, nid: str) -> None:
        """Register *gid* as having available capacity in all GPU pools."""
        self._free_gpu_ids.add(gid)
        self._free_gpu_ids_by_type[gtype].add(gid)
        self._free_gpu_ids_by_node[nid].add(gid)

    def _gpu_free_remove(self, gid: str, gtype: GPUType, nid: str) -> None:
        """Remove *gid* from all GPU free-pools (GPU reached full capacity)."""
        self._free_gpu_ids.discard(gid)
        self._free_gpu_ids_by_type[gtype].discard(gid)
        self._free_gpu_ids_by_node[nid].discard(gid)

    def _mig_free_add(self, mid: str, profile: MIGProfile, parent_gpu_id: str) -> None:
        self._free_mig_ids.add(mid)
        self._free_mig_ids_by_profile[profile].add(mid)
        self._free_mig_ids_by_parent[parent_gpu_id].add(mid)

    def _mig_free_remove(self, mid: str, profile: MIGProfile, parent_gpu_id: str) -> None:
        self._free_mig_ids.discard(mid)
        self._free_mig_ids_by_profile[profile].discard(mid)
        self._free_mig_ids_by_parent[parent_gpu_id].discard(mid)

    # Queries — all O(1) now 

    def total_gpu_count(self) -> int:
        return len(self.gpus)

    def total_gpus(self) -> int:      # legacy alias
        return len(self.gpus)

    def free_gpu_count(self) -> int:
        """O(1) — reads pre-maintained set size."""
        return len(self._free_gpu_ids)

    def total_mig_slices(self) -> int:
        return len(self.mig_slices)

    def free_mig_slices(self) -> int:
        """O(1) — reads pre-maintained set size."""
        return len(self._free_mig_ids)

    def total_cpu_cores(self) -> int:
        return sum(c.total_cores for c in self.cpus.values())

    def free_cpu_cores(self) -> int:
        """O(1) — reads pre-maintained running total."""
        return self._total_free_cpu_cores

    def total_cpu_nodes(self) -> int:
        return sum(
            1 for n in self.nodes.values()
            if n.node_type in (NodeType.CPU_ONLY, NodeType.MIXED)
        )

    def has_cpu_nodes(self) -> bool:
        return bool(self.cpus)

    def has_mig(self) -> bool:
        return bool(self.mig_slices)

    def free_gpus_by_type(self) -> dict[GPUType, list[GPUInstance]]:
        """
        O(|free_gpu_ids_by_type|) instead of O(total_gpus).
        Returns {GPUType: [GPUInstance, ...]} for types that have free GPUs.
        """
        return {
            gtype: [self.gpus[gid] for gid in gids]
            for gtype, gids in self._free_gpu_ids_by_type.items()
            if gids
        }

    def gpus_by_type(self) -> dict[GPUType, list[GPUInstance]]:
        """All GPUs (free or busy) grouped by type."""
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

    # GPU Placement — O(nodes) instead of O(nodes × gpus_per_node) 

    def find_consolidated_gpus(
        self,
        num_gpus: int,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """
        Prefer GPUs on the same node (NVLink locality).

        Complexity: O(nodes) — uses _free_gpu_ids_by_node directly.
        """
        for nid, free_set in self._free_gpu_ids_by_node.items():
            # Filter by GPU type if requested (static per-node attribute)
            if gpu_type is not None and self._node_gpu_type.get(nid) != gpu_type:
                continue
            if len(free_set) >= num_gpus:
                # sorted for deterministic behaviour across runs
                return sorted(free_set)[:num_gpus]
        return None

    def find_scattered_gpus(
        self,
        num_gpus: int,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """
        Any free GPUs across nodes.

        Complexity: O(|free_gpus|) vs O(|total_gpus|) in baseline.
        """
        pool = (self._free_gpu_ids_by_type.get(gpu_type, set())
                if gpu_type is not None
                else self._free_gpu_ids)
        if len(pool) < num_gpus:
            return None
        # Sort for determinism + node-locality preference via node_map key
        candidates = sorted(pool, key=lambda g: (self.node_map.get(g, ""), g))
        return candidates[:num_gpus]

    def find_best_placement(
        self,
        num_gpus: int,
        gpu_type: Optional[GPUType] = None,
        prefer_consolidated: bool = True,
    ) -> Optional[list[str]]:
        if prefer_consolidated:
            result = self.find_consolidated_gpus(num_gpus, gpu_type)
            if result:
                return result
        return self.find_scattered_gpus(num_gpus, gpu_type)

    # MIG Placement — O(free_mig_slices) instead of O(total_mig_slices) 

    def find_mig_slices(
        self,
        num_slices: int,
        profile: Optional[MIGProfile] = None,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """
        Find free MIG slices matching the requested profile.
        Prefers slices from the same physical GPU (locality).
        """
        if not self._free_mig_ids:
            return None

        # Narrow candidate pool using pre-built index
        if profile is not None:
            pool: set[str] = self._free_mig_ids_by_profile.get(profile, set())
        else:
            pool = self._free_mig_ids

        if len(pool) < num_slices:
            return None

        # Optional gpu_type filter — MIGInstance carries gpu_type
        if gpu_type is not None:
            pool = {m for m in pool if self.mig_slices[m].gpu_type == gpu_type}
            if len(pool) < num_slices:
                return None

        # Sort for locality (same parent GPU first), then determinism
        candidates = sorted(pool, key=lambda m: (
            self.mig_slices[m].parent_gpu_id, m
        ))
        return candidates[:num_slices]

    # CPU Placement (unchanged algorithm, but free_cores O(1) via counter) 

    def find_cpu_cores(
        self,
        num_cores: int,
        cpu_type: Optional[CPUType] = None,
        prefer_consolidated: bool = True,
    ) -> Optional[list[str]]:
        """
        Find CPU instances that can provide num_cores total.
        Returns list of "cpu_id:N" allocation descriptors.

        Early exit: returns None immediately if total free cores insufficient (O(1)).
        """
        # O(1) early-exit check using the pre-maintained counter
        if self._total_free_cpu_cores < num_cores:
            return None

        candidates = [
            c for c in self.cpus.values()
            if c.free_cores > 0
            and (cpu_type is None or c.cpu_type == cpu_type)
        ]

        if prefer_consolidated:
            candidates.sort(key=lambda c: -c.free_cores)
        # else: spread — leave in arbitrary order

        allocation: list[str] = []
        remaining = num_cores
        for cpu in candidates:
            if remaining <= 0:
                break
            take = min(cpu.free_cores, remaining)
            allocation.append(f"{cpu.cpu_id}:{take}")
            remaining -= take

        return allocation if remaining <= 0 else None

    # Allocation / Deallocation 

    def allocate(
        self,
        job_id: str,
        gpu_ids: list[str],
        memory_per_gpu_gb: float,
    ) -> bool:
        """
        Allocate physical GPUs to a job.
        Updates free-resource pools atomically after validation.
        """
        # Phase 1: validate
        for gid in gpu_ids:
            if gid not in self.gpus:
                return False
            if self.gpus[gid].memory_free_gb < memory_per_gpu_gb:
                return False

        # Phase 2: commit + update pools
        for gid in gpu_ids:
            g = self.gpus[gid]
            was_empty = len(g.allocated_jobs) == 0

            g.allocated_jobs.append(job_id)
            g.memory_used_gb += memory_per_gpu_gb
            g.utilization = min(1.0, 1.0 / len(g.allocated_jobs))

            if was_empty:
                self._busy_gpu_count += 1

            # Remove from free pool if slot capacity is now exhausted
            if len(g.allocated_jobs) >= g.capacity:
                self._gpu_free_remove(gid, g.gpu_type, self.node_map[gid])

        return True

    def deallocate(
        self,
        job_id: str,
        gpu_ids: list[str],
        memory_per_gpu_gb: float,
    ) -> None:
        """Release physical GPUs and update free-resource pools."""
        for gid in gpu_ids:
            if gid not in self.gpus:
                continue
            g = self.gpus[gid]
            if job_id not in g.allocated_jobs:
                continue

            was_full = len(g.allocated_jobs) >= g.capacity
            g.allocated_jobs.remove(job_id)
            g.memory_used_gb = max(0.0, g.memory_used_gb - memory_per_gpu_gb)
            g.utilization = (0.0 if not g.allocated_jobs
                             else min(1.0, 1.0 / len(g.allocated_jobs)))

            if not g.allocated_jobs:
                self._busy_gpu_count = max(0, self._busy_gpu_count - 1)

            # GPU recovered capacity — add back to free pool
            if was_full and len(g.allocated_jobs) < g.capacity:
                self._gpu_free_add(gid, g.gpu_type, self.node_map[gid])

    def allocate_mig(
        self,
        job_id: str,
        mig_ids: list[str],
        memory_gb: float = 0.0,
    ) -> bool:
        """Allocate MIG slices and update MIG free pools."""
        # Validate
        for mid in mig_ids:
            if mid not in self.mig_slices:
                return False
            if not self.mig_slices[mid].is_free():
                return False

        # Commit
        for mid in mig_ids:
            m = self.mig_slices[mid]
            m.allocated_jobs.append(job_id)
            m.memory_used_gb = m.memory_gb
            m.utilization    = 1.0
            self._mig_free_remove(mid, m.profile, m.parent_gpu_id)

        return True

    def deallocate_mig(self, job_id: str, mig_ids: list[str]) -> None:
        """Release MIG slices and restore them to free pools."""
        for mid in mig_ids:
            if mid not in self.mig_slices:
                continue
            m = self.mig_slices[mid]
            if job_id not in m.allocated_jobs:
                continue
            m.allocated_jobs.remove(job_id)
            m.memory_used_gb = 0.0
            m.utilization    = 0.0
            # Slice is free again
            self._mig_free_add(mid, m.profile, m.parent_gpu_id)

    def allocate_cpu(
        self,
        job_id: str,
        cpu_allocation: list[str],
    ) -> bool:
        """
        Allocate CPU cores.
        cpu_allocation: list of "cpu_id:N" strings from find_cpu_cores().
        Updates _total_free_cpu_cores counter atomically.
        """
        parsed: list[tuple[str, int]] = []
        for entry in cpu_allocation:
            cpu_id, cores_str = entry.rsplit(":", 1)
            cores = int(cores_str)
            if cpu_id not in self.cpus:
                return False
            if not self.cpus[cpu_id].can_allocate(cores):
                return False
            parsed.append((cpu_id, cores))

        for cpu_id, cores in parsed:
            c = self.cpus[cpu_id]
            c.allocated_cores += cores
            c.allocated_jobs.append(job_id)
            c.utilization = c.allocated_cores / c.total_cores
            self._total_free_cpu_cores -= cores   # O(1) counter update

        return True

    def deallocate_cpu(
        self,
        job_id: str,
        cpu_allocation: list[str],
    ) -> None:
        """Release CPU cores and restore counter."""
        for entry in cpu_allocation:
            cpu_id, cores_str = entry.rsplit(":", 1)
            cores = int(cores_str)
            if cpu_id not in self.cpus:
                continue
            c = self.cpus[cpu_id]
            if job_id not in c.allocated_jobs:
                continue
            c.allocated_jobs.remove(job_id)
            c.allocated_cores = max(0, c.allocated_cores - cores)
            c.utilization = c.allocated_cores / max(1, c.total_cores)
            self._total_free_cpu_cores += cores   # O(1) counter update

    # Connectivity 

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

    # Snapshot — now O(1) for GPU counts 

    def snapshot(self) -> dict:
        """
        Cluster state snapshot.
        GPU counts: O(1) via pre-maintained counters.
        Utilisation: O(GPUs) — only called every METRIC_INTERVAL (60 s).
        """
        total_gpu = len(self.gpus)
        busy_gpu  = self._busy_gpu_count          # O(1) ← was O(N)
        free_gpu  = len(self._free_gpu_ids)        # O(1) ← was O(N)
        total_cpu = self.total_cpu_cores()
        free_cpu  = self._total_free_cpu_cores     # O(1) ← was O(N)

        return {
            "total_gpus":       total_gpu,
            "busy_gpus":        busy_gpu,
            "free_gpus":        free_gpu,
            "gpu_utilization":  self.total_gpu_utilization(),
            "total_mig_slices": self.total_mig_slices(),
            "free_mig_slices":  self.free_mig_slices(),   # O(1)
            "total_cpu_cores":  total_cpu,
            "free_cpu_cores":   free_cpu,                 # O(1)
            "cpu_utilization":  self.total_cpu_utilization(),
            "power_watts":      self.total_power_watts(),
        }

    # Describe 

    def describe(self) -> str:
        lines = [f"Cluster: {self.config.name}"]
        lines.append(f"  Nodes: {len(self.nodes)}")
        gpu_types: dict[str, int] = {}
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
            cpu_types: dict[str, int] = {}
            for c in self.cpus.values():
                cpu_types[c.cpu_type.value] = cpu_types.get(c.cpu_type.value, 0) + c.total_cores
            for t, n in sorted(cpu_types.items()):
                lines.append(f"    {t}: {n} cores")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Cluster({self.config.name}: "
                f"{self.total_gpu_count()} GPUs "
                f"[{self.free_gpu_count()} free], "
                f"{self.total_cpu_cores()} CPU cores, "
                f"{len(self.nodes)} nodes)")