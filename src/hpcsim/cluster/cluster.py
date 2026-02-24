"""
Cluster resource manager.
Manages heterogeneous nodes/GPUs; exposes placement-aware allocation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .hardware import (
    GPUType, GPU_SPECS, GPUInstance, ServerNode,
    InterconnectType, INTERCONNECT_SPECS,
)


@dataclass
class ClusterConfig:
    name: str
    # Each entry: (gpu_type, num_nodes, gpus_per_node)
    nodes: list[tuple]
    intra_interconnect: InterconnectType = InterconnectType.NVLINK
    inter_interconnect: InterconnectType = InterconnectType.INFINIBAND
    cpu_cores_per_node: int = 64
    ram_gb_per_node: float = 512.0


# ── Preset cluster configurations ────────────────────────────────────────────
CLUSTER_CONFIGS: dict[str, ClusterConfig] = {

    "tiny_test": ClusterConfig(
        "tiny_test",
        nodes=[(GPUType.V100, 2, 4)],          # 8 GPUs
    ),
    "small_v100": ClusterConfig(
        "small_v100",
        nodes=[(GPUType.V100, 4, 8)],          # 32 GPUs
    ),
    "medium_heterogeneous_gavel": ClusterConfig(
        "medium_heterogeneous_gavel",
        nodes=[
            (GPUType.K80,  6, 8),   # 48 K80
            (GPUType.P100, 4, 8),   # 32 P100
            (GPUType.V100, 4, 8),   # 32 V100
        ],
    ),
    "large_mixed": ClusterConfig(
        "large_mixed",
        nodes=[
            (GPUType.V100,   8, 8),    # 64
            (GPUType.A100,   4, 8),    # 32
            (GPUType.RTX3090,4, 8),    # 32
        ],
    ),
    "gogh_hetero": ClusterConfig(
        "gogh_hetero",
        nodes=[
            (GPUType.K80,           4, 4),
            (GPUType.P100,          4, 4),
            (GPUType.V100,          4, 4),
            (GPUType.K80_UNCONSOL,  2, 4),
            (GPUType.P100_UNCONSOL, 2, 4),
            (GPUType.V100_UNCONSOL, 2, 4),
        ],
    ),
}


# ── Cluster ───────────────────────────────────────────────────────────────────

class Cluster:
    """
    HPC GPU Cluster resource manager.

    Maintains global state of all nodes/GPUs and provides
    placement-aware allocation for schedulers.

    Key design principles (from survey):
    - Gang scheduling: all GPUs allocated simultaneously (T6)
    - Elastic: variable GPU count allowed (T6)
    - GPU sharing: co-location via MPS/MIG (T5)
    - Placement sensitivity: consolidated vs distributed (T2)
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.nodes: dict[str, ServerNode] = {}
        self.gpus: dict[str, GPUInstance] = {}
        self.node_map: dict[str, str] = {}   # gpu_id → node_id
        self._build()

    def _build(self) -> None:
        counter: dict[GPUType, int] = {}
        for gpu_type, num_nodes, gpus_per_node in self.config.nodes:
            counter.setdefault(gpu_type, 0)
            for _ in range(num_nodes):
                idx = counter[gpu_type]
                nid = f"n_{gpu_type.value}_{idx}"
                node = ServerNode(
                    node_id=nid,
                    gpu_type=gpu_type,
                    num_gpus=gpus_per_node,
                    cpu_cores=self.config.cpu_cores_per_node,
                    ram_gb=self.config.ram_gb_per_node,
                    intra_interconnect=self.config.intra_interconnect,
                    inter_interconnect=self.config.inter_interconnect,
                )
                node.build_gpus()
                self.nodes[nid] = node
                for g in node.gpus:
                    self.gpus[g.gpu_id] = g
                    self.node_map[g.gpu_id] = nid
                counter[gpu_type] += 1

    # ── Queries ──────────────────────────────────────────────────────────────

    def total_gpus(self) -> int:
        return len(self.gpus)

    def free_gpu_count(self) -> int:
        return sum(1 for g in self.gpus.values() if g.is_free())

    def free_gpus_by_type(self) -> dict[GPUType, list[GPUInstance]]:
        d: dict[GPUType, list[GPUInstance]] = {}
        for g in self.gpus.values():
            if g.is_free():
                d.setdefault(g.gpu_type, []).append(g)
        return d

    def gpus_by_type(self) -> dict[GPUType, list[GPUInstance]]:
        d: dict[GPUType, list[GPUInstance]] = {}
        for g in self.gpus.values():
            d.setdefault(g.gpu_type, []).append(g)
        return d

    def total_gpu_utilization(self) -> float:
        if not self.gpus:
            return 0.0
        return sum(g.utilization for g in self.gpus.values()) / len(self.gpus)

    def total_power_watts(self) -> float:
        return sum(n.total_power_watts() for n in self.nodes.values())

    # ── Placement strategies  (Survey T2) ─────────────────────────────────────

    def find_consolidated_gpus(
        self, num_gpus: int,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Prefer GPUs on the same node (NVLink locality). Survey Fig 2(c)."""
        for node in self.nodes.values():
            if gpu_type and node.gpu_type != gpu_type:
                continue
            free = [g for g in node.gpus if g.is_free()]
            if len(free) >= num_gpus:
                return [g.gpu_id for g in free[:num_gpus]]
        return None

    def find_scattered_gpus(
        self, num_gpus: int,
        gpu_type: Optional[GPUType] = None,
    ) -> Optional[list[str]]:
        """Topology-agnostic placement — any free GPUs across nodes."""
        candidates = [
            g for g in self.gpus.values()
            if g.is_free() and (gpu_type is None or g.gpu_type == gpu_type)
        ]
        candidates.sort(key=lambda g: (self.node_map[g.gpu_id], g.gpu_id))
        if len(candidates) >= num_gpus:
            return [g.gpu_id for g in candidates[:num_gpus]]
        return None

    def find_best_placement(
        self,
        num_gpus: int,
        gpu_type: Optional[GPUType] = None,
        prefer_consolidated: bool = True,
    ) -> Optional[list[str]]:
        """
        Philly locality-relaxation: try consolidated, fall back to scattered.
        """
        if prefer_consolidated:
            result = self.find_consolidated_gpus(num_gpus, gpu_type)
            if result:
                return result
        return self.find_scattered_gpus(num_gpus, gpu_type)

    def is_consolidated(self, gpu_ids: list[str]) -> bool:
        nodes = {self.node_map[gid] for gid in gpu_ids if gid in self.node_map}
        return len(nodes) <= 1

    def effective_bandwidth_gbps(self, gpu_ids: list[str]) -> float:
        """Return effective bandwidth (intra-node or inter-node)."""
        nodes = {self.node_map[gid] for gid in gpu_ids if gid in self.node_map}
        if len(nodes) <= 1:
            nid = next(iter(nodes))
            return INTERCONNECT_SPECS[self.nodes[nid].intra_interconnect].bw_gbps
        return min(
            INTERCONNECT_SPECS[self.nodes[nid].inter_interconnect].bw_gbps
            for nid in nodes if nid in self.nodes
        )

    # ── Allocation / Deallocation ────────────────────────────────────────────

    def allocate(
        self,
        job_id: str,
        gpu_ids: list[str],
        memory_per_gpu_gb: float,
    ) -> bool:
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
        self, job_id: str, gpu_ids: list[str], memory_per_gpu_gb: float
    ) -> None:
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

    def snapshot(self) -> dict:
        total = len(self.gpus)
        busy = sum(1 for g in self.gpus.values() if g.allocated_jobs)
        return {
            "total_gpus": total,
            "busy_gpus": busy,
            "free_gpus": total - busy,
            "utilization": self.total_gpu_utilization(),
            "power_watts": self.total_power_watts(),
        }

    def __repr__(self) -> str:
        return (f"Cluster({self.config.name}: "
                f"{self.total_gpus()} GPUs / {len(self.nodes)} nodes, "
                f"free={self.free_gpu_count()})")
