"""
Resource-aware client selector base class.

Each concrete selector consumes:
  - quality signal (e.g. FedIF influence score, stratified in modality group)
  - resource profile (FLOPs, bandwidth, energy-per-flop/byte)
  - FL-side state (last-selected round, participation count, current round)

and returns a set of selected client ids per modality group.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClientResourceProfile:
    """Static resource profile of one client (assigned at init)."""
    client_id: int
    compute_flops: float          # FLOPs per second (device compute)
    bandwidth_kbps: float         # uplink bandwidth in Kbps
    energy_per_flop: float = 1.0e-10     # Joules per FLOP (nominal)
    energy_per_byte: float = 1.0e-7      # Joules per byte uploaded
    modality: str = "img"         # {"img","txt","img+txt"}

    def completion_time(self, batch_size: int, local_epochs: int,
                        samples: int, model_bytes: float) -> float:
        """Return estimated seconds for one round.

        Time_compute = (3 * B * E * samples / FLOPS_per_sec)   # forward+backward+grad
        Time_comm    = model_bytes * 8 / bandwidth_bps
        """
        compute_ops = 3.0 * float(batch_size) * float(local_epochs) * float(samples)
        t_comp = compute_ops / max(self.compute_flops, 1.0)
        t_comm = (model_bytes * 8.0) / max(self.bandwidth_kbps * 1000.0, 1.0)
        return t_comp + t_comm

    def energy_cost(self, batch_size: int, local_epochs: int,
                    samples: int, model_bytes: float) -> float:
        """E = alpha * FLOPs + beta * bytes_uploaded."""
        compute_ops = 3.0 * float(batch_size) * float(local_epochs) * float(samples)
        return self.energy_per_flop * compute_ops + self.energy_per_byte * model_bytes


@dataclass
class SelectionContext:
    """Per-round context given to selectors."""
    round: int
    candidate_ids_by_modality: Dict[str, List[int]]
    num_sample_by_modality: Dict[str, int]
    influence_scores: Dict[int, float]
    last_selected_round: Dict[int, int]
    fisher_info_by_client: Dict[int, Dict[str, "torch.Tensor"]] = field(default_factory=dict)
    data_sizes: Dict[int, int] = field(default_factory=dict)

    # per-client resource profile reference
    profiles: Dict[int, ClientResourceProfile] = field(default_factory=dict)

    # estimated round cost & energy per candidate (filled by selector preprocess)
    estimated_time: Dict[int, float] = field(default_factory=dict)
    estimated_energy: Dict[int, float] = field(default_factory=dict)


class BaseSelector(ABC):
    """Abstract resource-aware selector."""

    name: str = "base"

    def __init__(self, num_clients: int, args=None, **kwargs):
        self.num_clients = num_clients
        self.args = args
        # telemetry
        self.round_metrics: List[Dict[str, float]] = []

    # ---- hooks ----
    def on_round_start(self, ctx: SelectionContext):
        """Override if selector needs per-round preprocessing."""
        pass

    def on_round_end(self, ctx: SelectionContext, selected_ids: List[int]):
        """Override to update virtual queue / bandit counters etc."""
        pass

    @abstractmethod
    def select(self, ctx: SelectionContext) -> List[int]:
        """Return sorted list of selected client ids."""
        raise NotImplementedError

    # ---- helpers ----
    @staticmethod
    def _fisher_density(fisher_dict: Optional[Dict[str, "torch.Tensor"]]) -> float:
        """Normalized Fisher information density in [0, 1].

        Psi_k = mean_over_params( fisher ) / ( mean + eps )  -> collapses to a scalar
        proxy of "how much information this client carries per uploaded param".
        """
        if not fisher_dict:
            return 1.0
        import torch
        vals = []
        for v in fisher_dict.values():
            if v is None:
                continue
            vals.append(float(v.float().mean().item()))
        if not vals:
            return 1.0
        m = sum(vals) / len(vals)
        return float(m)
