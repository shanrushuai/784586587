"""
Load & assign FedScale-style real device profiles to FL clients.

Priority:
  1) If FedScale 'client_device_capacity' pickle exists -> sample real devices.
  2) Else fall back to synthetic 3-tier distribution (high/mid/low).

Each assignment is deterministic given a seed so comparisons are fair across
selector ablations.
"""
import os
import pickle
import logging
import random
from typing import Dict, List, Optional

from .base import ClientResourceProfile

logger = logging.getLogger(__name__)

# FedScale device_info path inside this repo (relative to repo root)
DEFAULT_FEDSCALE_PKL = os.path.join(
    "FedScale-master", "benchmark", "dataset", "data",
    "device_info", "client_device_capacity",
)


def _try_load_fedscale_capacity(path: str) -> Optional[Dict[int, dict]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            cap = pickle.load(f)
        if not isinstance(cap, dict) or len(cap) == 0:
            return None
        return cap
    except Exception as e:
        logger.warning(f"[ResourceProfiler] failed to read FedScale capacity: {e}")
        return None


# Synthetic fallback tiers (FLOPS in GFLOPS, bandwidth in Kbps)
_SYNTHETIC_TIERS = [
    dict(flops=5.0e11,  bw_kbps=50000.0,  weight=0.25),   # high-end (Snapdragon 8-gen style)
    dict(flops=1.0e11,  bw_kbps=8000.0,   weight=0.50),   # mid-range 4G
    dict(flops=2.0e10,  bw_kbps=1500.0,   weight=0.25),   # low-end / edge 3G
]


class ResourceProfiler:
    """Assign a static resource profile to every client id.

    Args:
      num_clients: total K.
      client_modalities: {cid: modality_str}.
      seed: rng seed for reproducible assignment.
      capacity_path: path to FedScale client_device_capacity pickle (optional).
      alpha_e_flop / beta_e_byte: energy coefficients.
    """

    def __init__(
        self,
        num_clients: int,
        client_modalities: Dict[int, str],
        seed: int = 42,
        capacity_path: Optional[str] = None,
        alpha_e_flop: float = 1.0e-10,
        beta_e_byte: float = 1.0e-7,
    ):
        self.num_clients = num_clients
        self.client_modalities = client_modalities
        self.seed = seed
        self.alpha_e_flop = alpha_e_flop
        self.beta_e_byte = beta_e_byte

        if capacity_path is None:
            capacity_path = DEFAULT_FEDSCALE_PKL

        self.capacity = _try_load_fedscale_capacity(capacity_path)
        self.source = "fedscale" if self.capacity else "synthetic"
        logger.info(f"[ResourceProfiler] source={self.source} K={num_clients}")

        self.profiles: Dict[int, ClientResourceProfile] = self._assign()

    def _assign(self) -> Dict[int, ClientResourceProfile]:
        rng = random.Random(self.seed)
        out: Dict[int, ClientResourceProfile] = {}

        if self.capacity:
            keys = list(self.capacity.keys())
            # Compute FLOPS range for energy scaling
            all_flops = []
            sampled_keys = []
            rng_pre = random.Random(self.seed)
            for _ in range(self.num_clients):
                sampled_keys.append(rng_pre.choice(keys))
            for k in sampled_keys:
                comp_raw = float(self.capacity[k].get("computation", 100.0))
                all_flops.append(max(comp_raw * 1e9, 2.0e9))
            flops_max = max(all_flops)

            rng = random.Random(self.seed)
            for cid in range(self.num_clients):
                k = rng.choice(keys)
                entry = self.capacity[k]
                # FedScale: 'computation' field varies by device.
                # Empirically the raw values span ~15 to ~199.
                # Map to a realistic mobile FLOPS range [2e9, 2e11].
                comp_raw = float(entry.get("computation", 100.0))
                comm_raw = float(entry.get("communication", 5000.0))
                flops = max(comp_raw * 1e9, 2.0e9)
                bw = max(comm_raw, 200.0)
                # Energy coefficients scale inversely with device capability:
                # weaker devices are less energy-efficient (higher alpha/beta).
                # ratio in [1, flops_max/flops_min] ≈ [1, 13]
                eff_ratio = flops_max / max(flops, 1.0)
                alpha_k = self.alpha_e_flop * eff_ratio
                beta_k = self.beta_e_byte * (1.0 + 0.5 * (eff_ratio - 1.0))
                out[cid] = ClientResourceProfile(
                    client_id=cid,
                    compute_flops=flops,
                    bandwidth_kbps=bw,
                    energy_per_flop=alpha_k,
                    energy_per_byte=beta_k,
                    modality=self.client_modalities.get(cid, "img"),
                )
        else:
            weights = [t["weight"] for t in _SYNTHETIC_TIERS]
            tier_max_flops = max(t["flops"] for t in _SYNTHETIC_TIERS)
            for cid in range(self.num_clients):
                tier = rng.choices(_SYNTHETIC_TIERS, weights=weights, k=1)[0]
                jitter = lambda x: x * rng.uniform(0.7, 1.3)
                f = jitter(tier["flops"])
                eff_ratio = tier_max_flops / max(f, 1.0)
                out[cid] = ClientResourceProfile(
                    client_id=cid,
                    compute_flops=f,
                    bandwidth_kbps=jitter(tier["bw_kbps"]),
                    energy_per_flop=self.alpha_e_flop * eff_ratio,
                    energy_per_byte=self.beta_e_byte * (1.0 + 0.5 * (eff_ratio - 1.0)),
                    modality=self.client_modalities.get(cid, "img"),
                )
        return out

    def get(self, cid: int) -> ClientResourceProfile:
        return self.profiles[cid]

    def summary(self) -> str:
        if not self.profiles:
            return "<empty>"
        flops = [p.compute_flops for p in self.profiles.values()]
        bws = [p.bandwidth_kbps for p in self.profiles.values()]
        alphas = [p.energy_per_flop for p in self.profiles.values()]
        betas = [p.energy_per_byte for p in self.profiles.values()]
        return (f"[profiles] src={self.source} "
                f"FLOPS min={min(flops):.2e} max={max(flops):.2e} "
                f"BW min={min(bws):.0f} max={max(bws):.0f} Kbps "
                f"alpha min={min(alphas):.2e} max={max(alphas):.2e} "
                f"beta min={min(betas):.2e} max={max(betas):.2e}")
