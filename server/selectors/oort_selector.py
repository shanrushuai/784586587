"""
Oort-style client selector (OSDI'21 baseline).

Utility = statistical_utility * duration_penalty
  statistical_utility = normalized_reward + UCB_exploration
  duration_penalty    = (preferred_duration / actual_duration) ^ round_penalty

Reference: oort/oort.py:289-308 in Oort-master.
"""
import math
import logging
import numpy as np
from typing import Dict, List

from .base import BaseSelector, SelectionContext

logger = logging.getLogger(__name__)


class OortSelector(BaseSelector):
    name = "oort"

    def __init__(self, num_clients: int, args=None,
                 round_penalty: float = 2.0,
                 duration_percentile: float = 80.0,
                 ucb_coeff: float = 0.1,
                 cut_off_util: float = 0.7,
                 **kwargs):
        super().__init__(num_clients, args, **kwargs)
        self.round_penalty = round_penalty
        self.duration_percentile = duration_percentile
        self.ucb_coeff = ucb_coeff
        self.cut_off_util = cut_off_util
        self.participation_count: Dict[int, int] = {i: 0 for i in range(num_clients)}

    def on_round_end(self, ctx: SelectionContext, selected_ids: List[int]):
        for cid in selected_ids:
            self.participation_count[cid] = self.participation_count.get(cid, 0) + 1

    def select(self, ctx: SelectionContext) -> List[int]:
        all_selected = []

        for modality, cids in ctx.candidate_ids_by_modality.items():
            if not cids:
                continue
            num_sample = ctx.num_sample_by_modality.get(modality, 1)

            # 1. statistical utility (normalized influence)
            raw = np.array([ctx.influence_scores.get(c, 0.0) for c in cids])
            r_min, r_max = raw.min(), raw.max()
            r_range = r_max - r_min + 1e-9
            stat_util = (raw - r_min) / r_range

            # 2. UCB exploration bonus
            t = max(ctx.round, 1)
            ucb = np.array([
                math.sqrt(self.ucb_coeff * math.log(t) /
                          max(self.participation_count.get(c, 0), 1))
                for c in cids
            ])

            # 3. duration penalty
            durations = np.array([ctx.estimated_time.get(c, 1.0) for c in cids])
            pref_dur = np.percentile(durations, self.duration_percentile)
            dur_penalty = np.where(
                durations > pref_dur,
                (pref_dur / np.maximum(durations, 1e-4)) ** self.round_penalty,
                1.0,
            )

            utility = (stat_util + ucb) * dur_penalty

            # top-k
            k = min(num_sample, len(cids))
            top_idx = np.argsort(-utility)[:k]
            selected = [cids[i] for i in top_idx]
            all_selected.extend(selected)

        return sorted(all_selected)
