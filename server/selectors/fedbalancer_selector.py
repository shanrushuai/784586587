"""
FedBalancer-style client selector (MobiSys'22 baseline).

Core idea: Oort utility + adaptive deadline penalty.
  1. Statistical utility from sample-level loss (we proxy with FedIF influence).
  2. System utility penalty: (deadline / completion_time) ^ alpha.
  3. Epsilon-greedy exploration with decay.
  4. Adaptive deadline: tighten when loss improves, relax when it degrades.

Reference: FedBalancer-main/oort.py + FedBalancer-main/fedbalancer.py
"""
import math
import logging
import numpy as np
from typing import Dict, List

from .base import BaseSelector, SelectionContext

logger = logging.getLogger(__name__)


class FedBalancerSelector(BaseSelector):
    name = "fedbalancer"

    def __init__(self, num_clients: int, args=None,
                 alpha: float = 2.0,
                 epsilon_init: float = 0.9,
                 epsilon_decay: float = 0.98,
                 epsilon_min: float = 0.2,
                 clip_percentile: float = 0.95,
                 deadline_ratio: float = 0.5,
                 ddl_stepsize: float = 0.05,
                 window: int = 5,
                 **kwargs):
        super().__init__(num_clients, args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.clip_percentile = clip_percentile
        self.deadline_ratio = deadline_ratio
        self.ddl_stepsize = ddl_stepsize
        self.window = window

        self.participation_count: Dict[int, int] = {i: 0 for i in range(num_clients)}
        self.last_selected: Dict[int, int] = {i: 0 for i in range(num_clients)}
        self._loss_history: List[float] = []
        self._deadline: float = 0.0
        self._deadline_initialized = False

    def _init_deadline(self, ctx: SelectionContext):
        if self._deadline_initialized:
            return
        times = list(ctx.estimated_time.values())
        if not times:
            self._deadline = 1000.0
        else:
            # Use median as baseline — tighter than P20-P95 midpoint.
            # deadline_ratio controls how far above median we allow.
            t_med = np.median(times)
            t_high = np.percentile(times, 90)
            self._deadline = t_med + (t_high - t_med) * self.deadline_ratio
        self._deadline_initialized = True
        logger.info(f"[FedBalancer] init deadline={self._deadline:.1f}")

    def on_round_start(self, ctx: SelectionContext):
        self._init_deadline(ctx)

    def on_round_end(self, ctx: SelectionContext, selected_ids: List[int]):
        for cid in selected_ids:
            self.participation_count[cid] = self.participation_count.get(cid, 0) + 1
            self.last_selected[cid] = ctx.round

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Track round loss (proxy: mean influence of selected)
        mean_inf = np.mean([ctx.influence_scores.get(c, 0.0) for c in selected_ids])
        self._loss_history.append(mean_inf)

        # Adaptive deadline control every `window` rounds
        if len(self._loss_history) >= 2 * self.window and ctx.round % self.window == 0:
            recent = np.mean(self._loss_history[-self.window:])
            older = np.mean(self._loss_history[-2*self.window:-self.window])
            if recent > older:
                self.deadline_ratio = min(self.deadline_ratio + self.ddl_stepsize, 1.0)
            else:
                self.deadline_ratio = max(self.deadline_ratio - self.ddl_stepsize, 0.0)
            times = list(ctx.estimated_time.values())
            if times:
                t_low = np.percentile(times, 20)
                t_high = np.percentile(times, 95)
                self._deadline = t_low + (t_high - t_low) * self.deadline_ratio
            logger.info(f"[FedBalancer] adaptive deadline={self._deadline:.1f} "
                        f"ratio={self.deadline_ratio:.2f} eps={self.epsilon:.3f}")

        # Telemetry
        total_e = sum(ctx.estimated_energy.get(c, 0.0) for c in selected_ids)
        max_t = max((ctx.estimated_time.get(c, 0.0) for c in selected_ids), default=0.0)
        self.round_metrics.append(dict(
            round=ctx.round, total_energy=total_e, max_time=max_t,
            deadline=self._deadline, epsilon=self.epsilon,
        ))
        logger.info(f"[FedBalancer] r={ctx.round} eps={self.epsilon:.3f} "
                    f"ddl={self._deadline:.1f} E={total_e:.1f} T={max_t:.1f}")

    def select(self, ctx: SelectionContext) -> List[int]:
        all_selected: List[int] = []
        t = max(ctx.round, 1)

        for modality, cids in ctx.candidate_ids_by_modality.items():
            if not cids:
                continue
            num_sample = ctx.num_sample_by_modality.get(modality, 1)
            num_sample = max(min(num_sample, len(cids)), 1)

            # 1. Statistical utility (influence-based, same as Oort)
            inf_raw = np.array([ctx.influence_scores.get(c, 0.0) for c in cids])
            i_min, i_max = inf_raw.min(), inf_raw.max()
            stat_util = (inf_raw - i_min) / (i_max - i_min + 1e-9)

            # 2. Fairness incentive for overlooked clients
            incentive = np.array([
                math.sqrt(0.1 * math.log(t + 1) /
                          max(self.last_selected.get(c, 0) + 1, 1))
                for c in cids
            ])

            # 3. Deadline-based system penalty
            durations = np.array([ctx.estimated_time.get(c, 1.0) for c in cids])
            deadline = max(self._deadline, 1.0)
            sys_penalty = np.where(
                durations > deadline,
                (deadline / np.maximum(durations, 1e-4)) ** self.alpha,
                1.0,
            )

            utility = (stat_util + incentive) * sys_penalty

            # 4. Clip at percentile
            clip_val = np.percentile(utility, self.clip_percentile * 100)
            utility = np.minimum(utility, clip_val)

            # 5. Epsilon-greedy: exploit (probability sampling) + explore (fast)
            n_exploit = max(int(num_sample * (1 - self.epsilon)), 1)
            n_explore = num_sample - n_exploit

            # Exploit: sample proportional to utility (not pure top-k)
            util_pos = np.maximum(utility, 0.0) + 1e-9
            probs = util_pos / util_pos.sum()
            exploit_idx = list(np.random.choice(
                len(cids), size=min(n_exploit, len(cids)), replace=False, p=probs))
            exploit_ids = [cids[i] for i in exploit_idx]

            if n_explore > 0:
                remaining = [i for i in range(len(cids)) if i not in exploit_idx]
                if remaining:
                    rem_times = durations[remaining]
                    fast_order = np.argsort(rem_times)[:n_explore]
                    explore_ids = [cids[remaining[i]] for i in fast_order]
                else:
                    explore_ids = []
            else:
                explore_ids = []

            all_selected.extend(exploit_ids + explore_ids)

        return sorted(all_selected)
