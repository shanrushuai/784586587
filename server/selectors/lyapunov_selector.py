"""
Robust multi-resource adaptive Lyapunov selector (RMAL-ED).

The original selector optimized an energy-only drift-plus-penalty score.  This
version keeps that energy guarantee but adds delay control as a first-class
Lyapunov constraint:

    score_k = V_t * quality_k
              - w_E * (Q_E^m + q_E^k + rho) * e_hat_k
              - w_T * (Q_T^m + rho) * Delta tau_hat_k
              - w_C * (Z + rho) * tail_excess_k

where e_hat and tau_hat are robust median-MAD normalized resources, Q_E/Q_T
are modality-level resource queues, q_E is a per-client energy debt queue, and
Z is a CVaR tail-latency queue.  Delta tau_hat is set-dependent: it penalizes
the marginal increase in the current selected set's max latency, which matches
the FL round-time objective.
"""
import math
import logging
import numpy as np
from typing import Dict, List, Tuple

from .base import BaseSelector, SelectionContext

logger = logging.getLogger(__name__)


class LyapunovSelector(BaseSelector):
    name = "lyapunov"

    def __init__(self, num_clients: int, args=None,
                 V: float = 3.0,
                 momentum_mu: float = 0.1,
                 ucb_beta: float = 0.3,
                 energy_budget_per_round: Dict[str, float] = None,
                 use_fisher_density: bool = True,
                 energy_weight: float = 1.0,
                 delay_weight: float = 1.0,
                 tail_weight: float = 0.5,
                 energy_budget_scale: float = 0.90,
                 time_budget_scale: float = 1.00,
                 tail_budget_scale: float = 1.05,
                 cvar_alpha: float = 0.80,
                 robust_norm: bool = True,
                 resource_clip: float = 4.0,
                 resource_floor: float = 0.20,
                 adaptive_V: bool = True,
                 V_min: float = 1.0,
                 V_max: float = 6.0,
                 adaptive_energy_coeff: float = 0.5,
                 adaptive_time_coeff: float = 0.5,
                 adaptive_tail_coeff: float = 0.3,
                 **kwargs):
        super().__init__(num_clients, args, **kwargs)
        self.V = float(V)
        self.mu = float(momentum_mu)
        self.ucb_beta = float(ucb_beta)
        self.use_fisher_density = bool(use_fisher_density)

        self.energy_weight = float(energy_weight)
        self.delay_weight = float(delay_weight)
        self.tail_weight = float(tail_weight)
        self.energy_budget_scale = float(energy_budget_scale)
        self.time_budget_scale = float(time_budget_scale)
        self.tail_budget_scale = float(tail_budget_scale)
        self.cvar_alpha = min(max(float(cvar_alpha), 0.5), 0.99)
        self.robust_norm = bool(robust_norm)
        self.resource_clip = max(float(resource_clip), 1.0)
        self.resource_floor = max(float(resource_floor), 0.0)

        self.adaptive_V = bool(adaptive_V)
        self.V_min = float(V_min)
        self.V_max = max(float(V_max), self.V_min)
        self.adaptive_energy_coeff = float(adaptive_energy_coeff)
        self.adaptive_time_coeff = float(adaptive_time_coeff)
        self.adaptive_tail_coeff = float(adaptive_tail_coeff)
        self._current_V = min(max(self.V, self.V_min), self.V_max)

        # Per-client energy debt queue.  Kept as self.queue for compatibility
        # with existing log parsing and plots.
        self.queue: Dict[int, float] = {i: 0.0 for i in range(num_clients)}
        self.participation_count: Dict[int, int] = {i: 0 for i in range(num_clients)}

        # Modality-level vector queues for total normalized energy and max time.
        self.energy_queue_by_modality: Dict[str, float] = {}
        self.time_queue_by_modality: Dict[str, float] = {}

        # Budgets are normalized quantities.
        self.energy_budget: Dict[str, float] = energy_budget_per_round or {}
        self.client_energy_budget: Dict[str, float] = {}
        self.time_budget: Dict[str, float] = {}
        self.tail_budget: float = 1.0

        # Normalizer scales and telemetry state.
        self._energy_scale: Dict[str, float] = {}
        self._time_scale: Dict[str, float] = {}
        self._budget_initialized = bool(self.energy_budget)
        self.tail_queue: float = 0.0
        self._tail_time_history: List[float] = []
        self._tail_history_window = 50

    # ------------------------------------------------------------------
    @staticmethod
    def _median_mad_scale(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 1.0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = median + 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = float(np.mean(values)) if values.size else 1.0
        return max(scale, 1e-9)

    def _scale_for(self, values: np.ndarray) -> float:
        if self.robust_norm:
            return self._median_mad_scale(values)
        return max(float(np.mean(values)), 1e-9)

    def _clip_norm(self, values: np.ndarray, scale: float) -> np.ndarray:
        lo = 1.0 / self.resource_clip
        hi = self.resource_clip
        return np.clip(values / max(scale, 1e-9), lo, hi)

    def _norm_arrays(self, ctx: SelectionContext, modality: str,
                     cids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        energy_raw = np.array([ctx.estimated_energy.get(c, 0.0) for c in cids], dtype=float)
        time_raw = np.array([ctx.estimated_time.get(c, 0.0) for c in cids], dtype=float)
        e_scale = self._energy_scale.get(modality, self._scale_for(energy_raw))
        t_scale = self._time_scale.get(modality, self._scale_for(time_raw))
        return self._clip_norm(energy_raw, e_scale), self._clip_norm(time_raw, t_scale)

    # ------------------------------------------------------------------
    def _init_budget_if_needed(self, ctx: SelectionContext):
        """Initialize robust resource budgets from the current candidate pool."""
        if self._budget_initialized and self.time_budget:
            return

        for m, cids in ctx.candidate_ids_by_modality.items():
            if not cids:
                continue
            energies = np.array([ctx.estimated_energy.get(c, 1.0) for c in cids], dtype=float)
            times = np.array([ctx.estimated_time.get(c, 1.0) for c in cids], dtype=float)
            self._energy_scale[m] = self._scale_for(energies)
            self._time_scale[m] = self._scale_for(times)

            e_norm = self._clip_norm(energies, self._energy_scale[m])
            t_norm = self._clip_norm(times, self._time_scale[m])
            n_pick = max(ctx.num_sample_by_modality.get(m, 1), 1)

            if m not in self.energy_budget:
                self.energy_budget[m] = (
                    self.energy_budget_scale * float(np.median(e_norm)) * n_pick
                )
            self.client_energy_budget[m] = self.energy_budget[m] / max(len(cids), 1)

            # For a max-latency resource, the expected random max of k samples is
            # around the k/(k+1) quantile.  This avoids an unrealistically tight
            # median deadline when selecting multiple clients in a modality.
            max_quantile = 100.0 * n_pick / (n_pick + 1.0)
            max_quantile = min(max(max_quantile, 50.0), 90.0)
            self.time_budget[m] = (
                self.time_budget_scale * float(np.percentile(t_norm, max_quantile))
            )

            self.energy_queue_by_modality.setdefault(m, 0.0)
            self.time_queue_by_modality.setdefault(m, 0.0)

        if self.time_budget:
            self.tail_budget = self.tail_budget_scale * max(self.time_budget.values())
        self._budget_initialized = True
        logger.info(
            "[Lyapunov] init RMAL "
            f"E_budget={self.energy_budget} T_budget={self.time_budget} "
            f"tail_budget={self.tail_budget:.3f} "
            f"E_scale={self._energy_scale} T_scale={self._time_scale}"
        )

    # ------------------------------------------------------------------
    def _update_adaptive_V(self):
        if not self.adaptive_V:
            self._current_V = self.V
            return
        e_pressure = float(np.mean(list(self.energy_queue_by_modality.values()))) \
            if self.energy_queue_by_modality else 0.0
        t_pressure = float(np.mean(list(self.time_queue_by_modality.values()))) \
            if self.time_queue_by_modality else 0.0
        pressure = (
            self.adaptive_energy_coeff * e_pressure
            + self.adaptive_time_coeff * t_pressure
            + self.adaptive_tail_coeff * self.tail_queue
        )
        v = self.V / (1.0 + max(pressure, 0.0))
        self._current_V = min(max(v, self.V_min), self.V_max)

    def on_round_start(self, ctx: SelectionContext):
        self._init_budget_if_needed(ctx)
        self._update_adaptive_V()

    def _current_tail_cvar(self, round_time_norm: float) -> float:
        hist = list(self._tail_time_history) + [float(round_time_norm)]
        if not hist:
            return float(round_time_norm)
        var_alpha = float(np.percentile(hist, self.cvar_alpha * 100.0))
        tail = [x for x in hist if x >= var_alpha]
        return float(np.mean(tail)) if tail else var_alpha

    def on_round_end(self, ctx: SelectionContext, selected_ids: List[int]):
        """Update per-client, per-modality, and CVaR tail-latency queues."""
        selected_set = set(selected_ids)
        selected_norm_times: List[float] = []

        for m, cids in ctx.candidate_ids_by_modality.items():
            if not cids:
                continue
            e_norm, t_norm = self._norm_arrays(ctx, m, cids)
            idx_by_cid = {cid: i for i, cid in enumerate(cids)}
            selected_idx = [idx_by_cid[cid] for cid in cids if cid in selected_set]

            e_selected = float(np.sum(e_norm[selected_idx])) if selected_idx else 0.0
            t_selected = float(np.max(t_norm[selected_idx])) if selected_idx else 0.0
            selected_norm_times.extend([float(t_norm[i]) for i in selected_idx])

            e_budget = self.energy_budget.get(m, 0.0)
            t_budget = self.time_budget.get(m, 0.0)
            old_eq = self.energy_queue_by_modality.get(m, 0.0)
            old_tq = self.time_queue_by_modality.get(m, 0.0)
            self.energy_queue_by_modality[m] = max((1.0 - self.mu) * old_eq
                                                   + e_selected - e_budget, 0.0)
            self.time_queue_by_modality[m] = max((1.0 - self.mu) * old_tq
                                                 + t_selected - t_budget, 0.0)

            client_budget = self.client_energy_budget.get(m, 0.0)
            for cid, e_k in zip(cids, e_norm):
                e_debt = float(e_k) if cid in selected_set else 0.0
                q_old = self.queue.get(cid, 0.0)
                self.queue[cid] = max((1.0 - self.mu) * q_old
                                      + e_debt - client_budget, 0.0)

        round_time_norm = max(selected_norm_times) if selected_norm_times else 0.0
        tail_cvar = self._current_tail_cvar(round_time_norm)
        self.tail_queue = max((1.0 - self.mu) * self.tail_queue
                              + tail_cvar - self.tail_budget, 0.0)
        self._tail_time_history.append(round_time_norm)
        if len(self._tail_time_history) > self._tail_history_window:
            self._tail_time_history = self._tail_time_history[-self._tail_history_window:]

        for cid in selected_ids:
            self.participation_count[cid] = self.participation_count.get(cid, 0) + 1

        avg_q = float(np.mean(list(self.queue.values())))
        max_q = float(np.max(list(self.queue.values())))
        avg_eq = float(np.mean(list(self.energy_queue_by_modality.values()))) \
            if self.energy_queue_by_modality else 0.0
        avg_tq = float(np.mean(list(self.time_queue_by_modality.values()))) \
            if self.time_queue_by_modality else 0.0
        total_e = sum(ctx.estimated_energy.get(c, 0.0) for c in selected_ids)
        total_t = max((ctx.estimated_time.get(c, 0.0) for c in selected_ids), default=0.0)
        rec = dict(round=ctx.round, avg_queue=avg_q, max_queue=max_q,
                   avg_energy_queue=avg_eq, avg_time_queue=avg_tq,
                   tail_queue=self.tail_queue, V_t=self._current_V,
                   total_energy=total_e, max_time=float(total_t),
                   norm_tail_cvar=tail_cvar)
        self.round_metrics.append(rec)
        logger.info(
            f"[Lyapunov] r={ctx.round} V_t={self._current_V:.3f} "
            f"avg_Q={avg_q:.3f} max_Q={max_q:.3f} "
            f"E_Q={avg_eq:.3f} T_Q={avg_tq:.3f} Z={self.tail_queue:.3f} "
            f"CVaR={tail_cvar:.3f} E_round={total_e:.3f} T_round={total_t:.3f}"
        )

    # ------------------------------------------------------------------
    def _fisher_density_for(self, cid: int, ctx: SelectionContext) -> float:
        if not self.use_fisher_density:
            return 1.0
        return self._fisher_density(ctx.fisher_info_by_client.get(cid))

    def _quality_array(self, ctx: SelectionContext, cids: List[int], t: int) -> np.ndarray:
        inf_raw = np.array([ctx.influence_scores.get(c, 0.0) for c in cids], dtype=float)
        i_min, i_max = inf_raw.min(), inf_raw.max()
        inf_norm = (inf_raw - i_min) / (i_max - i_min + 1e-9)
        ucb = np.array([
            self.ucb_beta * math.sqrt(math.log(t + 1) /
                                      max(self.participation_count.get(c, 0), 1))
            for c in cids
        ], dtype=float)
        psi = np.array([self._fisher_density_for(c, ctx) for c in cids], dtype=float)
        psi = psi / (float(np.mean(psi)) + 1e-9)
        return (inf_norm + ucb) * psi

    def select(self, ctx: SelectionContext) -> List[int]:
        t = max(ctx.round, 1)
        state = {}
        quota = {}
        remaining = {}
        selected_by_modality = {}

        for modality, cids in ctx.candidate_ids_by_modality.items():
            if not cids:
                continue
            num_sample = ctx.num_sample_by_modality.get(modality, 1)
            num_sample = max(min(num_sample, len(cids)), 1)

            energy, duration = self._norm_arrays(ctx, modality, cids)
            state[modality] = dict(
                cids=cids,
                quality=self._quality_array(ctx, cids, t),
                energy=energy,
                duration=duration,
                client_q=np.array([self.queue.get(c, 0.0) for c in cids], dtype=float),
            )
            quota[modality] = num_sample
            remaining[modality] = set(range(len(cids)))
            selected_by_modality[modality] = []

        current_round_time = 0.0
        while any(quota.get(m, 0) > 0 and remaining.get(m) for m in state):
            best_modality = None
            best_i = None
            best_score = -float("inf")

            for modality, values in state.items():
                if quota.get(modality, 0) <= 0 or not remaining.get(modality):
                    continue
                e_shadow = self.energy_queue_by_modality.get(modality, 0.0) + self.resource_floor
                t_shadow = self.time_queue_by_modality.get(modality, 0.0) + self.resource_floor
                z_shadow = self.tail_queue + self.resource_floor

                for i in remaining[modality]:
                    duration_i = float(values["duration"][i])
                    marginal_time = max(current_round_time, duration_i) - current_round_time
                    tail_excess = max(duration_i - self.tail_budget, 0.0)
                    energy_drift = (e_shadow + float(values["client_q"][i])) \
                        * float(values["energy"][i])
                    time_drift = t_shadow * marginal_time
                    tail_drift = z_shadow * tail_excess
                    score = (
                        self._current_V * float(values["quality"][i])
                        - self.energy_weight * energy_drift
                        - self.delay_weight * time_drift
                        - self.tail_weight * tail_drift
                    )
                    if score > best_score:
                        best_score = score
                        best_modality = modality
                        best_i = i

            if best_modality is None or best_i is None:
                break

            selected_by_modality[best_modality].append(best_i)
            remaining[best_modality].remove(best_i)
            quota[best_modality] -= 1
            current_round_time = max(current_round_time,
                                     float(state[best_modality]["duration"][best_i]))

        all_selected: List[int] = []
        for modality, values in state.items():
            selected_idx = selected_by_modality.get(modality, [])
            selected = [values["cids"][i] for i in selected_idx]
            all_selected.extend(selected)
            if t == 1 or t % 5 == 0:
                selected_energy = float(np.sum(values["energy"][selected_idx])) \
                    if selected_idx else 0.0
                selected_time = float(np.max(values["duration"][selected_idx])) \
                    if selected_idx else 0.0
                logger.info(
                    f"[Lyapunov-RMAL][{modality}] r={t} "
                    f"VQ.mean={(self._current_V * values['quality']).mean():.3f} "
                    f"E_Q={self.energy_queue_by_modality.get(modality, 0.0):.3f} "
                    f"T_Q={self.time_queue_by_modality.get(modality, 0.0):.3f} "
                    f"E_norm={selected_energy:.3f} T_norm={selected_time:.3f} "
                    f"selected={selected}"
                )

        return sorted(all_selected)
