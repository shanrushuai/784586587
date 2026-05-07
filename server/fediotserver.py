import logging
import torch

from .fedavgserver import FedavgServer, get_name_type, get_name_modality

logger = logging.getLogger(__name__)


class FediotServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FediotServer, self).__init__(**kwargs)
        if self.args.shared_param != 'blocks' or self.args.share_scope != 'modality_exact':
            logger.warning(
                "[FedIoT] Recommended settings are "
                "--shared_param blocks --share_scope modality_exact. "
                f"Current: shared_param={self.args.shared_param}, share_scope={self.args.share_scope}"
            )
        logger.info(f"[FedIoT] mm_scale={self.args.mm_scale}")

    def _get_client_coefficient_numerator(self, param_name, identifier, numerator):
        if numerator == 0:
            return 0
        if get_name_type(param_name) != 'blocks':
            return numerator
        if self.clients[identifier].modality != 'img+txt':
            return numerator
        # FedIoT amplifies multimodal client contributions when aggregating transformer blocks.
        return numerator * self.args.mm_scale

    def _get_base_numerator(self, param_name, identifier, numerator, fedavg=False):
        """Return the unscaled aggregation numerator for a parameter/client pair."""
        if numerator == 0:
            return 0

        scope = self.param_scope[param_name]
        client = self.clients[identifier]

        if scope == 'all':
            return numerator
        if scope == 'dataset':
            return numerator if client.dataset == self.dataset else 0
        if scope == 'task':
            return numerator if client.task == self.task else 0
        if scope == 'modality':
            if fedavg:
                return numerator if client.modality == self.modality else 0
            return numerator if (client.modality in self.modality or self.modality in client.modality) else 0
        if scope == 'modality_exact':
            param_modality = get_name_modality(param_name, self.args.modalities)
            if param_modality is None:
                return numerator if (client.modality in self.modality or self.modality in client.modality) else 0
            return numerator if (client.modality == param_modality or param_modality in client.modality) else 0

        return 0

    def _aggregate(self, ids, updated_sizes, fedavg=False):
        """Aggregate with a strict weighted average and FedIoT multimodal scaling.

        The FedAvg base implementation updates a running tensor in sequence:
        ``w += coef * (w_i - w)``. That is order-dependent and is not equivalent to
        FedIoT/FedAvg weighted averaging when more than one client contributes.
        """
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(
            f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] '
            f'[Round: {str(self.round).zfill(4)}] Aggregate updated signals!'
        )

        base_sd = self.global_model.cpu().required_params()
        uploaded = {identifier: dict(self.clients[identifier].upload()) for identifier in ids}
        final_sd = {}
        scaled_contribs = 0

        for param_name, global_param in base_sd.items():
            numerators = {}
            for identifier in ids:
                if param_name not in uploaded[identifier]:
                    numerators[identifier] = 0
                    continue

                numerator = self._get_base_numerator(
                    param_name=param_name,
                    identifier=identifier,
                    numerator=updated_sizes[identifier],
                    fedavg=fedavg,
                )
                if not fedavg:
                    scaled = self._get_client_coefficient_numerator(param_name, identifier, numerator)
                    if scaled != numerator:
                        scaled_contribs += 1
                    numerator = scaled

                if (
                    not fedavg
                    and self.clients[identifier].modality != self.modality
                    and self.out_modality_scale != 1
                ):
                    numerator *= self.out_modality_scale

                numerators[identifier] = numerator

            denominator = sum(numerators.values())
            if denominator == 0:
                final_sd[param_name] = global_param.detach().clone()
                continue

            if not torch.is_floating_point(global_param):
                source_id = next((identifier for identifier, numerator in numerators.items() if numerator != 0), None)
                final_sd[param_name] = (
                    uploaded[source_id][param_name].detach().clone()
                    if source_id is not None else global_param.detach().clone()
                )
                continue

            averaged = torch.zeros_like(global_param)
            for identifier, numerator in numerators.items():
                if numerator == 0:
                    continue
                coefficient = float(numerator / denominator)
                averaged += uploaded[identifier][param_name].to(dtype=global_param.dtype) * coefficient
            final_sd[param_name] = averaged

        self.global_model.load_state_dict(final_sd, strict=False)
        if not fedavg:
            logger.info(
                f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] '
                f'[Round: {str(self.round).zfill(4)}] FedIoT strict weighted average '
                f'completed | mm_scale={self.args.mm_scale} | scaled_contribs={scaled_contribs}'
            )

        logger.info(
            f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] '
            f'[Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!'
        )
