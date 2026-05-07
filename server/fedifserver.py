# src/server/fedifserver.py
import os
import gc
import json
import torch
import random
import logging
import numpy as np
import copy
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn as nn

# 导入父类及相关常量
from src.server.fedavgserver import FedavgServer, DATASET_2_MODALITY, DATASET_2_TASK, TASK_2_CRITERION
import src.criterions
from src.server.fedavgserver import get_name_modality
from src.server.selectors import build_selector, ResourceProfiler
from src.server.selectors.base import SelectionContext
logger = logging.getLogger(__name__)


class FedifServer(FedavgServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model_str):
        """
        FedIF-Cola Server (Enhanced with Recency/Staleness & Stratified Normalization)

        Mechanism:
        1. Influence Score: Measures contribution to global multi-modal task (Quality).
        2. Recency Score: Compensates for seldom-selected clients (Exploration).
        3. Stratified Norm: Normalizes scores within each modality group to prevent MM clients from dominating.
        """
        # ================= [Init Parent Logic] =================
        # 1. 拆解 server_dataset 以匹配父类接口
        # data.py 返回的是 ((raw_tests, server_datasets), client_datasetss)
        # main.py 解包后传进来的是 (raw_tests, server_datasets)
        validation_source = None
        testing_source = None

        if isinstance(server_dataset, (tuple, list)) and len(server_dataset) == 2:
            validation_source = server_dataset[0]  # raw_tests 字典 (包含 'fedif_golden')
            testing_source = server_dataset[1]  # server_datasets (用于 server 端评估)
        elif isinstance(server_dataset, dict):
            validation_source = server_dataset
            testing_source = None
        else:
            logger.warning(f"[FedIF] server_dataset type {type(server_dataset)} unexpected. Treating as empty.")
            validation_source = {}
            testing_source = None

        # 2. 准备给父类 FedAvg 的数据 (必须移除 'fedif_golden' 防止父类评估报错)
        fedavg_dataset_dict = validation_source.copy() if validation_source else {}
        if 'fedif_golden' in fedavg_dataset_dict:
            del fedavg_dataset_dict['fedif_golden']

        # 3. 重新包装成元组 (None, dict) 传给父类
        # FedAvgServer._set_loaders 会取 datasets[1] 作为评估集
        fedavg_pack = (None, fedavg_dataset_dict)

        # 4. [必须最先执行] 调用父类初始化
        super(FedifServer, self).__init__(args, writer, fedavg_pack, client_datasets, model_str)

        # ================= [FedIF Specific Init] =================
        # 5. 保留完整的 validation source (包含黄金集)
        self.validation_source = validation_source
        self.fedif_gamma = getattr(args, 'fedif_gamma', 0.9)  # 历史分数衰减系数 (EMA)
        self.recency_alpha = 0.5  # 陈旧度权重

        # 6. 状态记录器
        self.client_influence = {i: 0.0 for i in range(args.K)}
        self.client_last_selected = {i: -1 for i in range(args.K)}

        # ===== [Resource-aware Selector Module] =====
        # Switchable client selector with energy/bandwidth/latency awareness.
        # args.resource_selector in {"none","oort","lyapunov"}.
        self.resource_selector_name = getattr(args, "resource_selector", "none")
        self.resource_selector = None
        self.resource_profiler = None
        self._round_resource_log = []
        # Always init profiler for resource telemetry (even when selector=none)
        client_modalities = {c.id: c.modality for c in self.clients}
        self.resource_profiler = ResourceProfiler(
            num_clients=args.K,
            client_modalities=client_modalities,
            seed=getattr(args, "seed", 42),
            capacity_path=getattr(args, "fedscale_capacity_path", None),
            alpha_e_flop=getattr(args, "energy_per_flop", 1.0e-10),
            beta_e_byte=getattr(args, "energy_per_byte", 1.0e-7),
        )
        logger.info(self.resource_profiler.summary())
        if self.resource_selector_name and self.resource_selector_name != "none":

            sel_kwargs = dict(num_clients=args.K, args=args)
            if self.resource_selector_name == "lyapunov":
                sel_kwargs.update(
                    V=getattr(args, "lyap_V", 3.0),
                    momentum_mu=getattr(args, "lyap_mu", 0.1),
                    ucb_beta=getattr(args, "lyap_beta", 0.3),
                    use_fisher_density=getattr(args, "lyap_fisher_density", True),
                    energy_weight=getattr(args, "lyap_energy_weight", 1.0),
                    delay_weight=getattr(args, "lyap_delay_weight", 1.0),
                    tail_weight=getattr(args, "lyap_tail_weight", 0.5),
                    energy_budget_scale=getattr(args, "lyap_energy_budget_scale", 0.90),
                    time_budget_scale=getattr(args, "lyap_time_budget_scale", 1.00),
                    tail_budget_scale=getattr(args, "lyap_tail_budget_scale", 1.05),
                    cvar_alpha=getattr(args, "lyap_cvar_alpha", 0.80),
                    robust_norm=not getattr(args, "lyap_mean_norm", False),
                    resource_clip=getattr(args, "lyap_resource_clip", 4.0),
                    resource_floor=getattr(args, "lyap_resource_floor", 0.20),
                    adaptive_V=not getattr(args, "lyap_static_V", False),
                    V_min=getattr(args, "lyap_V_min", 1.0),
                    V_max=getattr(args, "lyap_V_max", 6.0),
                    adaptive_energy_coeff=getattr(args, "lyap_adapt_energy_coeff", 0.5),
                    adaptive_time_coeff=getattr(args, "lyap_adapt_time_coeff", 0.5),
                    adaptive_tail_coeff=getattr(args, "lyap_adapt_tail_coeff", 0.3),
                )
            elif self.resource_selector_name == "oort":
                sel_kwargs.update(
                    round_penalty=getattr(args, "oort_penalty", 2.0),
                    duration_percentile=getattr(args, "oort_pref_pct", 80.0),
                    ucb_coeff=getattr(args, "oort_ucb", 0.1),
                )
            elif self.resource_selector_name == "fedbalancer":
                sel_kwargs.update(
                    alpha=getattr(args, "fb_alpha", 2.0),
                    epsilon_init=getattr(args, "fb_epsilon", 0.9),
                    ddl_stepsize=getattr(args, "fb_ddl_step", 0.05),
                    window=getattr(args, "fb_window", 5),
                )
            self.resource_selector = build_selector(self.resource_selector_name, **sel_kwargs)
            logger.info(f"[FedIF] ResourceSelector enabled: {self.resource_selector_name}")

        # 7. 准备多模态验证数据 (用于计算 g_val)
        self.mm_val_loaders = []  # 存储所有版本的 Loader
        self.mm_dataset_name = None
        self.mm_val_loader = None  # 当前轮次使用的 loader
        self.anchor_val_loaders = {}  # e.g. {"CIFAR100": loader, "AG_NEWS": loader}

        for k, ds in self.validation_source.items():
            # 跳过 FedIF 检索 golden
            if isinstance(k, str) and k.startswith("fedif_golden"):
                continue
            # 只给真正的“任务数据集”建 loader（CIFAR100 / AG_NEWS）
            if k in DATASET_2_MODALITY and DATASET_2_MODALITY[k] in ("img", "txt"):
                self.anchor_val_loaders[k] = DataLoader(
                    ds,
                    batch_size=getattr(self.args, "fedif_anchor_bs", self.args.B),
                    shuffle=True,
                    drop_last=True,
                    num_workers=getattr(self.args, "num_workers", 2),
                    pin_memory=True
                )

        print(f"[FedIF] anchor_val_loaders = {list(self.anchor_val_loaders.keys())}")

        if isinstance(self.validation_source, dict):
            # A. 优先查找包含 'fedif_golden' 的 key (如 fedif_golden_v0, fedif_golden_v1)
            # 排序是为了保证不同实验复现时加载顺序一致
            golden_keys = sorted([k for k in self.validation_source.keys() if 'fedif_golden' in k])

            if golden_keys:
                logger.info(f"[FedIF] Found {len(golden_keys)} Golden Sets: {golden_keys}. Enabling Random Rotation.")

                # 确定模型名称 (通常取第一个 img+txt 模型，如 Coco 或 Flickr30k)
                # 这一步是为了确定用哪个模型来算梯度
                self.mm_dataset_name = next(
                    (k for k in self.global_models.keys() if DATASET_2_MODALITY.get(k) == 'img+txt'), None)

                if self.mm_dataset_name:
                    # 为每个版本创建一个 DataLoader
                    for key in golden_keys:
                        d_set = self.validation_source[key]
                        loader = DataLoader(
                            d_set,
                            batch_size=args.B,
                            shuffle=True,  # 必须 Shuffle
                            num_workers=args.num_thread,
                            drop_last=True
                        )
                        self.mm_val_loaders.append(loader)

            # B. 回退逻辑：如果没有找到 golden set，尝试找任意 img+txt 数据集
            if not self.mm_val_loaders:
                for d_name in self.validation_source.keys():
                    if DATASET_2_MODALITY.get(d_name) == 'img+txt':
                        self.mm_dataset_name = d_name
                        logger.info(f"[FedIF] Golden Set NOT found. Fallback to dataset: {d_name}")
                        d_set = self.validation_source[d_name]
                        loader = DataLoader(
                            d_set,
                            batch_size=args.B,
                            shuffle=True,
                            num_workers=args.num_thread,
                            drop_last=True
                        )
                        self.mm_val_loaders.append(loader)
                        break

        # 初始化：默认使用列表中的第一个 (如果有的话)
        if self.mm_val_loaders:
            self.mm_val_loader = self.mm_val_loaders[0]
        else:
            logger.warning("[FedIF] ⚠️ No suitable validation set found. FedIF calculation will be skipped.")

    def _get_param_map(self, client_modality):
        """
        [关键] 参数名映射规则
        FedCola 架构:
          - Server (MM): blockses.0 (Image), blockses.1 (Text)
          - Client (Img): blocks (需要映射到 blockses.0)
          - Client (Txt): blocks (需要映射到 blockses.1)
        """
        if client_modality == 'img+txt':
            return lambda x: x

        def mapper(client_key):
            server_key = client_key
            # Client Image -> Server Image Stream (.0)
            if client_modality == 'img':
                if 'blocks' in client_key and 'blockses' not in client_key:
                    server_key = client_key.replace('blocks', 'blockses.0')
                elif 'blockses.0' in client_key:
                    server_key = client_key
            # Client Text -> Server Text Stream (.1)
            elif client_modality == 'txt':
                if 'blocks' in client_key and 'blockses' not in client_key:
                    server_key = client_key.replace('blocks', 'blockses.1')
                elif 'blockses.0' in client_key:
                    # 理论上纯文本客户端不应包含 blockses.0，但在 FedCola 互补训练中可能会有辅助参数
                    # 这里保持原样或进行特定处理
                    pass
            return server_key

        return mapper

    def _calc_mm_validation_gradients(self):
        """
        计算服务器端多模态(mm)模型在验证集上的梯度(g_val)
        - 支持随机轮换多个 golden loader
        - 梯度累积平滑
        - 只提取 attn.qkv / attn.proj
        - 返回 CPU grads dict
        """
        # ===== [0] 随机轮换验证集版本 =====
        if getattr(self, "mm_val_loaders", None):
            if isinstance(self.mm_val_loaders, dict) and len(self.mm_val_loaders) > 1:
                self.mm_val_loader = random.choice(list(self.mm_val_loaders.values()))
                if hasattr(self, "_val_iterator"):
                    del self._val_iterator
            elif isinstance(self.mm_val_loaders, list) and len(self.mm_val_loaders) > 1:
                self.mm_val_loader = random.choice(self.mm_val_loaders)
                if hasattr(self, "_val_iterator"):
                    del self._val_iterator

        if getattr(self, "mm_val_loader", None) is None or getattr(self, "mm_dataset_name", None) is None:
            return {}

        mm_model = self.global_models[self.mm_dataset_name]
        mm_model.to(self.server_device)
        mm_model.eval()
        mm_model.zero_grad(set_to_none=True)

        # ===== [配置] 梯度累积 =====
        accumulation_steps = 8
        max_grad_norm = 1.0

        # Loss
        criterion_name = TASK_2_CRITERION.get('img+txt', 'ContrastiveLoss')
        if hasattr(src.criterions, criterion_name):
            criterion_cls = getattr(src.criterions, criterion_name)
        elif hasattr(nn, criterion_name):
            criterion_cls = getattr(nn, criterion_name)
        else:
            from src.criterions.contrastive_loss import ContrastiveLoss
            criterion_cls = ContrastiveLoss
        criterion = criterion_cls().to(self.server_device)

        # iterator
        if not hasattr(self, "_val_iterator"):
            self._val_iterator = iter(self.mm_val_loader)

        total_loss_val = 0.0
        used_batches = 0

        for step in range(accumulation_steps):
            try:
                batch = next(self._val_iterator)
            except StopIteration:
                # 重新建 iterator
                self._val_iterator = iter(DataLoader(
                    self.mm_val_loader.dataset,
                    batch_size=self.args.B,
                    shuffle=True,
                    num_workers=getattr(self.args, "num_thread", 2),
                    drop_last=True
                ))
                batch = next(self._val_iterator)

            # 兼容 tuple/dict
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.server_device)
                targets = batch[1].to(self.server_device)
            elif isinstance(batch, dict):
                inputs = batch["image"].to(self.server_device)
                targets = batch["text"].to(self.server_device)
            else:
                continue

            outputs = mm_model([inputs, targets], feat_out=True)
            loss = criterion(*outputs)
            loss = loss / float(accumulation_steps)
            loss.backward()

            total_loss_val += loss.item()
            used_batches += 1

        torch.nn.utils.clip_grad_norm_(mm_model.parameters(), max_grad_norm)

        val_grads = {}
        for name, param in mm_model.named_parameters():
            if param.grad is None:
                continue
            if ("attn.qkv" in name) or ("attn.proj" in name):
                val_grads[name] = param.grad.detach().cpu()  # ✅ CPU缓存

        mm_model.zero_grad(set_to_none=True)
        mm_model.to("cpu")
        torch.cuda.empty_cache()

        print(
            f"[FedIF] g_val[{self.mm_dataset_name}] (mm) batches={used_batches} loss={total_loss_val:.4f} #keys={len(val_grads)}")
        return val_grads

    def _calc_cls_validation_gradients_for_dataset(self, dataset_name: str):
        """
        为单任务分类数据集（CIFAR100 / AG_NEWS）计算 golden grads：
        - CIFAR100: img stream
        - AG_NEWS : txt stream
        输出：{param_name: grad_tensor(cpu)}
        """
        modality = DATASET_2_MODALITY[dataset_name]
        assert modality in ("img", "txt")

        model = self.global_models[dataset_name]
        model.eval()
        model.to(self.server_device)
        model.zero_grad(set_to_none=True)

        # criterion: CrossEntropyLoss
        task = DATASET_2_TASK.get(dataset_name, "cls")  # CIFAR100/AG_NEWS 都是 cls
        criterion_name = TASK_2_CRITERION.get(task, "CrossEntropyLoss")

        if hasattr(src.criterions, criterion_name):
            criterion_cls = getattr(src.criterions, criterion_name)
        elif hasattr(nn, criterion_name):
            criterion_cls = getattr(nn, criterion_name)
        else:
            criterion_cls = nn.CrossEntropyLoss

        criterion = criterion_cls().to(self.server_device)
        loader = self.anchor_val_loaders.get(dataset_name, None)
        if loader is None:
            print(f"[FedIF] No anchor loader for {dataset_name}, skip g_val.")
            return {}

        accum_steps = getattr(self.args, "fedif_anchor_accum", 4)
        max_batches = getattr(self.args, "fedif_anchor_max_batches", 8)

        batches = 0
        loss_sum = 0.0

        for (inputs, targets) in loader:
            inputs = inputs.to(self.server_device)
            targets = targets.to(self.server_device)

            if modality == "img":
                out = model([inputs, None])
                logits = out[0] if isinstance(out, (list, tuple)) else out
            else:  # txt
                out = model([None, inputs])
                logits = out[1] if isinstance(out, (list, tuple)) else out

            loss = criterion(logits, targets) / float(accum_steps)
            loss.backward()
            loss_sum += loss.item()

            batches += 1
            if batches % accum_steps == 0:
                # 防爆
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            if batches >= max_batches:
                break

        # 只收集 attention 梯度（qkv / proj）
        val_grads = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if ("attn.qkv" in name) or ("attn.proj" in name):
                val_grads[name] = p.grad.detach().cpu()

        model.to("cpu")
        torch.cuda.empty_cache()

        print(
            f"[FedIF] g_val[{dataset_name}] modality={modality} batches={batches} loss={loss_sum:.4f} #keys={len(val_grads)}")
        return val_grads

    def _calc_multi_anchor_validation_gradients(self):
        """
        返回 dict_of_dict:
        {
          "CIFAR100": {...},
          "AG_NEWS": {...},
          "Coco": {...}  # 如果你也要为检索任务存
        }
        """
        all_grads = {}

        # 1) 分类数据集：按 dataset 算
        for ds in list(self.anchor_val_loaders.keys()):
            all_grads[ds] = self._calc_cls_validation_gradients_for_dataset(ds)

        # 2) 检索数据集：仍用 mm golden
        # 这里你可以选择用 mm_dataset_name 作为 key
        if getattr(self, "mm_val_loaders", None):
            all_grads[self.mm_dataset_name] = self._calc_mm_validation_gradients()

        return all_grads

    @torch.no_grad()
    def _update_influence_scores(self, selected_ids, val_grads_pack):
        """
        influence(client) = dot( delta_w , g_val )
        ✅ multi-anchor: 每个客户端用自己 dataset 的 g_val
        """
        # 1) base_sds：每个 dataset 的 server 基准权重（CPU）
        base_sds = {}
        for dataset_name, model in self.global_models.items():
            base_sds[dataset_name] = {k: v.cpu() for k, v in model.state_dict().items()}

        stratified_raw_scores = defaultdict(dict)

        for client_id in selected_ids:
            client = self.clients[client_id]

            # 2) ✅ 为该 client 选择对应 dataset 的 g_val
            val_grads = val_grads_pack.get(getattr(client, "dataset", None), None)
            if not val_grads:
                # fallback：没有就用 mm 的（避免全 1.0）
                val_grads = val_grads_pack.get(getattr(self, "mm_dataset_name", ""), {})
            if not val_grads:
                continue
            print(f"[INFL] cid={client_id} ds={client.dataset} g_keys={len(val_grads)}")

            if client.dataset not in base_sds:
                continue

            key_mapper = self._get_param_map(client.modality)
            client_sd = dict(client.upload())
            base_sd = base_sds[client.dataset]

            client_vec_list = []
            val_grad_vec_list = []

            for client_k, client_w in client_sd.items():
                # 只做 attention
                if ('attn.qkv' not in client_k) and ('attn.proj' not in client_k):
                    continue

                server_k = key_mapper(client_k)

                # ✅ attn blockses.1 -> blockses.0 fallback（你写的这个逻辑是对的）
                if (server_k not in val_grads or server_k not in base_sd) and (
                        ("attn.qkv" in server_k) or ("attn.proj" in server_k)) and ("blockses.1." in server_k):
                    server_k2 = server_k.replace("blockses.1.", "blockses.0.", 1)
                    if (server_k2 in val_grads) and (server_k2 in base_sd):
                        server_k = server_k2

                if (server_k in val_grads) and (server_k in base_sd):
                    w_base = base_sd[server_k].to(self.server_device)
                    w_new = client_w.to(self.server_device)

                    delta_w = w_base - w_new

                    if delta_w.shape != val_grads[server_k].shape:
                        continue

                    client_vec_list.append(delta_w.flatten())

                    # ✅ 关键：val_grads 在 CPU，必须搬到 server_device
                    val_grad_vec_list.append(val_grads[server_k].to(self.server_device).flatten())

            if len(client_vec_list) == 0:
                continue

            client_vec = torch.cat(client_vec_list)
            val_grad_vec = torch.cat(val_grad_vec_list)

            denom = (torch.norm(client_vec) * torch.norm(val_grad_vec) + 1e-12)
            raw_score = (torch.dot(client_vec, val_grad_vec) / denom).item()
            stratified_raw_scores[client.modality][client_id] = raw_score

        # 2. 分模态归一化 (In-Modality Normalization)
        # 确保每个模态组的第一名都能得到 1.0 的分数
        for modality, id_score_map in stratified_raw_scores.items():
            if not id_score_map: continue

            scores = list(id_score_map.values())
            min_s, max_s = min(scores), max(scores)
            div = max_s - min_s + 1e-9

            # logger.info(f"[FedIF] {modality} Stats: Min={min_s:.4f}, Max={max_s:.4f}")

            for cid, r_score in id_score_map.items():
                # 组内 Min-Max 归一化
                eps = getattr(self.args, 'fedif_norm_eps', 0.05)  # 新增超参，默认 0.05
                range_s = max_s - min_s

                if range_s < 1e-6:
                    # 组内分数几乎一样：给一个“中性值”，避免全变 0
                    norm_score = 0.5
                else:
                    base = (r_score - min_s) / (range_s + 1e-9)  # 仍然是 min-max
                    norm_score = eps + (1 - eps) * base  # 映射到 [eps, 1]

                # EMA 更新历史分数
                old_score = self.client_influence.get(cid, 0.0)
                self.client_influence[cid] = (1 - self.fedif_gamma) * old_score + \
                                             self.fedif_gamma * norm_score

        # 打印 Top-10
        sorted_influence = sorted(self.client_influence.items(), key=lambda x: x[1], reverse=True)
        top_k_str = ", ".join([f"ID{k}:{v:.4f}" for k, v in sorted_influence[:10]])
        logger.info(f"[FedIF] Top-10 Influential Clients (Stratified): {top_k_str}")

    def _log_resource_telemetry(self, selected_ids):
        """Log estimated energy/time for ANY selection path (baseline included)."""
        if self.resource_profiler is None:
            return
        B = int(getattr(self.args, "B", 32))
        E = int(getattr(self.args, "E", 1))
        total_e, times = 0.0, []
        for cid in selected_ids:
            prof = self.resource_profiler.get(cid)
            samples = int(getattr(self.clients[cid], "num_samples",
                                  getattr(self.clients[cid], "n", 256)))
            mb = self._estimate_model_bytes(cid)
            total_e += prof.energy_cost(B, E, samples, mb)
            times.append(prof.completion_time(B, E, samples, mb))
        max_t = max(times) if times else 0.0
        self._round_resource_log.append(dict(
            round=self.round, selected=len(selected_ids),
            total_energy=total_e, max_round_time=max_t,
        ))
        logger.info(
            f"[FedIF][resource] r={self.round} "
            f"n={len(selected_ids)} E_round={total_e:.4f} T_round={max_t:.4f}"
        )

    def _sample_clients(self, exclude=[]):
        """
        基于 '分层影响力 + 陈旧度' 的混合采样
        若启用 resource_selector，则交给资源感知模块（Oort/Lyapunov）
        """
        # Warmup Phase: 随机采样
        if self.round <= self.args.warmup_rounds:
            sampled = super(FedifServer, self)._sample_clients(exclude)
            for cid in sampled:
                self.client_last_selected[cid] = self.round
            self._log_resource_telemetry(sampled)
            return sampled

        # ===== Resource-aware path =====
        if self.resource_selector is not None:
            sampled = self._sample_clients_resource_aware(exclude)
            for cid in sampled:
                self.client_last_selected[cid] = self.round
            for i, id in enumerate(sampled):
                self.clients[id].device = 'cuda:%d' % (
                    i % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'
            return sampled

        logger.info(f'[FedIF] [Round: {str(self.round).zfill(4)}] Using Stratified Influence + Recency Sampling!')
        sampled_client_ids = []

        for i, dataset in enumerate(self.args.datasets):
            candidate_ids = [client.id for client in self.clients if client.dataset == dataset]
            if not candidate_ids: continue

            # 过滤排除名单
            candidate_ids = [c for c in candidate_ids if c not in exclude]
            if not candidate_ids: continue

            num_sampled = max(int(self.Cs[dataset] * len(candidate_ids)), 1)

            # 1. 获取影响力分数 (Quality)
            raw_inf_scores = np.array([self.client_influence[uid] for uid in candidate_ids])

            # 当前批次再次局部归一化，确保相对差异生效
            eps = getattr(self.args, 'fedif_batch_norm_eps', 0.05)

            if raw_inf_scores.max() > raw_inf_scores.min():
                base = (raw_inf_scores - raw_inf_scores.min()) / (raw_inf_scores.max() - raw_inf_scores.min() + 1e-9)
                inf_scores = eps + (1 - eps) * base
            else:
                inf_scores = np.ones_like(raw_inf_scores) * 0.5

            # 2. 计算陈旧度 (Staleness)
            # staleness = 当前轮 - 上次选中轮
            staleness = np.array([self.round - self.client_last_selected[uid] for uid in candidate_ids])

            # Log 平滑防止数值爆炸
            norm_recency = np.log(1 + staleness)
            # 归一化 Recency 到 [0, 1]
            if norm_recency.max() > norm_recency.min():
                norm_recency = (norm_recency - norm_recency.min()) / (norm_recency.max() - norm_recency.min() + 1e-9)

            # 3. 融合分数
            final_scores = inf_scores + self.recency_alpha * norm_recency

            # 4. Softmax 概率分布
            # Temperature T=1.0. T 越小越贪婪，T 越大越随机
            temperature = 0.9
            exp_scores = np.exp((final_scores - np.max(final_scores)) / temperature)
            probs = exp_scores / np.sum(exp_scores)

            # 5. 概率采样
            try:
                selected = np.random.choice(candidate_ids, size=num_sampled, replace=False, p=probs)
            except ValueError:
                # 容错处理 (如 probs 含 NaN)
                selected = np.random.choice(candidate_ids, size=num_sampled, replace=False)

            sampled_client_ids.extend(selected)

            # 更新选中记录
            for cid in selected:
                self.client_last_selected[cid] = self.round

        sampled_client_ids = sorted(sampled_client_ids)

        # 设置设备
        for i, id in enumerate(sampled_client_ids):
            self.clients[id].device = 'cuda:%d' % (
                    i % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'

        self._log_resource_telemetry(sampled_client_ids)
        return sampled_client_ids

    # ======================================================================
    # Resource-aware client selection (NEW MODULE)
    # ======================================================================
    def _estimate_model_bytes(self, client_id: int = None) -> float:
        """Approximate per-client upload size in bytes.

        FedCola's client.upload() returns the *full* state_dict; the server
        decides aggregation scope per param via param_scope.
        We look up the correct global_model by the client's dataset so that
        img+txt clients get the larger dual-stream model size.
        """
        if not hasattr(self, "_upload_bytes_cache"):
            self._upload_bytes_cache = {}
        if client_id is not None:
            modality = self.clients[client_id].modality
        else:
            modality = "img"
        if modality in self._upload_bytes_cache:
            return self._upload_bytes_cache[modality]

        try:
            # Find a global_model matching this modality
            model = None
            if client_id is not None:
                ds = getattr(self.clients[client_id], "dataset", None)
                if ds and ds in self.global_models:
                    model = self.global_models[ds]
            if model is None:
                for ds, m in self.global_models.items():
                    if DATASET_2_MODALITY.get(ds) == modality:
                        model = m
                        break
            if model is None:
                model = next(iter(self.global_models.values()))
            total = sum(p.numel() for p in model.parameters())
            bytes_ = max(total * 4.0, 1.0e6)
        except Exception:
            bytes_ = 1.0e7
        self._upload_bytes_cache[modality] = bytes_
        return bytes_

    def _build_selection_context(self, candidates_by_modality, num_sample_by_modality):
        """Collect profile-derived time/energy and influence/fisher into SelectionContext."""
        B = int(getattr(self.args, "B", 32))
        E = int(getattr(self.args, "E", 1))

        est_time, est_energy = {}, {}
        data_sizes = {}
        fisher_by_cid = {}

        for cids in candidates_by_modality.values():
            for cid in cids:
                prof = self.resource_profiler.get(cid)
                client = self.clients[cid]
                samples = int(getattr(client, "num_samples",
                                      getattr(client, "n", 256)))
                model_bytes = self._estimate_model_bytes(cid)
                data_sizes[cid] = samples
                est_time[cid] = prof.completion_time(B, E, samples, model_bytes)
                est_energy[cid] = prof.energy_cost(B, E, samples, model_bytes)
                fm = getattr(client, "fisher_matrix", None)
                if fm:
                    fisher_by_cid[cid] = fm

        return SelectionContext(
            round=self.round,
            candidate_ids_by_modality=candidates_by_modality,
            num_sample_by_modality=num_sample_by_modality,
            influence_scores=dict(self.client_influence),
            last_selected_round=dict(self.client_last_selected),
            fisher_info_by_client=fisher_by_cid,
            data_sizes=data_sizes,
            profiles=self.resource_profiler.profiles,
            estimated_time=est_time,
            estimated_energy=est_energy,
        )

    def _sample_clients_resource_aware(self, exclude=[]):
        """Delegate selection to self.resource_selector (Oort / Lyapunov)."""
        # group candidates by modality (one bucket per dataset's modality).
        cand_by_mod = defaultdict(list)
        num_by_mod = {}
        for dataset in self.args.datasets:
            modality = DATASET_2_MODALITY.get(dataset, "img")
            cids = [c.id for c in self.clients if c.dataset == dataset and c.id not in exclude]
            if not cids:
                continue
            cand_by_mod[modality].extend(cids)
            num_by_mod[modality] = num_by_mod.get(modality, 0) + max(
                int(self.Cs[dataset] * len(cids)), 1)

        ctx = self._build_selection_context(dict(cand_by_mod), num_by_mod)
        self.resource_selector.on_round_start(ctx)
        selected = self.resource_selector.select(ctx)
        self.resource_selector.on_round_end(ctx, selected)

        # per-round telemetry
        total_E = sum(ctx.estimated_energy.get(c, 0.0) for c in selected)
        max_T = max((ctx.estimated_time.get(c, 0.0) for c in selected), default=0.0)
        self._round_resource_log.append(dict(
            round=self.round, selected=len(selected),
            total_energy=total_E, max_round_time=max_T,
        ))
        logger.info(
            f"[FedIF][{self.resource_selector_name}] r={self.round} "
            f"n={len(selected)} E_round={total_E:.4f} T_round={max_T:.4f}"
        )
        return sorted(selected)

    def _is_attn_qkv_proj(self, name: str) -> bool:
        return ('attn.qkv' in name) or ('attn.proj' in name)

    def _is_shared_attn(self, name: str) -> bool:
        # 只对“共享范围不是 dataset”的 attention 参数启用
        if not self._is_attn_qkv_proj(name):
            return False
        return self.param_scope.get(name, 'dataset') != 'dataset'

    def _is_cls_client(self, cid: int) -> bool:
        return getattr(self.clients[cid], "task", None) == "cls"

    def _is_ret_client(self, cid: int) -> bool:
        m = getattr(self.clients[cid], "modality", None)
        t = getattr(self.clients[cid], "task", None)
        return (m == "img+txt") or (t in ("img+txt", "retrieval", "rtv"))

    def _map_param_to_mm_key(self, param_name: str, check_grads: dict = None) -> str:
        """
        把参数名映射到 mm_model 的 key，用于从某个梯度字典里取 golden grad。
        ✅ 支持两种梯度字典：
        - cls anchor grads: key 是 blocks.*
        - mm  anchor grads: key 是 blockses.0/1.*

        规则：
        1) 若 check_grads 里直接有 param_name，优先返回（cls anchor）
        2) 否则做 blocks -> blockses.{0/1} 映射
        3) attention fallback: blockses.1 -> blockses.0
        """
        if check_grads is not None and param_name in check_grads:
            return param_name  # ✅ 分类 anchor 直接命中

        # 1) 理论映射
        if self.modality == "img":
            mm_key = param_name.replace("blocks.", "blockses.0.", 1)
        elif self.modality == "txt":
            mm_key = param_name.replace("blocks.", "blockses.1.", 1)
        else:
            mm_key = param_name

        if check_grads is None:
            return mm_key

        if mm_key in check_grads:
            return mm_key

        # 2) attention fallback
        if (("attn.qkv" in mm_key) or ("attn.proj" in mm_key)) and ("blockses.1." in mm_key):
            mm_key2 = mm_key.replace("blockses.1.", "blockses.0.", 1)
            if mm_key2 in check_grads:
                return mm_key2

        return mm_key

    @torch.no_grad()
    def _gold_guided_project(self, delta: torch.Tensor, gold_grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if gold_grad is None:
            return delta
        if delta.shape != gold_grad.shape:
            return delta

        # ✅ 关键：把 CPU 上缓存的 gold_grad 搬到 delta 所在 device
        gold_grad = gold_grad.to(device=delta.device, dtype=delta.dtype)

        d = delta.view(-1)
        g = gold_grad.view(-1)

        dot = torch.dot(d, g)
        if dot <= 0:
            return delta

        g_norm2 = torch.dot(g, g) + eps
        return delta - (dot / g_norm2) * gold_grad

    @torch.no_grad()
    def _gold_guided_project_soft(
            self,
            delta: torch.Tensor,
            gold_grad: torch.Tensor,
            beta: float = 0.3,
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        软投影：只移除一部分与 gold_grad 冲突的分量
        delta' = delta - beta * (delta^T g / ||g||^2) * g
        beta ∈ (0, 1)
        """
        if gold_grad is None or beta <= 0:
            return delta
        if delta.shape != gold_grad.shape:
            return delta

        gold_grad = gold_grad.to(device=delta.device, dtype=delta.dtype)

        d = delta.view(-1)
        g = gold_grad.view(-1)

        dot = torch.dot(d, g)
        if dot <= 0:
            return delta

        g_norm2 = torch.dot(g, g) + eps
        return delta - beta * (dot / g_norm2) * gold_grad

    def _aggregate(self, ids, updated_sizes, fedavg: bool = False):
        """
        golden-guided client-wise PCGrad（检索优先）：
        - 只对共享 attention(qkv/proj) 生效
        - 只修正分类客户端更新（img/txt cls），检索客户端更新不动
        - 其它参数聚合保持与父类一致
        """
        assert set(updated_sizes.keys()) == set(ids)
        # ===== 新增：取检索 anchor（mm）梯度，用于 hard constraint =====


        # 软投影强度（分类约束的弱化）
        soft_beta = float(getattr(self.args, "fedif_soft_beta", 0.3))  # 推荐 0.2~0.3

        # -------------------------------
        # ✅ [NEW] 选取“当前数据集对应的 g_val”
        # 支持:
        # 1) old: _latest_val_grads = {param->grad}
        # 2) new: _latest_val_grads = {dataset-> {param->grad}}
        # -------------------------------
        val_grads_pack = getattr(self, "_latest_val_grads", None)

        # 选择当前 dataset 的梯度（multi-anchor）
        if isinstance(val_grads_pack, dict) and self.dataset in val_grads_pack and isinstance(
                val_grads_pack[self.dataset], dict):
            val_grads = val_grads_pack[self.dataset]
            grads_src = f"anchor:{self.dataset}"
        else:
            # fallback：旧版/单源
            val_grads = val_grads_pack
            grads_src = "single-anchor"

        # 没有 golden 梯度就退化为原聚合
        if not val_grads:
            return super(FedifServer, self)._aggregate(ids, updated_sizes, fedavg=fedavg)
        ret_grads = None
        if isinstance(val_grads_pack, dict) and getattr(self, "mm_dataset_name", None) in val_grads_pack:
            ret_grads = val_grads_pack[self.mm_dataset_name]

        # -------------------------------
        # ✅ debug：只打印一次每个 (dataset|modality)
        # -------------------------------
        if not hasattr(self, "_debug_gold_key_done"):
            self._debug_gold_key_done = set()

        debug_tag = f"{self.dataset}|{self.modality}"
        if debug_tag not in self._debug_gold_key_done:
            self._debug_gold_key_done.add(debug_tag)

            gold_attn_keys = sorted([k for k in val_grads.keys() if ("attn.qkv" in k or "attn.proj" in k)])
            print(f"\n[GOLD-KEY-DEBUG] {debug_tag} ({grads_src}) #gold_attn_keys={len(gold_attn_keys)}")
            print("[GOLD-KEY-DEBUG] sample gold keys:")
            for k in gold_attn_keys[:10]:
                print("   ", k)

        # -------------------------------
        # 检索模型本体不做 cls-vs-gold 投影
        # -------------------------------
        if getattr(self, "modality", None) == "img+txt":
            return super(FedifServer, self)._aggregate(ids, updated_sizes, fedavg=fedavg)

        base_sd = self.global_model.cpu().required_params()
        final_sd = {k: v.clone() for k, v in base_sd.items()}

        # 预取客户端权重
        local_sds = {cid: dict(self.clients[cid].upload()) for cid in ids}

        # === 1) 复刻父类系数计算（保持你原来的逻辑不动） ===
        if fedavg:
            coefficients = {}
            for param_name in base_sd.keys():
                new_num = {}
                for cid, n in updated_sizes.items():
                    if self.param_scope[param_name] == 'all':
                        new_num[cid] = n
                    elif self.param_scope[param_name] == 'dataset':
                        new_num[cid] = n if self.clients[cid].dataset == self.dataset else 0
                    elif self.param_scope[param_name] == 'task':
                        new_num[cid] = n if self.clients[cid].task == self.task else 0
                    elif self.param_scope[param_name] == 'modality':
                        new_num[cid] = n if self.clients[cid].modality == self.modality else 0
                denom = sum(new_num.values())
                coefficients[param_name] = {cid: (new_num[cid] / denom if denom != 0 else 0.0) for cid in ids}
        else:
            coefficients = {}
            for param_name in base_sd.keys():
                new_num = {}
                old_sum = sum(updated_sizes.values())
                param_modality = get_name_modality(param_name, self.args.modalities)

                for cid, n in updated_sizes.items():
                    if self.param_scope[param_name] == 'all':
                        new_num[cid] = n
                    elif self.param_scope[param_name] == 'dataset':
                        new_num[cid] = n if self.clients[cid].dataset == self.dataset else 0
                    elif self.param_scope[param_name] == 'task':
                        new_num[cid] = n if self.clients[cid].task == self.task else 0
                    elif self.param_scope[param_name] == 'modality':
                        new_num[cid] = n if (
                                self.clients[cid].modality in self.modality or self.modality in self.clients[
                            cid].modality
                        ) else 0
                    elif self.param_scope[param_name] == 'modality_exact':
                        new_num[cid] = n if (
                                self.clients[cid].modality == param_modality or (
                                    param_modality and param_modality in self.clients[cid].modality)
                        ) else 0

                    if self.clients[cid].modality != self.modality and getattr(self, "out_modality_scale", 1.0) != 1:
                        old_sum -= new_num[cid]
                        new_num[cid] *= float(self.out_modality_scale)
                        old_sum += new_num[cid]

                if getattr(self.args, "compensation", False):
                    if self.args.share_scope == 'all':
                        denom = old_sum
                        coefficients[param_name] = {cid: (new_num[cid] / denom if denom != 0 else 0.0) for cid in ids}
                    elif self.args.share_scope == 'modality':
                        comp_size = sum(
                            size for cid2, size in updated_sizes.items()
                            if (self.clients[cid2].modality in self.modality or self.modality in self.clients[
                                cid2].modality)
                        )
                        coefficients[param_name] = {cid: (new_num[cid] / comp_size if comp_size != 0 else 0.0) for cid
                                                    in ids}
                    elif self.args.share_scope == 'modality_exact':
                        if param_modality:
                            comp_size = sum(
                                size for cid2, size in updated_sizes.items()
                                if (self.clients[cid2].modality == param_modality or param_modality in self.clients[
                                    cid2].modality)
                            )
                        else:
                            comp_size = sum(
                                size for cid2, size in updated_sizes.items()
                                if (self.clients[cid2].modality in self.modality or self.modality in self.clients[
                                    cid2].modality)
                            )
                        coefficients[param_name] = {cid: (new_num[cid] / comp_size if comp_size != 0 else 0.0) for cid
                                                    in ids}
                else:
                    denom = sum(new_num.values())
                    coefficients[param_name] = {cid: (new_num[cid] / denom if denom != 0 else 0.0) for cid in ids}

        # === 2) 非-attention 参数：完全按原逻辑聚合 ===
        with torch.no_grad():
            for cid in ids:
                local_sd = local_sds[cid]
                for param_name in base_sd.keys():
                    if self._is_shared_attn(param_name):
                        continue
                    if param_name not in local_sd:
                        continue
                    if not torch.is_floating_point(final_sd[param_name]):
                        continue
                    if local_sd[param_name].shape != final_sd[param_name].shape:
                        continue

                    coef = coefficients[param_name][cid]
                    if coef == 0.0:
                        continue
                    final_sd[param_name] += (local_sd[param_name] - final_sd[param_name]) * float(coef)

        # === 3) 共享 attention：golden-PCGrad + Fisher-Influence 逐参数聚合 ===
        # === 3) 共享 attention：检索优先（hard） + 分类软约束（soft） + influence*fisher 聚合 ===
        with torch.no_grad():
            for param_name in base_sd.keys():
                if not self._is_shared_attn(param_name):
                    continue

                base = base_sd[param_name].to(torch.float32)

                # -------- 1) 取两类 golden 梯度 --------
                # (A) 分类 anchor（当前 dataset）
                gold_cls = val_grads.get(param_name, None)
                if gold_cls is None:
                    mm_key_cls = self._map_param_to_mm_key(param_name, check_grads=val_grads)
                    gold_cls = val_grads.get(mm_key_cls, None)
                if gold_cls is not None:
                    gold_cls = gold_cls.to(torch.float32)

                # (B) 检索 anchor（mm_dataset）
                gold_ret = None
                if ret_grads:
                    mm_key_ret = self._map_param_to_mm_key(param_name, check_grads=ret_grads)
                    gold_ret = ret_grads.get(mm_key_ret, None)
                    if gold_ret is not None:
                        gold_ret = gold_ret.to(torch.float32)

                # -------- 2) influence*fisher 加权聚合 --------
                num = torch.zeros_like(base)
                den = torch.zeros_like(base)

                for cid in ids:
                    coef = coefficients[param_name][cid]
                    if coef == 0.0:
                        continue

                    local_sd = local_sds[cid]
                    if param_name not in local_sd:
                        continue

                    # 保留你原本的“检索/分类约束条件”
                    if self._is_ret_client(cid) and (self.modality not in self.clients[cid].modality):
                        continue
                    if self._is_cls_client(cid) and (self.clients[cid].modality != self.modality):
                        continue

                    local = local_sd[param_name].to(torch.float32)
                    delta = local - base

                    # -------- 3) 只对分类 client 做多约束投影（检索优先 hard -> 分类 soft）--------
                    if self._is_cls_client(cid):
                        delta_scaled = delta * float(coef)

                        # (1) hard: 检索优先（100%）
                        if gold_ret is not None:
                            delta_scaled = self._gold_guided_project(delta_scaled, gold_ret)

                        # (2) soft: 分类弱约束（beta 0.2~0.3）
                        if gold_cls is not None and soft_beta > 0:
                            delta_scaled = self._gold_guided_project_soft(delta_scaled, gold_cls, beta=soft_beta)

                        delta = delta_scaled / (float(coef) + 1e-12)

                    local_proj = base + delta

                    # influence（EMA 后的分数，本来就在 [eps,1] 附近，不会炸）
                    influence = float(self.client_influence.get(cid, 1.0))
                    if influence < 0:
                        influence = 0.0

                    # fisher（客户端提供；缺失则用 1）
                    f = self._get_client_fisher(cid, param_name, like=base)
                    if f is None:
                        f = torch.ones_like(base)
                    else:
                        f = self._normalize_fisher(f)

                    w = f * (float(coef) * influence)
                    num += w * local_proj
                    den += w

                new_param = torch.where(den > 0, num / (den + 1e-12), base)
                final_sd[param_name] = new_param.to(base_sd[param_name].dtype)

        self.global_model.load_state_dict(final_sd, strict=False)

    def _get_client_fisher(self, cid: int, param_name: str, like: torch.Tensor):
        """
        从 self.clients[cid].fisher_matrix 取指定参数的 fisher。
        - 缺失 / shape 不一致 -> None
        """
        fm = getattr(self.clients[cid], "fisher_matrix", None)
        if not fm:
            return None
        f = fm.get(param_name, None)
        if f is None:
            return None
        if tuple(f.shape) != tuple(like.shape):
            return None
        return f.to(device=like.device, dtype=like.dtype)

    def _normalize_fisher(self, f: torch.Tensor):
        """
        轻量归一化：clip + 除以均值（避免某些 client fisher 尺度极大）
        """
        clip = float(getattr(self.args, "fisher_clip", 10.0))
        eps = float(getattr(self.args, "fisher_norm_eps", 1e-6))
        if clip > 0:
            f = f.clamp(max=clip)
        return f / (f.mean() + eps)


    def update(self):
        """
        FedIF 主更新循环: Sample -> Train -> Val (Score) -> Aggregate
        """
        # 1. 采样客户端
        selected_ids = self._sample_clients()
        logger.info(f"[FedIF] Selected Clients: {selected_ids}")

        # 2. 本地训练
        # retain_model=True 对 FedIF 至关重要，否则无法获取权重计算影响力
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)

        # 3. FedIF 贡献评估 (训练后)
        if self.round > 2:
            val_grads_pack = self._calc_multi_anchor_validation_gradients()
            self._latest_val_grads = val_grads_pack
            self._update_influence_scores(selected_ids, val_grads_pack)

        # 4. 聚合 (FedCola 逻辑)
        for i, dataset in enumerate(self.global_models.keys()):
            self.global_model = self.global_models[dataset]
            self.task = DATASET_2_TASK[dataset]
            self.modality = DATASET_2_MODALITY[dataset]
            self.dataset = dataset

            try:
                self.out_modality_scale = self.args.out_modality_scales[i]
            except (AttributeError, IndexError):
                self.out_modality_scale = 1.0

            self._aggregate(selected_ids, updated_sizes)
            self.global_models[dataset] = self.global_model

        # 5. 辅助模态聚合 (FedCola 特有逻辑)
        if self.args.with_aux:
            for dataset in self.global_models.keys():
                self.global_model = self.global_models[dataset]
                self.modality = DATASET_2_MODALITY[dataset]

                if self.modality == 'img+txt':
                    continue
                elif self.modality == 'img':
                    txt_dataset = next((d for d in self.global_models.keys() if DATASET_2_MODALITY[d] == 'txt'), None)
                    if txt_dataset:
                        aux_model = self.global_models[txt_dataset]
                        sd = aux_model.state_dict()
                        auxes = {}
                        for k in self.global_model.aux_params().keys():
                            aux_key_src = k.replace('aux_', '').replace('blockses.0', 'blockses.1')
                            if aux_key_src in sd: auxes.update({k: sd[aux_key_src]})
                        self.global_model.load_state_dict(auxes, strict=False)
                elif self.modality == 'txt':
                    img_dataset = next((d for d in self.global_models.keys() if DATASET_2_MODALITY[d] == 'img'), None)
                    if img_dataset:
                        aux_model = self.global_models[img_dataset]
                        sd = aux_model.state_dict()
                        auxes = {}
                        for k in self.global_model.aux_params().keys():
                            aux_key_src = k.replace('aux_', '').replace('blockses.1', 'blockses.0')
                            if aux_key_src in sd: auxes.update({k: sd[aux_key_src]})
                        self.global_model.load_state_dict(auxes, strict=False)

        # 6. 学习率衰减 & 清理
        if self.round % self.args.lr_decay_step == 0:
            self.curr_lr *= self.args.lr_decay

        # 7. Dump resource metrics periodically
        if self.resource_selector is not None and self._round_resource_log:
            self._dump_resource_metrics()

        self._empty_client_models()
        return selected_ids

    # ------------------------------------------------------------------
    # Resource telemetry persistence
    # ------------------------------------------------------------------
    def _dump_resource_metrics(self):
        """Append per-round resource metrics to a JSON-lines file."""
        import json as _json
        path = os.path.join(getattr(self.args, "result_path", "."), "resource_metrics.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                for rec in self._round_resource_log:
                    f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            self._round_resource_log.clear()
        except Exception as e:
            logger.warning(f"[FedIF] resource metrics dump failed: {e}")
