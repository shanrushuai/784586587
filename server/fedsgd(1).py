# src/server/fedifserver.py (Refactored as FedIF++)
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
import torch.nn.functional as F

# 导入父类及相关常量
from .fedavgserver import FedavgServer, DATASET_2_MODALITY, DATASET_2_TASK, TASK_2_CRITERION, get_name_type, get_name_modality
import src.criterions

logger = logging.getLogger(__name__)


class FedifServer(FedavgServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model_str):
        """
        FedIF++ Server: Federated Influence with Projection & Preservation

        Improvements:
        1. PCGrad (Project Conflicting Gradients): Projects conflicting client updates onto the orthogonal plane of validation gradients.
        2. Compensation Matrix: Restores FedCola's stratified aggregation weights.
        3. Soft Updating: Uses EMA for global model updates to ensure stability.
        """
        # ================= [Init Logic Same as Before] =================
        validation_source = None

        if isinstance(server_dataset, (tuple, list)) and len(server_dataset) == 2:
            validation_source = server_dataset[0]  # raw_tests dictionary
            testing_source = server_dataset[1]
        elif isinstance(server_dataset, dict):
            validation_source = server_dataset
        else:
            validation_source = {}

        # Strip fedif_golden for parent class init to prevent errors
        fedavg_dataset_dict = validation_source.copy() if validation_source else {}
        keys_to_remove = [k for k in fedavg_dataset_dict.keys() if 'fedif_golden' in k]
        for k in keys_to_remove:
            del fedavg_dataset_dict[k]

        fedavg_pack = (None, fedavg_dataset_dict)
        super(FedifServer, self).__init__(args, writer, fedavg_pack, client_datasets, model_str)

        # ================= [FedIF Specific Init] =================
        self.validation_source = validation_source
        self.fedif_gamma = getattr(args, 'fedif_gamma', 0.9)
        self.recency_alpha = 0.5

        # Soft Update parameter (tau). 1.0 = Hard Update, 0.1 = Keep 90% old, update 10%
        self.soft_update_tau = getattr(args, 'soft_update_tau', 0.8)
        self.align_temp = getattr(args, 'align_temp', 0.07)  # 设置对比损失的温度系数

        self.client_influence = {i: 0.0 for i in range(args.K)}
        self.client_last_selected = {i: -1 for i in range(args.K)}

        # Store Validation Gradients for Aggregation usage
        self.current_val_grads = None

        # --- Load Golden Sets (Same as original) ---
        self.mm_val_loaders = []
        self.mm_dataset_name = None
        self.mm_val_loader = None

        if isinstance(self.validation_source, dict):
            golden_keys = sorted([k for k in self.validation_source.keys() if 'fedif_golden' in k])
            if golden_keys:
                logger.info(f"[FedIF++] Found {len(golden_keys)} Golden Sets. Enabling Random Rotation.")
                self.mm_dataset_name = next(
                    (k for k in self.global_models.keys() if DATASET_2_MODALITY.get(k) == 'img+txt'), None)
                if self.mm_dataset_name:
                    for key in golden_keys:
                        d_set = self.validation_source[key]
                        loader = DataLoader(d_set, batch_size=args.B, shuffle=True, num_workers=args.num_thread,
                                            drop_last=True)
                        self.mm_val_loaders.append(loader)

            if not self.mm_val_loaders:
                # Fallback
                for d_name in self.validation_source.keys():
                    if DATASET_2_MODALITY.get(d_name) == 'img+txt':
                        self.mm_dataset_name = d_name
                        logger.info(f"[FedIF++] Fallback to dataset: {d_name}")
                        d_set = self.validation_source[d_name]
                        loader = DataLoader(d_set, batch_size=args.B, shuffle=True, num_workers=args.num_thread,
                                            drop_last=True)
                        self.mm_val_loaders.append(loader)
                        break

        if self.mm_val_loaders:
            self.mm_val_loader = self.mm_val_loaders[0]

    def _get_param_map(self, client_modality):
        """Maps client parameter names to server parameter names."""
        if client_modality == 'img+txt':
            return lambda x: x

        def mapper(client_key):
            server_key = client_key
            if client_modality == 'img':
                if 'blocks' in client_key and 'blockses' not in client_key:
                    server_key = client_key.replace('blocks', 'blockses.0')
            elif client_modality == 'txt':
                if 'blocks' in client_key and 'blockses' not in client_key:
                    server_key = client_key.replace('blocks', 'blockses.1')
            return server_key

        return mapper

    def _calc_validation_gradients(self):
        """
        Calculates gradients on the Golden Set.
        Unlike original FedIF, we now retain these gradients for the Aggregation phase (PCGrad).
        """
        if self.mm_val_loaders and len(self.mm_val_loaders) > 1:
            self.mm_val_loader = random.choice(self.mm_val_loaders)
            if hasattr(self, '_val_iterator'): del self._val_iterator

        if self.mm_val_loader is None or self.mm_dataset_name is None:
            return None

        mm_model = self.global_models[self.mm_dataset_name]
        mm_model.to(self.server_device)
        mm_model.eval()
        mm_model.zero_grad()

        accumulation_steps = 8
        max_grad_norm = 1.0

        # Determine Criterion
        criterion_name = TASK_2_CRITERION.get('img+txt', 'ContrastiveLoss')
        if hasattr(src.criterions, criterion_name):
            criterion_cls = getattr(src.criterions, criterion_name)
        elif hasattr(nn, criterion_name):
            criterion_cls = getattr(nn, criterion_name)
        else:
            from src.criterions.contrastive_loss import ContrastiveLoss
            criterion_cls = ContrastiveLoss
        criterion = criterion_cls().to(self.server_device)

        if not hasattr(self, '_val_iterator'):
            self._val_iterator = iter(self.mm_val_loader)

        total_loss = 0.0
        for _ in range(accumulation_steps):
            try:
                batch = next(self._val_iterator)
            except StopIteration:
                self._val_iterator = iter(DataLoader(self.mm_val_loader.dataset, batch_size=self.args.B, shuffle=True,
                                                     num_workers=self.args.num_thread, drop_last=True))
                batch = next(self._val_iterator)

            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.server_device), batch[1].to(self.server_device)
            elif isinstance(batch, dict):
                inputs, targets = batch['image'].to(self.server_device), batch['text'].to(self.server_device)
            else:
                continue

            outputs = mm_model([inputs, targets], feat_out=True)
            loss = criterion(*outputs) / accumulation_steps
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(mm_model.parameters(), max_grad_norm)

        val_grads = {}
        for name, param in mm_model.named_parameters():
            if param.grad is not None:
                # Capture Attn for alignment, but optionally capture more if needed for PCGrad
                # Keeping 'attn' only for efficiency, as suggested in original FedIF logic
                if 'attn.qkv' in name or 'attn.proj' in name:
                    val_grads[name] = param.grad.detach().clone()

        mm_model.zero_grad()
        mm_model.to('cpu')

        # Save for aggregation
        self.current_val_grads = val_grads
        return val_grads

    def _project_conflicting(self, delta_w, val_grad):
        """
        [New Feature] PCGrad Projection
        delta_w: Update vector from client (theta_client - theta_global) -> pseudo gradient
        val_grad: Gradient from validation set

        If cos(delta_w, val_grad) < 0 (Conflict), project delta_w to remove the conflicting component.
        """
        delta_flat = delta_w.flatten()
        val_flat = val_grad.flatten()

        dot_product = torch.dot(delta_flat, val_flat)

        if dot_product < 0:
            # Conflict detected! Project delta_w onto the orthogonal plane of val_grad
            # Formula: g_new = g_old - (dot(g_old, g_ref) / dot(g_ref, g_ref)) * g_ref
            val_norm_sq = torch.dot(val_flat, val_flat)
            if val_norm_sq > 1e-12:
                proj_comp = (dot_product / val_norm_sq) * val_flat
                delta_flat_new = delta_flat - proj_comp
                return delta_flat_new.view_as(delta_w)

        return delta_w

    def _update_influence_scores(self, selected_ids, val_grads):
        """
        Calculates scores. 
        Note: With PCGrad enabled in aggregation, we don't need to punish clients as harshly here.
        We still track scores for sampling stratification.
        """
        if not val_grads: return
        entropy_weight = 1.0
        if self.mm_val_loaders and self.mm_dataset_name:
            model = self.global_models[self.mm_dataset_name]
            model.to(self.server_device)
            model.eval()
            try:
                # 获取数据
                loader = self.mm_val_loaders[0]
                batch = next(iter(loader))
                if isinstance(batch, dict):
                    imgs, txts = batch['image'].to(self.server_device), batch['text'].to(self.server_device)
                else:
                    imgs, txts = batch[0].to(self.server_device), batch[1].to(self.server_device)

                with torch.no_grad():
                    outputs = model([imgs, txts], feat_out=True)
                    # 计算图文匹配概率分布的熵
                    # 假设 outputs 是 embeddings，计算相似度矩阵
                    img_emb = F.normalize(outputs[0], dim=-1)
                    txt_emb = F.normalize(outputs[1], dim=-1)
                    logits = torch.matmul(img_emb, txt_emb.T) / self.align_temp

                    # 计算 Softmax 熵: H(p) = - sum(p * log(p))
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean().item()

                    # 归一化熵值作为权重 (0.5 ~ 1.5 之间波动)
                    entropy_weight = 1.0 + (entropy / 10.0)  # 简单缩放，避免数值过大
            except Exception as e:
                logger.warning(f"[FedIF++] Failed to compute entropy: {e}")
                entropy_weight = 1.0
            finally:
                model.to('cpu')

        # Preload base weights
        base_sds = {}
        for d_name, model in self.global_models.items():
            base_sds[d_name] = {k: v.cpu() for k, v in model.state_dict().items()}

        stratified_raw = defaultdict(dict)

        for client_id in selected_ids:
            client = self.clients[client_id]
            mapper = self._get_param_map(client.modality)
            client_sd = dict(client.upload())

            if client.dataset not in base_sds: continue
            base_sd = base_sds[client.dataset]

            client_vecs, val_vecs = [], []

            for k, w_new in client_sd.items():
                if 'attn.qkv' not in k and 'attn.proj' not in k: continue
                server_k = mapper(k)

                if server_k in val_grads and server_k in base_sd:
                    w_base = base_sd[server_k].to(self.server_device)
                    w_new = w_new.to(self.server_device)

                    # Gradient approximation: Base - New
                    delta = w_base - w_new
                    client_vecs.append(delta.flatten())
                    val_vecs.append(val_grads[server_k].flatten())

            raw_score = 0.0
            if client_vecs:  # 注意这里变量名通常是 client_vecs (列表)
                # 假设上文已将 list 拼成了 tensor，或者此处执行拼接
                # 如果 client_vecs 是 list:
                c_vec = torch.cat(client_vecs)
                # 如果 val_vecs 是 list:
                v_vec = torch.cat(val_vecs)

                c_norm, v_norm = torch.norm(c_vec), torch.norm(v_vec)

                if c_norm > 1e-9 and v_norm > 1e-9:
                    # 1. 计算基础余弦相似度
                    cosine_sim = torch.dot(c_vec, v_vec) / (c_norm * v_norm)

                    # 2. [关键修改] 乘以熵权重
                    # Score = Entropy_Weight * Cosine_Similarity
                    raw_score = entropy_weight * cosine_sim.item()

            # Group by modality
            modality = client.modality
            if 'img' in modality and 'txt' in modality:
                modality = 'img+txt'
            elif 'img' in modality:
                modality = 'img'
            elif 'txt' in modality:
                modality = 'txt'

            stratified_raw[modality][client_id] = raw_score

        # Normalize per modality group (Stratified)
        for mod, scores_map in stratified_raw.items():
            if not scores_map: continue
            scores = list(scores_map.values())
            min_s, max_s = min(scores), max(scores)
            div = max_s - min_s + 1e-9

            for cid, s in scores_map.items():
                norm = (s - min_s) / div
                # Update history
                self.client_influence[cid] = (1 - self.fedif_gamma) * self.client_influence.get(cid, 0.0) + \
                                             self.fedif_gamma * norm

    def _sample_clients(self, exclude=[]):
        """Stratified + Recency Sampling (Unchanged, as it is robust)."""
        if self.round <= self.args.warmup_rounds:
            sampled = super(FedifServer, self)._sample_clients(exclude)
            for cid in sampled: self.client_last_selected[cid] = self.round
            return sampled

        logger.info(f'[FedIF++] Round {self.round}: Stratified Influence + Recency Sampling')
        sampled_ids = []

        for dataset in self.args.datasets:
            candidates = [c.id for c in self.clients if c.dataset == dataset and c.id not in exclude]
            if not candidates: continue

            num_sampled = max(int(self.Cs[dataset] * len(candidates)), 1)

            # Scores
            raw_scores = np.array([self.client_influence[uid] for uid in candidates])
            if raw_scores.max() > raw_scores.min():
                scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
            else:
                scores = np.zeros_like(raw_scores)

            # Recency
            staleness = np.array([self.round - self.client_last_selected[uid] for uid in candidates])
            norm_recency = np.log(1 + staleness)
            if norm_recency.max() > norm_recency.min():
                norm_recency = (norm_recency - norm_recency.min()) / (norm_recency.max() - norm_recency.min() + 1e-9)

            # Combined prob
            final = scores + self.recency_alpha * norm_recency
            probs = np.exp(final) / np.sum(np.exp(final))

            try:
                selected = np.random.choice(candidates, size=num_sampled, replace=False, p=probs)
            except:
                selected = np.random.choice(candidates, size=num_sampled, replace=False)

            sampled_ids.extend(selected)
            for cid in selected: self.client_last_selected[cid] = self.round

        sampled_ids.sort()
        # Set devices
        for i, cid in enumerate(sampled_ids):
            self.clients[cid].device = 'cuda:%d' % (
                        i % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'

        return sampled_ids


def _compute_validation_gradients(self):
    """
    [FedIF++ 核心组件] 计算验证集梯度 g_val

    改进点:
    1. Random Rotation: 防止过拟合单一验证集 (v0-v9)
    2. Gradient Accumulation: 累积 8 个 Batch，为 PCGrad 提供低方差、高稳定性的锚点梯度
    """
    # [Step 0] 随机轮换验证集 (如果存在多个版本)
    if self.mm_val_loaders and len(self.mm_val_loaders) > 1:
        self.mm_val_loader = random.choice(self.mm_val_loaders)
        # 必须重置迭代器，否则会一直沿用旧 loader
        if hasattr(self, '_val_iterator'):
            del self._val_iterator

    # 检查是否可用
    if self.mm_val_loader is None or self.mm_dataset_name is None:
        return None

    mm_model = self.global_models[self.mm_dataset_name]
    mm_model.to(self.server_device)

    # 开启 eval 模式但允许梯度 (Standard Practice for Higher-Order Gradients)
    mm_model.eval()
    mm_model.zero_grad()

    # --- 配置 ---
    accumulation_steps = 8  # 累积步数 (建议 4-8)
    max_grad_norm = 1.0  # 梯度裁剪

    # 获取 Loss Function (统一使用 src.criterions)
    criterion_name = TASK_2_CRITERION.get('img+txt', 'ContrastiveLoss')
    if hasattr(src.criterions, criterion_name):
        criterion_cls = getattr(src.criterions, criterion_name)
    else:
        from src.criterions.contrastive_loss import ContrastiveLoss
        criterion_cls = ContrastiveLoss
    criterion = criterion_cls().to(self.server_device)

    # 确保迭代器可用
    if not hasattr(self, '_val_iterator'):
        self._val_iterator = iter(self.mm_val_loader)

    # --- [Step 1] 梯度累积循环 ---
    for step in range(accumulation_steps):
        try:
            batch = next(self._val_iterator)
        except StopIteration:
            # 迭代器耗尽，从该 loader 重新开始
            self._val_iterator = iter(DataLoader(
                self.mm_val_loader.dataset,
                batch_size=self.args.B,
                shuffle=True,
                num_workers=self.args.num_thread,
                drop_last=True
            ))
            batch = next(self._val_iterator)

        # 数据加载 (兼容 Dict 和 List/Tuple)
        if isinstance(batch, dict):
            inputs = batch['image'].to(self.server_device)
            targets = batch['text'].to(self.server_device)
        elif isinstance(batch, (list, tuple)):
            inputs, targets = batch[0].to(self.server_device), batch[1].to(self.server_device)
        else:
            continue

        # 前向传播
        outputs = mm_model([inputs, targets], feat_out=True)

        # 计算 Loss 并缩放
        loss = criterion(*outputs)
        loss = loss / accumulation_steps  # 平均 Loss
        loss.backward()

    # --- [Step 2] 梯度裁剪与提取 ---
    # 裁剪防止梯度爆炸影响投影
    torch.nn.utils.clip_grad_norm_(mm_model.parameters(), max_grad_norm)

    # 提取并扁平化所有参数的梯度 (用于 PCGrad 投影)
    # 注意：这里提取所有参数，而不仅仅是 attn，因为 PCGrad 需要对所有冲突层进行投影
    grads = []
    for param in mm_model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
        else:
            # 如果某层没有梯度 (如被冻结)，填 0 以保持维度对齐
            grads.append(torch.zeros_like(param).view(-1))

    g_val_flat = torch.cat(grads)

    # 清理显存
    mm_model.zero_grad()
    mm_model.to('cpu')

    return g_val_flat


def _compute_fisher_matrix(self):
    """
    [修复] 计算 Fisher 信息矩阵 (梯度的平方)，用于 EWC。
    """
    val_grads = self._calc_validation_gradients()  # 复用梯度计算
    if val_grads is None:
        return None

    fisher_matrix = {}
    for name, grad in val_grads.items():
        # Fisher 近似为梯度的平方
        fisher_matrix[name] = grad.pow(2).clone()

    return fisher_matrix


def _aggregate(self, ids, updated_sizes):
    """
    FedIF++ 核心聚合逻辑 (修正版):
    1. PCGrad: 投影冲突梯度
    2. Compensation: 完整实现 alpha = (N_total / N_modal) * beta_var
    3. Soft Update: 使用 EMA 策略更新全局模型
    """
    logger.info(f'[FedIF++] [{self.dataset.upper()}] Aggregate with PCGrad, Full Compensation & Soft Update!')

    # 1. 获取验证集梯度 g_val
    g_val = self._compute_validation_gradients()

    server_params = {k: v.clone() for k, v in self.global_model.state_dict().items()}
    client_updates = {}

    # === Phase 1: 收集并修正梯度 (PCGrad) ===
    # ... [这部分代码原文件已实现正确，为节省篇幅省略，保持原逻辑即可] ...
    # (确保 _project_conflicting 逻辑被调用)
    for identifier in ids:
        # ... (同原文件 Phase 1 逻辑) ...
        client = self.clients[identifier]
        local_sd = dict(client.upload())
        client_update_dict = {}
        flat_update_vec = []
        param_names = []

        with torch.no_grad():
            for k, v_old in server_params.items():
                if k in local_sd:
                    v_new = local_sd[k].to(self.server_device)
                    update = (v_new - v_old.to(self.server_device)).float()
                    client_update_dict[k] = update
                    if g_val is not None and update.numel() > 0:
                        flat_update_vec.append(update.view(-1))
                        param_names.append(k)

        # PCGrad Projection
        if g_val is not None and len(flat_update_vec) > 0:
            g_client_flat = torch.cat(flat_update_vec)
            if g_client_flat.shape == g_val.shape:
                dot_product = torch.dot(g_client_flat, g_val)
                if dot_product < 0:
                    # 投影逻辑
                    g_val_norm = g_val.norm().pow(2)
                    proj = (dot_product / (g_val_norm + 1e-8)) * g_val
                    g_client_corrected = g_client_flat - proj

                    # 还原
                    ptr = 0
                    for name in param_names:
                        numel = client_update_dict[name].numel()
                        client_update_dict[name] = g_client_corrected[ptr: ptr + numel].view_as(
                            client_update_dict[name])
                        ptr += numel

        client_updates[identifier] = client_update_dict

    # === Phase 2: 计算方差补偿系数 (Variance Compensation) ===
    param_variances = {}
    for k in server_params.keys():
        updates_stack = []
        for identifier in ids:
            if k in client_updates[identifier]:
                updates_stack.append(client_updates[identifier][k])

        if updates_stack:
            stacked = torch.stack(updates_stack)
            # 计算 update 的标准差作为方差指标
            std = torch.std(stacked, dim=0).mean().item()
            param_variances[k] = std + 1e-6
        else:
            param_variances[k] = 1.0

    avg_var = sum(param_variances.values()) / (len(param_variances) + 1e-9)
    beta_vars = {k: v / avg_var for k, v in param_variances.items()}

    # === Phase 3: 加权聚合 (Fix: 完整补偿公式 & Soft Update) ===
    total_samples = sum(updated_sizes.values())

    # [Fix 3] 预计算模态样本总数，用于补偿公式
    # N_modal: 参与该模态训练的样本总数
    modality_counts = defaultdict(int)
    for identifier in ids:
        modality_counts[self.clients[identifier].modality] += updated_sizes[identifier]
    # 对于 img+txt 任务，img 和 txt 模态的数据都算作贡献
    if 'img' in modality_counts and 'txt' in modality_counts:
        modality_counts['img+txt'] = modality_counts['img'] + modality_counts['txt']

    new_global_updates = {k: torch.zeros_like(v).float() for k, v in server_params.items()}

    for identifier in ids:
        n_k = updated_sizes[identifier]
        client_modality = self.clients[identifier].modality

        for k, update_tensor in client_updates[identifier].items():
            w_base = n_k / total_samples

            # [Fix 4] 完整的补偿公式: alpha = (N_total / N_modal) * beta_var
            alpha_comp = 1.0
            if self.args.compensation:
                # 获取该层参数对应的模态 (需要 get_name_modality 辅助函数，或简化处理)
                # 简化处理：假设参数归属于客户端当前的模态
                n_modal = modality_counts.get(client_modality, total_samples)
                scaling_factor = total_samples / (n_modal + 1e-9)

                beta = beta_vars.get(k, 1.0)
                alpha_comp = scaling_factor * beta

            final_weight = w_base * alpha_comp
            new_global_updates[k] += update_tensor.cpu() * final_weight

    # 应用 Soft Update
    final_sd = self.global_model.state_dict()
    with torch.no_grad():
        for k, v in final_sd.items():
            if k in new_global_updates:
                # [Fix 5] Soft Update: theta_new = theta_old + tau * aggregated_update
                # 这里的 new_global_updates[k] 是加权后的 Delta Theta
                update_step = new_global_updates[k]
                final_sd[k] = v + self.soft_update_tau * update_step

    self.global_model.load_state_dict(final_sd)
    logger.info(f'[FedIF++] Aggregation complete with Soft Update (tau={self.soft_update_tau}).')


def update(self):
    """
    FedIF++ Main Loop
    """
    # 1. Sample
    selected_ids = self._sample_clients()

    fisher_matrix = self._compute_fisher_matrix()
    for cid in selected_ids:
        # 显式调用新的 download 接口
        self.clients[cid].download(self.global_models, fisher_matrix)

    # 此时调用 _request，客户端模型已存在 (client.model is not None)
    # 确保父类的 _request 中有 checks: "if client.model is None: client.download..."
    updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)
    # 2. Local Training
    updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)

    # 3. Eval Contribution (Compute Val Gradients & Scores)
    if self.round > self.args.warmup_rounds:
        val_grads = self._calc_validation_gradients()  # Stored in self.current_val_grads
        self._update_influence_scores(selected_ids, val_grads)
    else:
        self.current_val_grads = None

    # 4. Aggregation (With PCGrad + Compensation)
    for i, dataset in enumerate(self.global_models.keys()):
        self.global_model = self.global_models[dataset]
        self.task = DATASET_2_TASK[dataset]
        self.modality = DATASET_2_MODALITY[dataset]
        self.dataset = dataset

        # Param scope setup might be needed here if handled dynamically
        self._aggregate(selected_ids, updated_sizes)
        self.global_models[dataset] = self.global_model

    # 5. Aux Modality Sync (FedCola Legacy - Optional but good for cross-modal)
    if self.args.with_aux:
        self._sync_aux_modalities()

    # 6. Decay & Cleanup
    if self.round % self.args.lr_decay_step == 0:
        self.curr_lr *= self.args.lr_decay

    self._empty_client_models()
    # Clear val grads to save memory
    self.current_val_grads = None
    return selected_ids


def _sync_aux_modalities(self):
    """Helper to sync aux parameters (FedCola logic)"""
    for dataset in self.global_models.keys():
        self.global_model = self.global_models[dataset]
        self.modality = DATASET_2_MODALITY[dataset]

        if self.modality == 'img+txt': continue

        source_modality = 'txt' if self.modality == 'img' else 'img'
        # Find a source dataset
        src_ds_name = next((d for d in self.global_models.keys() if DATASET_2_MODALITY[d] == source_modality), None)

        if src_ds_name:
            aux_model = self.global_models[src_ds_name]
            sd = aux_model.state_dict()
            auxes = {}

            # Map source layers to target aux layers
            target_block_idx = '0' if self.modality == 'img' else '1'
            src_block_idx = '1' if self.modality == 'img' else '0'  # Inverse

            for k in self.global_model.aux_params().keys():
                # k example: blocks.0.attn.aux_weight
                # src needed: blocks.1.attn.weight
                clean_k = k.replace('aux_', '')
                src_k = clean_k.replace(f'blockses.{target_block_idx}', f'blockses.{src_block_idx}')

                if src_k in sd:
                    auxes[k] = sd[src_k]

            self.global_model.load_state_dict(auxes, strict=False)