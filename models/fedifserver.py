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
from .fedavgserver import FedavgServer, DATASET_2_MODALITY, DATASET_2_TASK, TASK_2_CRITERION
import src.criterions

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

        # 7. 准备多模态验证数据 (用于计算 g_val)
        self.mm_val_loaders = []  # 存储所有版本的 Loader
        self.mm_dataset_name = None
        self.mm_val_loader = None  # 当前轮次使用的 loader

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

    def _calc_validation_gradients(self):
        """
        [修改后] 计算服务器端多模态模型在验证集上的梯度 (g_val)
        特性：
        1. Random Rotation: 每轮随机切换验证集版本。
        2. Gradient Accumulation: 累积 8 个 Batch 以平滑梯度。
        """
        # [Step 0] 随机轮换验证集 (从 v0-v4 中随机选一个)
        if self.mm_val_loaders and len(self.mm_val_loaders) > 1:
            self.mm_val_loader = random.choice(self.mm_val_loaders)
            # 切换 Loader 后务必删除旧迭代器，确保下次从新 Loader 取数据
            if hasattr(self, '_val_iterator'):
                del self._val_iterator

        # 检查是否可用
        if self.mm_val_loader is None or self.mm_dataset_name is None:
            return None

        mm_model = self.global_models[self.mm_dataset_name]
        mm_model.to(self.server_device)

        # 必须开启 eval 模式但允许梯度 (Standard Practice for Influence Function)
        mm_model.eval()
        mm_model.zero_grad()

        # --- [配置] 梯度累积参数 ---
        accumulation_steps = 8  # 累积 8 个 Batch
        max_grad_norm = 1.0  # 梯度裁剪阈值

        # 获取 Loss Function
        criterion_name = TASK_2_CRITERION.get('img+txt', 'ContrastiveLoss')
        if hasattr(src.criterions, criterion_name):
            criterion_cls = getattr(src.criterions, criterion_name)
        elif hasattr(nn, criterion_name):
            criterion_cls = getattr(nn, criterion_name)
        else:
            from src.criterions.contrastive_loss import ContrastiveLoss
            criterion_cls = ContrastiveLoss
        criterion = criterion_cls().to(self.server_device)

        # 确保迭代器可用
        if not hasattr(self, '_val_iterator'):
            self._val_iterator = iter(self.mm_val_loader)

        # --- [Step 1] 核心循环：梯度累积 ---
        total_loss_val = 0.0

        for step in range(accumulation_steps):
            try:
                batch = next(self._val_iterator)
            except StopIteration:
                # 迭代器耗尽，重建并重新获取
                self._val_iterator = iter(DataLoader(
                    self.mm_val_loader.dataset,
                    batch_size=self.args.B,
                    shuffle=True,
                    num_workers=self.args.num_thread,
                    drop_last=True
                ))
                batch = next(self._val_iterator)

            # 数据迁移 (兼容 tuple 和 dict 两种格式)
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.server_device), batch[1].to(self.server_device)
            elif isinstance(batch, dict):
                # 适配 FedIFGoldenDataset 返回字典的情况
                inputs = batch['image'].to(self.server_device)
                targets = batch['text'].to(self.server_device)
            else:
                continue

            # 前向传播
            outputs = mm_model([inputs, targets], feat_out=True)

            # 计算 Loss
            loss = criterion(*outputs)

            # [关键] Loss 除以步数，实现平均效果
            loss = loss / accumulation_steps
            loss.backward()

            total_loss_val += loss.item()

        # --- [Step 2] 梯度裁剪与提取 ---
        torch.nn.utils.clip_grad_norm_(mm_model.parameters(), max_grad_norm)

        val_grads = {}
        for name, param in mm_model.named_parameters():
            if param.grad is not None:
                # 只提取 Attention 层 (QKV, Proj) 作为特征指纹
                if 'attn.qkv' in name or 'attn.proj' in name:
                    val_grads[name] = param.grad.detach().clone()

        # 清理显存
        mm_model.zero_grad()
        mm_model.to('cpu')

        return val_grads

    def _update_influence_scores(self, selected_ids, val_grads):
        """
        计算客户端影响力: Dot(Client_Update, Validation_Gradient)
        [精进] 使用分层归一化 (Stratified Normalization) 确保各模态公平竞争
        """
        if val_grads is None or len(val_grads) == 0:
            return

        logger.info(f"[FedIF] Evaluating contribution of {len(selected_ids)} clients...")

        # 1. 预加载 Base Model 参数 (CPU) 避免循环 IO
        base_sds = {}
        for dataset_name, model in self.global_models.items():
            base_sds[dataset_name] = {k: v.cpu() for k, v in model.state_dict().items()}

        # 临时存储原始分数，按模态分组: {'img': {id: score}, 'txt': {...}}
        stratified_raw_scores = defaultdict(dict)

        for client_id in selected_ids:
            client = self.clients[client_id]
            key_mapper = self._get_param_map(client.modality)
            client_sd = dict(client.upload())

            if client.dataset not in base_sds:
                continue
            base_sd = base_sds[client.dataset]

            client_vec_list = []
            val_grad_vec_list = []

            # --- 计算向量点积/余弦相似度 ---
            for client_k, client_w in client_sd.items():
                # 过滤层
                if 'attn.qkv' not in client_k and 'attn.proj' not in client_k:
                    continue

                server_k = key_mapper(client_k)

                # 匹配检查
                if server_k in val_grads and server_k in base_sd:
                    w_base = base_sd[server_k].to(self.server_device)
                    w_new = client_w.to(self.server_device)

                    # Delta W = Base - New (近似梯度方向)
                    delta_w = w_base - w_new

                    if delta_w.shape != val_grads[server_k].shape:
                        continue

                    client_vec_list.append(delta_w.flatten())
                    val_grad_vec_list.append(val_grads[server_k].flatten())

            raw_score = 0.0
            if len(client_vec_list) > 0:
                client_vec = torch.cat(client_vec_list)
                val_vec = torch.cat(val_grad_vec_list)

                # 使用余弦相似度消除量级差异 (FedIF 推荐)
                c_norm = torch.norm(client_vec)
                v_norm = torch.norm(val_vec)
                if c_norm > 1e-9 and v_norm > 1e-9:
                    raw_score = torch.dot(client_vec, val_vec) / (c_norm * v_norm)
                    raw_score = raw_score.item()

            # 按模态分组记录
            modality_key = client.modality
            if 'img' in modality_key and 'txt' in modality_key:
                modality_key = 'img+txt'
            elif 'img' in modality_key:
                modality_key = 'img'
            elif 'txt' in modality_key:
                modality_key = 'txt'

            stratified_raw_scores[modality_key][client_id] = raw_score

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
                norm_score = (r_score - min_s) / div

                # EMA 更新历史分数
                old_score = self.client_influence.get(cid, 0.0)
                self.client_influence[cid] = (1 - self.fedif_gamma) * old_score + \
                                             self.fedif_gamma * norm_score

        # 打印 Top-10
        sorted_influence = sorted(self.client_influence.items(), key=lambda x: x[1], reverse=True)
        top_k_str = ", ".join([f"ID{k}:{v:.4f}" for k, v in sorted_influence[:10]])
        logger.info(f"[FedIF] Top-10 Influential Clients (Stratified): {top_k_str}")

    def _sample_clients(self, exclude=[]):
        """
        基于 '分层影响力 + 陈旧度' 的混合采样
        """
        # Warmup Phase: 随机采样
        if self.round <= self.args.warmup_rounds:
            sampled = super(FedifServer, self)._sample_clients(exclude)
            for cid in sampled:
                self.client_last_selected[cid] = self.round
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
            if raw_inf_scores.max() > raw_inf_scores.min():
                inf_scores = (raw_inf_scores - raw_inf_scores.min()) / (
                            raw_inf_scores.max() - raw_inf_scores.min() + 1e-9)
            else:
                inf_scores = np.zeros_like(raw_inf_scores)

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
            temperature = 1.0
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

        return sampled_client_ids

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
        if self.round > self.args.warmup_rounds:
            val_grads = self._calc_validation_gradients()
            self._update_influence_scores(selected_ids, val_grads)

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

        self._empty_client_models()
        return selected_ids