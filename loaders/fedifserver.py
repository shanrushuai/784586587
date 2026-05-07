import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import random
import math
import os
from collections import defaultdict
from torch.utils.data import DataLoader

# 导入基础组件
from src.server.fedavgserver import FedavgServer, DATASET_2_MODALITY, DATASET_2_TASK, TASK_2_CRITERION
import src.criterions
# 导入评估器
from src.fedif_evaluator import FedIfEvaluator

logger = logging.getLogger(__name__)


class FedifServer(FedavgServer):
    """
    FedIF-Complete: 最终完整版服务器

    【发论文关键特性】
    1. Ablation Switches: 支持通过参数关闭核心模块，方便做对比实验。
       - no_intra: 关闭同模态加权。
       - no_sscf: 关闭语义对齐。
    2. Auto Evaluation: 训练每10轮自动计算 Zero-shot R@1 并画 t-SNE。
    3. Mechanisms: 包含动量梯度、总量守恒加权、SSCF、逆向同步。
    """

    def __init__(self, args, writer, server_dataset, client_datasets, model_str):
        # --- 1. 数据清洗 ---
        validation_source = None
        server_holdout = {}
        if isinstance(server_dataset, (tuple, list)) and len(server_dataset) == 2:
            validation_source = server_dataset[0]
            server_holdout = server_dataset[1]
        elif isinstance(server_dataset, dict):
            validation_source = server_dataset

        fedavg_clean = validation_source.copy() if validation_source else {}
        for k in [k for k in fedavg_clean.keys() if 'fedif' in k]: del fedavg_clean[k]

        super(FedifServer, self).__init__(args, writer, (fedavg_clean, server_holdout), client_datasets, model_str)

        # --- 2. 核心参数与开关 ---
        self.validation_source = validation_source

        # [Ablation Switches] 默认开启，命令行带 --no_xxx 时关闭
        self.enable_intra = not getattr(args, 'no_intra', False)
        self.enable_sscf = not getattr(args, 'no_sscf', False)

        logger.info(
            f"[FedIF-Config] Intra-Weighting: {'ON' if self.enable_intra else 'OFF'} | SSCF: {'ON' if self.enable_sscf else 'OFF'}")

        # Intra-Modal 参数
        self.fedif_gamma = getattr(args, 'fedif_gamma', 0.8)
        self.recency_alpha = 0.5
        self.current_beta = 0.0
        self.client_scores = defaultdict(float)
        self.client_last_selected = {i: -1 for i in range(args.K)}

        # PWA 参数
        self.val_grad_buffer = {}
        self.val_momentum = 0.5
        self.align_lr = getattr(args, 'align_lr', 1e-4)
        self.align_steps = getattr(args, 'align_steps', 20)
        self.align_temp = 0.07

        # 初始化 Golden Set
        self.mm_val_loaders = []
        self.mm_dataset_name = None
        self._init_golden_loaders(args)

        # 初始化评估器
        self.fedif_evaluator = FedIfEvaluator(self)

    def _init_golden_loaders(self, args):
        if not isinstance(self.validation_source, dict): return
        golden_keys = sorted([k for k in self.validation_source.keys() if 'fedif_golden' in k])
        target_ds = next((k for k in self.global_models if DATASET_2_MODALITY.get(k) == 'img+txt'), None)
        if target_ds:
            self.mm_dataset_name = target_ds
            target_keys = golden_keys if golden_keys else [target_ds]
            for key in target_keys:
                if key in self.validation_source:
                    loader = DataLoader(self.validation_source[key], batch_size=args.B, shuffle=True,
                                        num_workers=args.num_thread, drop_last=True)
                    self.mm_val_loaders.append(loader)

    def _get_param_map(self, client_modality):
        if client_modality == 'img+txt': return lambda x: x

        def mapper(k):
            if client_modality == 'img' and 'blocks' in k and 'blockses' not in k: return k.replace('blocks',
                                                                                                    'blockses.0')
            if client_modality == 'txt' and 'blocks' in k and 'blockses' not in k: return k.replace('blocks',
                                                                                                    'blockses.1')
            return k

        return mapper

    def _calc_momentum_gradients(self):
        """计算动量梯度"""
        if not self.mm_val_loaders or not self.mm_dataset_name: return None
        loader = random.choice(self.mm_val_loaders)
        model = self.global_models[self.mm_dataset_name]
        model.to(self.server_device)
        model.eval()
        model.zero_grad()

        # 梯度累积
        accum_steps = 4
        criterion_name = TASK_2_CRITERION.get('img+txt', 'ContrastiveLoss')
        try:
            criterion = getattr(src.criterions, criterion_name)().to(self.server_device)
        except:
            criterion = nn.CrossEntropyLoss().to(self.server_device)

        iter_loader = iter(loader)
        for _ in range(accum_steps):
            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(loader); batch = next(iter_loader)

            if isinstance(batch, dict):
                inputs, targets = batch['image'].to(self.server_device), batch['text'].to(self.server_device)
            elif isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.server_device), batch[1].to(self.server_device)
            else:
                continue

            #with torch.no_grad():
            outputs = model([inputs, targets], feat_out=True)
            loss = criterion(*outputs) / accum_steps
            loss.backward()

        current_grads = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and ('attn' in name):
                    g = param.grad.detach().cpu()
                    if name in self.val_grad_buffer:
                        g = self.val_momentum * self.val_grad_buffer[name] + (1 - self.val_momentum) * g
                    self.val_grad_buffer[name] = g
                    current_grads[name] = g
        model.zero_grad();
        model.to('cpu')
        return current_grads

    def _update_intra_modal_scores(self, selected_ids, val_grads):
        """
        [FedIF-v2 Modified] 任务正交投影评分
        不再惩罚与服务器梯度正交的客户端，而是奖励其信息量（梯度幅度）。
        """
        if not val_grads: return
        base_sds = {d: {k: v.cpu() for k, v in m.state_dict().items()} for d, m in self.global_models.items()}
        groups = defaultdict(dict)

        for cid in selected_ids:
            client = self.clients[cid]
            mapper = self._get_param_map(client.modality)
            if not hasattr(self, '_client_uploads'): self._client_uploads = {}
            if cid not in self._client_uploads: self._client_uploads[cid] = dict(client.upload())
            client_sd = self._client_uploads[cid]
            base_sd = base_sds.get(client.dataset, {})

            vec_c, vec_v = [], []
            for k, w in client_sd.items():
                server_k = mapper(k)
                if server_k in val_grads and server_k in base_sd and 'attn' in server_k:
                    # 计算更新量 Delta
                    delta = (base_sd[server_k] - w.cpu()).flatten()
                    g = val_grads[server_k].flatten()

                    # 降采样以减少计算开销
                    if len(delta) > 5000:
                        idx = torch.randperm(len(delta))[:5000]
                        delta, g = delta[idx], g[idx]
                    vec_c.append(delta)
                    vec_v.append(g)

            score = 0.0
            if vec_c:
                vc, vv = torch.cat(vec_c), torch.cat(vec_v)
                cn, vn = torch.norm(vc), torch.norm(vv)

                # [关键修改] 计算基础指标
                cosine_sim = 0.0
                if cn > 1e-9 and vn > 1e-9:
                    cosine_sim = (torch.dot(vc, vv) / (cn * vn)).item()

                # 使用 Log 平滑梯度模长，作为信息量的代理
                magnitude = torch.log(1 + cn).item()

                # [关键修改] 混合评分策略 (Hybrid Scoring) [cite: 284]
                if cosine_sim > 0:
                    # 方向一致：奖励方向一致性 + 少量幅度奖励
                    score = cosine_sim * (1 + 0.1 * magnitude)
                else:
                    # 方向冲突/正交：不再归零，而是奖励其多样性 (Diversity Bonus)
                    # AG_NEWS 等单模态任务通常落入此区间
                    score = 0.1 * abs(cosine_sim) + 0.5 * magnitude

            m_key = 'mm' if ('img' in client.modality and 'txt' in client.modality) else client.modality
            groups[m_key][cid] = score

        for m_key, cid_score_map in groups.items():
            if not cid_score_map: continue
            scores = np.array(list(cid_score_map.values()))
            if len(scores) > 1:
                median = np.median(scores)
                mad = np.median(np.abs(scores - median)) + 1e-6
                for cid, raw in cid_score_map.items():
                    z = (raw - median) / mad
                    self.client_scores[cid] = self.fedif_gamma * self.client_scores[cid] + (1 - self.fedif_gamma) * z

    def _compute_total_preserving_weights(self, ids, original_sizes):
        """总量守恒加权 (Ablation: 如果禁用则返回原值)"""
        # [Ablation Switch 1]
        if not self.enable_intra or self.current_beta <= 1e-3:
            return original_sizes

        weighted_sizes = {}
        groups = defaultdict(list)
        for cid in ids:
            client = self.clients[cid]
            m_key = 'mm' if ('img' in client.modality and 'txt' in client.modality) else client.modality
            groups[m_key].append(cid)

        for m_key, group_cids in groups.items():
            total_original = sum([original_sizes[cid] for cid in group_cids])
            if total_original == 0: continue
            temp_sizes = {};
            total_new = 0.0
            for cid in group_cids:
                base = original_sizes[cid]
                z = self.client_scores[cid]
                multiplier = max(0.1, 1.0 + self.current_beta * math.tanh(z))
                new_s = base * multiplier
                temp_sizes[cid] = new_s
                total_new += new_s
            scale = total_original / (total_new + 1e-9)
            for cid in group_cids: weighted_sizes[cid] = temp_sizes[cid] * scale
        return weighted_sizes

    def _perform_sscf(self):
        """语义对齐 (Ablation: 如果禁用则不执行)"""
        # [Ablation Switch 2]
        if not self.enable_sscf: return
        if not self.mm_val_loaders or not self.mm_dataset_name: return

        model = self.global_models[self.mm_dataset_name]
        model.to(self.server_device)
        model.train()

        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'head' in name or 'proj' in name or 'norm' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': head_params, 'lr': self.align_lr},
            {'params': backbone_params, 'lr': self.align_lr * 0.1}
        ], weight_decay=1e-4)

        loader = random.choice(self.mm_val_loaders)
        iter_loader = iter(loader)

        for _ in range(self.align_steps):
            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(loader); batch = next(iter_loader)
            if isinstance(batch, dict):
                imgs, txts = batch['image'].to(self.server_device), batch['text'].to(self.server_device)
            elif isinstance(batch, (list, tuple)):
                imgs, txts = batch[0].to(self.server_device), batch[1].to(self.server_device)
            else:
                continue

            optimizer.zero_grad()
            outputs = model([imgs, txts], feat_out=True)
            loss = 0.0
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                img_emb = F.normalize(outputs[0], dim=-1)
                txt_emb = F.normalize(outputs[1], dim=-1)
                logits = torch.matmul(img_emb, txt_emb.T) / self.align_temp
                labels = torch.arange(logits.size(0), device=self.server_device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.to('cpu')
        self._sync_aligned_weights_back(model)

    def _sync_aligned_weights_back(self, mm_model):
        mm_sd = mm_model.state_dict()
        for d_name, u_model in self.global_models.items():
            modality = DATASET_2_MODALITY[d_name]
            if modality == 'img+txt': continue
            mapper = self._get_param_map(modality)
            u_sd = u_model.state_dict()
            new_sd = {}
            for u_key in u_sd.keys():
                mm_key = mapper(u_key)
                if mm_key in mm_sd and 'head' not in u_key and 'fc' not in u_key:
                    new_sd[u_key] = mm_sd[mm_key]
            u_model.load_state_dict(new_sd, strict=False)

    def _sync_aligned_weights_back(self, mm_model):
        """
        [FedIF-v2 Modified] 弹性软同步与解耦头
        1. Soft Sync: 使用 EMA 缓慢注入对齐知识，防止覆盖本地分类特征。
        2. Decoupling: 强制隔离 Transformer 的最后深层 (blockses.11)。
        """
        mm_sd = mm_model.state_dict()

        # [关键修改] 定义软同步系数 (建议 0.2) [cite: 295]
        ALPHA = 0.2

        for d_name, u_model in self.global_models.items():
            modality = DATASET_2_MODALITY[d_name]
            if modality == 'img+txt': continue

            mapper = self._get_param_map(modality)
            u_sd = u_model.state_dict()

            # 直接在参数对象上操作，避免创建新的 state_dict 导致显存激增
            with torch.no_grad():
                for u_key, u_param in u_model.named_parameters():
                    mm_key = mapper(u_key)

                    # 检查 1: 参数是否存在于多模态模型中
                    if mm_key not in mm_sd:
                        continue

                    # 检查 2: [关键修改] 解耦头架构 (Decoupled Head)
                    # 排除分类头 (head, fc) 以及 Transformer 的最后一层 (blockses.11)
                    # 注意：根据具体模型层数，这里假设是 12 层 ViT/Bert，最后一层索引为 11
                    if 'head' in u_key or 'fc' in u_key or 'blockses.11' in u_key or 'norm' in u_key:
                        continue

                    # 检查 3: 形状匹配
                    mm_param = mm_sd[mm_key]
                    if u_param.shape != mm_param.shape:
                        continue

                    # [关键修改] 弹性软同步 (Elastic Soft Sync) [cite: 297]
                    # 原地操作: u_param = (1 - ALPHA) * u_param + ALPHA * mm_param
                    u_param.data.mul_(1 - ALPHA).add_(mm_param.data, alpha=ALPHA)

    def update(self):
        """Main Update Loop"""
        # 1. 采样 (Ablation: 如果禁用 Intra，则回退到随机采样)
        if self.enable_intra:
            selected_ids = self._sample_clients()
        else:
            selected_ids = super(FedifServer, self)._sample_clients()

        self.client_uploads_cache = {}

        # 2. 训练
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)

        # 3. 评分
        if self.round > self.args.warmup_rounds:
            val_grads = self._calc_momentum_gradients()
            self._update_intra_modal_scores(selected_ids, val_grads)
            progress = (self.round - self.args.warmup_rounds) / 50.0
            self.current_beta = min(1.0, max(0.0, progress))
        else:
            self.current_beta = 0.0

        # 4. 聚合 (计算有效权重)
        effective_sizes = self._compute_total_preserving_weights(selected_ids, updated_sizes)

        for i, dataset in enumerate(self.global_models.keys()):
            self.global_model = self.global_models[dataset]
            self.task = DATASET_2_TASK[dataset]
            self.modality = DATASET_2_MODALITY[dataset]
            self.dataset = dataset
            try:
                self.out_modality_scale = self.args.out_modality_scales[i]
            except:
                self.out_modality_scale = 1.0
            self._aggregate(selected_ids, effective_sizes)
            self.global_models[dataset] = self.global_model

        # 5. Aux 同步
        if self.args.with_aux: self._sync_aux_legacy()

        # 6. SSCF (Warmup 后执行)
        if self.round > self.args.warmup_rounds:
            self._perform_sscf()

        # === [执行论文评估] ===
        # 每 10 轮跑一次 Zero-shot 和 t-SNE
        if self.round % 10 == 0:
            plot_dir = os.path.join(self.args.log_path, 'plots')
            self.evaluator.run_full_eval(self.round, plot_dir)

        # 7. Decay
        if self.round % self.args.lr_decay_step == 0:
            self.curr_lr *= self.args.lr_decay
            self.align_lr *= self.args.lr_decay

        self._empty_client_models()
        self.client_uploads_cache = {}
        return selected_ids

    def _sample_clients(self, exclude=[]):
        """Influence-based Sampling"""
        if self.round <= self.args.warmup_rounds or not self.enable_intra:
            return super(FedifServer, self)._sample_clients(exclude)

        sampled_ids = []
        for dataset in self.args.datasets:
            candidates = [c.id for c in self.clients if c.dataset == dataset and c.id not in exclude]
            if not candidates: continue
            num_sampled = max(int(self.Cs[dataset] * len(candidates)), 1)

            inf_scores = np.array([self.client_scores[uid] for uid in candidates])
            recency = np.array([self.round - self.client_last_selected[uid] for uid in candidates])

            if inf_scores.max() > inf_scores.min():
                inf_scores = (inf_scores - inf_scores.min()) / (inf_scores.max() - inf_scores.min() + 1e-9)
            else:
                inf_scores = np.zeros_like(inf_scores)

            norm_rec = np.log(1 + recency)
            if norm_rec.max() > norm_rec.min():
                norm_rec = (norm_rec - norm_rec.min()) / (norm_rec.max() - norm_rec.min() + 1e-9)

            logits = inf_scores + self.recency_alpha * norm_rec
            probs = np.exp(logits) / np.sum(np.exp(logits))
            selected = np.random.choice(candidates, size=num_sampled, replace=False, p=probs)
            sampled_ids.extend(selected)
            for uid in selected: self.client_last_selected[uid] = self.round
        return sorted(sampled_ids)

    def _sync_aux_legacy(self):
        for dataset in self.global_models.keys():
            self.global_model = self.global_models[dataset]
            modality = DATASET_2_MODALITY[dataset]
            if modality == 'img+txt': continue
            target = 'txt' if modality == 'img' else 'img'
            target_ds = next((d for d in self.global_models if DATASET_2_MODALITY[d] == target), None)
            if target_ds:
                aux_sd = self.global_models[target_ds].state_dict()
                new_sd = {}
                swap = ('blockses.0', 'blockses.1') if modality == 'img' else ('blockses.1', 'blockses.0')
                for k in self.global_model.aux_params().keys():
                    src = k.replace('aux_', '').replace(swap[0], swap[1])
                    if src in aux_sd: new_sd[k] = aux_sd[src]
                self.global_model.load_state_dict(new_sd, strict=False)