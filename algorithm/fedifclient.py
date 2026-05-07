# src/client/fedifclient.py
import torch
import logging
import copy
from src.client.fedavgclient import FedavgClient
from src import TqdmToLogger, MetricManager

logger = logging.getLogger(__name__)


class FedifClient(FedavgClient):
    """
    FedIF++ 客户端：引入 EWC (Elastic Weight Consolidation) 防止灾难性遗忘。
    """

    def __init__(self, **kwargs):
        super(FedifClient, self).__init__(**kwargs)
        self.fisher_matrix = {}  # 存储服务端下发的 Fisher 信息
        self.global_params = {}  # 存储服务端下发的锚点参数 (theta_server)
        self.ewc_lambda = 0.4  # EWC 正则化系数 (可作为 args 传入)

    def download(self, models, fisher_matrix=None):
        """
        重写 download，除了下载模型参数外，还接收 Fisher 矩阵
        """
        super().download(models)

        # 保存全局模型的参数作为 EWC 的锚点
        self.global_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        # 接收 Fisher 矩阵 (如果服务端提供了)
        if fisher_matrix is not None:
            self.fisher_matrix = fisher_matrix
        else:
            self.fisher_matrix = {}

    def update(self):
        """
        重写 Update 循环，加入 EWC Loss
        -> 5.3 改进三：从“硬替换”转向“弹性权重巩固 (EWC)”
        """
        mm = MetricManager(self.eval_metrics) if self.modality != 'img+txt' else MetricManager([])
        self.model.train()
        self.model.to(self.device)

        if self.args.distributed or (self.args.mm_distributed and self.modality == 'img+txt'):
            self.model = torch.nn.DataParallel(self.model).cuda()

        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        # EWC 需要根据 modality 筛选相关的 Fisher 参数
        # 这里简化处理，假设 fisher_matrix key 与 parameter name 对应

        for e in TqdmToLogger(range(self.args.E), logger=logger, desc=f'[{self.task.upper()}] ...EWC update... '):
            for batch in self.train_loader:
                optimizer.zero_grad()

                # === 1. 计算原始任务 Loss (Base Loss) ===
                if self.modality == 'img':
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model([inputs, None])[0]
                    task_loss = self.criterion()(outputs, targets)
                elif self.modality == 'txt':
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model([None, inputs])[1]
                    task_loss = self.criterion()(outputs, targets)
                elif self.modality == 'img+txt':
                    inputs, targets, _, _, _ = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model([inputs, targets], feat_out=True)
                    task_loss = self.criterion()(*outputs)

                # === 2. 计算 EWC Regularization Loss ===
                ewc_loss = 0.0
                if self.fisher_matrix:
                    for name, param in self.model.named_parameters():
                        if name in self.fisher_matrix and name in self.global_params:
                            # L_ewc = (F/2) * (theta - theta_global)^2
                            # 注意：self.global_params 需要在同一设备上
                            anchor = self.global_params[name].to(self.device)
                            fisher = self.fisher_matrix[name].to(self.device)
                            ewc_loss += (fisher * (param - anchor) ** 2).sum()

                    ewc_loss = (self.ewc_lambda / 2) * ewc_loss

                # === 3. 总 Loss ===
                loss = task_loss + ewc_loss

                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                # 记录 (仅记录 task_loss 以便观察性能)
                if self.modality != 'img+txt':
                    mm.track(task_loss.item(), outputs, targets)
                else:
                    mm.track(task_loss.item(), outputs[0])
            else:
                mm.aggregate(len(self.training_set), e + 1)

        # 清理缓存
        self.global_params = {}
        self.model.to('cpu')
        return mm.results