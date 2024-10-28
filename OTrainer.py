import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Faster R-CNN 训练器

        参数：
        - model_train (nn.Module): Faster R-CNN 模型，用于训练模式。
        - optimizer (torch.optim.Optimizer): 优化器。
        """
        super(FasterRCNNTrainer, self).__init__()
        self.model_train = model_train
        self.optimizer = optimizer

        # 超参数
        self.rpn_sigma = 1
        self.roi_sigma = 1
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

        # 目标创建器
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

    def _fast_rcnn_loc_loss(
        self,
        pred_loc: torch.Tensor,
        gt_loc: torch.Tensor,
        gt_label: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        计算 Fast R-CNN 的定位损失

        参数：
        - pred_loc (torch.Tensor): 预测的偏移量，形状为 [num_pos, 4]
        - gt_loc (torch.Tensor): 真实的偏移量，形状为 [num_pos, 4]
        - gt_label (torch.Tensor): 真实的标签，形状为 [num_pos]
        - sigma (float): 超参数

        返回：
        - regression_loss (torch.Tensor): 定位损失标量
        """
        # 只计算正样本的损失
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma**2
        regression_diff = (gt_loc - pred_loc).abs().float()

        regression_loss = torch.where(
            regression_diff < (1.0 / sigma_squared),
            0.5 * sigma_squared * regression_diff**2,
            regression_diff - 0.5 / sigma_squared,
        ).sum()

        num_pos = (gt_label > 0).float().sum()
        regression_loss /= torch.max(
            num_pos, torch.tensor(1.0, device=regression_loss.device)
        )

        return regression_loss

    def forward(
        self,
        imgs: torch.Tensor,
        bboxes: List[np.ndarray],
        labels: List[np.ndarray],
        scale: float,
    ) -> List[float]:
        """
        前向传播并计算损失

        参数：
        - imgs (torch.Tensor): 输入图像，形状为 [batch_size, 3, H, W]
        - bboxes (List[np.ndarray]): 每张图像的真实框列表，长度为 batch_size
        - labels (List[np.ndarray]): 每张图像的真实标签列表，长度为 batch_size
        - scale (float): 图像缩放因子

        返回：
        - losses (List[float]): [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss]
        """
        batch_size = imgs.shape[0]
        img_size = imgs.shape[2:]  # (H, W)

        # 提取基础特征
        base_feature = self.model_train(imgs, mode="extractor")

        # 获取 RPN 输出
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(
            [base_feature, img_size], scale=scale, mode="rpn"
        )

        # 确保 anchor 在 CPU 并转换为 numpy 数组
        anchor_np = anchor.cpu().numpy()

        # 初始化损失累加器和 Fast R-CNN 目标收集器
        rpn_loc_loss_all = 0.0
        rpn_cls_loss_all = 0.0
        roi_loc_loss_all = 0.0
        roi_cls_loss_all = 0.0

        sample_rois = []
        sample_indices = []
        gt_roi_locs = []
        gt_roi_labels = []

        for i in range(batch_size):
            # 获取当前图像的 RPN 输出和真实框
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[i]

            # 获取 RPN 目标
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor_np)

            gt_rpn_loc = torch.from_numpy(gt_rpn_loc).to(rpn_locs.device).float()
            gt_rpn_label = torch.from_numpy(gt_rpn_label).to(rpn_locs.device).long()

            # 计算 RPN 损失
            rpn_loc_loss = self._fast_rcnn_loc_loss(
                rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma
            )
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss

            # 获取 Fast R-CNN 目标
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi, bbox, label, self.loc_normalize_std
            )

            sample_rois.append(torch.from_numpy(sample_roi).to(rpn_locs.device).float())
            sample_indices.append(
                torch.full(
                    (sample_roi.shape[0],),
                    i,
                    dtype=torch.float32,
                    device=rpn_locs.device,
                )
            )
            gt_roi_locs.append(torch.from_numpy(gt_roi_loc).to(rpn_locs.device).float())
            gt_roi_labels.append(
                torch.from_numpy(gt_roi_label).to(rpn_locs.device).long()
            )

        # 合并所有样本的 RoIs 和索引
        if sample_rois:
            sample_rois = torch.cat(sample_rois, dim=0)  # [num_samples, 4]
            sample_indices = torch.cat(sample_indices, dim=0)  # [num_samples]
            gt_roi_locs = torch.cat(gt_roi_locs, dim=0)  # [num_samples, 4]
            gt_roi_labels = torch.cat(gt_roi_labels, dim=0)  # [num_samples]

            # 获取 Fast R-CNN 的预测
            roi_cls_locs, roi_scores = self.model_train(
                [base_feature, sample_rois, sample_indices, img_size], mode="head"
            )

            # 计算 Fast R-CNN 损失
            if roi_cls_locs is not None and roi_scores is not None:
                # Reshape roi_cls_locs to [num_samples, num_classes, 4]
                roi_cls_locs = roi_cls_locs.view(
                    -1, self.model_train.head.num_classes, 4
                )

                # 根据 gt_roi_labels 选择对应的回归参数
                # Assuming gt_roi_labels start from 0 (background) to num_classes-1
                # Adjust if labels start from 1
                roi_loc = roi_cls_locs[
                    torch.arange(roi_cls_locs.size(0), device=roi_cls_locs.device),
                    gt_roi_labels,
                ]

                # 计算定位损失
                roi_loc_loss = self._fast_rcnn_loc_loss(
                    roi_loc, gt_roi_locs, gt_roi_labels, self.roi_sigma
                )
                # 计算分类损失
                roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels)

                roi_loc_loss_all += roi_loc_loss
                roi_cls_loss_all += roi_cls_loss

        # 平均损失
        losses = [
            rpn_loc_loss_all / batch_size,
            rpn_cls_loss_all / batch_size,
            roi_loc_loss_all / batch_size,
            roi_cls_loss_all / batch_size,
        ]
        total_loss = sum(losses)
        losses.append(total_loss.item())

        return losses

    def train_step(
        self,
        imgs: torch.Tensor,
        bboxes: List[np.ndarray],
        labels: List[np.ndarray],
        scale: float,
        fp16: bool = False,
        scaler=None,
    ) -> List[float]:
        """
        训练步骤

        参数：
        - imgs (torch.Tensor): 输入图像，形状为 [batch_size, 3, H, W]
        - bboxes (List[np.ndarray]): 每张图像的真实框列表，长度为 batch_size
        - labels (List[np.ndarray]): 每张图像的真实标签列表，长度为 batch_size
        - scale (float): 图像缩放因子
        - fp16 (bool): 是否使用半精度训练
        - scaler: GradScaler 对象（用于混合精度训练）

        返回：
        - losses (List[float]): [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss]
        """
        self.optimizer.zero_grad()

        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale)
            losses[-1].backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)
            # 反向传播
            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()

        return losses
