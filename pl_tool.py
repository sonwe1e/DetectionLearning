import torch
import torch.nn as nn
import lightning.pytorch as pl
from utils import AnchorTargetCreator, ProposalTargetCreator
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            betas=(0.9, 0.95),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.06,
            steps_per_epoch=self.len_trainloader,
            anneal_strategy="linear",
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        batch_size = images.shape[0]
        feature, rpn_locs, rpn_scores, rois, roi_indices, anchors = (
            self.model.get_rpn_results(images)
        )
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
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchors)

            # 计算 RPN 损失
            rpn_loc_loss = self._fast_rcnn_loc_loss(
                rpn_loc, gt_rpn_loc, gt_rpn_label, self.opt.rpn_sigma
            )
            rpn_cls_loss = self.ce_loss(rpn_score, gt_rpn_label)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss

            # 获取 Fast R-CNN 目标
            gt_roi_loc, gt_roi_label, sample_roi = self.proposal_target_creator(
                roi, bbox, label
            )

            sample_rois.append(sample_roi)
            sample_indices.append(
                torch.full(
                    (sample_roi.shape[0],),
                    i,
                    dtype=torch.float32,
                    device=rpn_locs.device,
                )
            )
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_labels.append(gt_roi_label.long())

        # 合并所有样本的 RoIs 和索引
        if sample_rois:
            sample_rois = torch.cat(sample_rois, dim=0)  # [num_samples, 4]
            sample_indices = torch.cat(sample_indices, dim=0)  # [num_samples]
            gt_roi_locs = torch.cat(gt_roi_locs, dim=0)  # [num_samples, 4]
            gt_roi_labels = torch.cat(gt_roi_labels, dim=0)  # [num_samples]

            # 获取 Fast R-CNN 的预测
            cls_scores, bbox_deltas = self.model.get_roi_results(
                feature, sample_rois, sample_indices
            )
            # 计算 Fast R-CNN 损失
            if bbox_deltas is not None and cls_scores is not None:
                # Reshape roi_cls_locs to [num_samples, num_classes, 4]
                bbox_deltas = bbox_deltas.view(-1, self.opt.num_classes, 4)

                roi_loc = bbox_deltas[
                    torch.arange(bbox_deltas.size(0), device=bbox_deltas.device),
                    gt_roi_labels,
                ]

                # 计算定位损失
                roi_loc_loss = self._fast_rcnn_loc_loss(
                    roi_loc, gt_roi_locs, gt_roi_labels, self.opt.roi_sigma
                )
                # 计算分类损失
                roi_cls_loss = self.ce_loss(cls_scores, gt_roi_labels)

                roi_loc_loss_all += roi_loc_loss
                roi_cls_loss_all += roi_cls_loss

        # 平均损失
        loss = (
            (rpn_loc_loss_all + rpn_cls_loss_all + roi_loc_loss_all + roi_cls_loss_all)
            / batch_size
            / 2
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

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
