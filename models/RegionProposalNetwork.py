import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np
from typing import List, Tuple


def generate_anchor_base(
    base_size: int = 16,
    ratios: List[float] = [0.5, 1, 2],
    anchor_scales: List[int] = [8, 16, 32],
) -> np.ndarray:
    """
    生成基础先验框（anchor base）。

    参数：
    - base_size (int): 基础尺寸。默认为 16。
    - ratios (List[float]): 先验框的宽高比列表。默认为 [0.5, 1, 2]。
    - anchor_scales (List[int]): 先验框的尺度列表。默认为 [8, 16, 32]。

    返回：
    - anchor_base (np.ndarray): 生成的基础先验框，形状为 [num_anchors, 4]。
    """
    num_ratios = len(ratios)
    num_scales = len(anchor_scales)
    anchor_base = np.zeros((num_ratios * num_scales, 4), dtype=np.float32)

    for i, ratio in enumerate(ratios):
        for j, scale in enumerate(anchor_scales):
            h = base_size * scale * np.sqrt(ratio)
            w = base_size * scale * np.sqrt(1.0 / ratio)

            index = i * num_scales + j
            anchor_base[index] = [-h / 2.0, -w / 2.0, h / 2.0, w / 2.0]

    return anchor_base


def loc2bbox(src_bbox: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """
    将回归预测转换为边界框。

    参数：
    - src_bbox (torch.Tensor): 源先验框，形状为 [num_anchors, 4]。
    - loc (torch.Tensor): 回归预测，形状为 [num_anchors, 4]。

    返回：
    - dst_bbox (torch.Tensor): 调整后的边界框，形状为 [num_anchors, 4]。
    """
    if src_bbox.size == 0:
        return torch.zeros((0, 4), dtype=loc.dtype, device=loc.device)

    # 计算源先验框的宽度、高度和中心点坐标
    src_width = (src_bbox[:, 2] - src_bbox[:, 0]).unsqueeze(1)  # [num_anchors, 1]
    src_height = (src_bbox[:, 3] - src_bbox[:, 1]).unsqueeze(1)  # [num_anchors, 1]
    src_ctr_x = src_bbox[:, 0].unsqueeze(1) + 0.5 * src_width  # [num_anchors, 1]
    src_ctr_y = src_bbox[:, 1].unsqueeze(1) + 0.5 * src_height  # [num_anchors, 1]

    # 分别提取 loc 中的偏移量
    dx, dy, dw, dh = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]

    # 计算调整后的中心点坐标
    ctr_x = dx * src_width.squeeze(1) + src_ctr_x.squeeze(1)
    ctr_y = dy * src_height.squeeze(1) + src_ctr_y.squeeze(1)

    # 计算调整后的宽度和高度
    w = torch.exp(dw) * src_width.squeeze(1)
    h = torch.exp(dh) * src_height.squeeze(1)

    # 计算调整后的边界框坐标
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0] = ctr_y - 0.5 * h  # y1
    dst_bbox[:, 1] = ctr_x - 0.5 * w  # x1
    dst_bbox[:, 2] = ctr_y + 0.5 * h  # y2
    dst_bbox[:, 3] = ctr_x + 0.5 * w  # x2

    return dst_bbox


def _enumerate_shifted_anchor(
    anchor_base: np.ndarray, feat_stride: int, height: int, width: int
) -> np.ndarray:
    """
    生成所有偏移后的先验框。

    参数：
    - anchor_base (np.ndarray): 基础先验框，形状为 [num_anchors, 4]。
    - feat_stride (int): 特征图的步长。
    - height (int): 特征图的高度。
    - width (int): 特征图的宽度。

    返回：
    - shifted_anchors (np.ndarray): 生成的先验框，形状为 [height * width * num_anchors, 4]。
    """
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()
    A = anchor_base.shape[0]
    K = shifts.shape[0]
    anchors = anchor_base.reshape((1, A, 4)) + shifts.reshape((K, 1, 4))
    anchors = anchors.reshape((K * A, 4))
    return anchors


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        mid_channels: int = 512,
        ratios: List[float] = [0.5, 1, 2],
        anchor_scales: List[int] = [8, 16, 32],
        feat_stride: int = 16,
        mode: str = "training",
        nms_iou: float = 0.7,
        n_train_pre_nms: int = 12000,
        n_train_post_nms: int = 600,
        n_test_pre_nms: int = 3000,
        n_test_post_nms: int = 300,
        min_size: int = 16,
    ):
        """
        区域提议网络 (RPN)

        参数：
        - in_channels (int): 输入特征图的通道数。
        - mid_channels (int): 中间卷积层的通道数。
        - ratios (List[float]): 先验框的宽高比列表。
        - anchor_scales (List[int]): 先验框的尺度列表。
        - feat_stride (int): 特征图的步长。
        - mode (str): 模式，"training" 或 "evaluation"。
        - nms_iou (float): 非极大抑制的 IoU 阈值。
        - n_train_pre_nms (int): 训练时非极大抑制前的建议框数量。
        - n_train_post_nms (int): 训练时非极大抑制后的建议框数量。
        - n_test_pre_nms (int): 测试时非极大抑制前的建议框数量。
        - n_test_post_nms (int): 测试时非极大抑制后的建议框数量。
        - min_size (int): 建议框的最小尺寸。
        """
        super(RegionProposalNetwork, self).__init__()

        # 生成基础先验框，形状为 [num_anchors, 4]
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios
        )
        self.num_anchors = self.anchor_base.shape[0]

        # 特征整合卷积层
        self.conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.score_conv = nn.Conv2d(
            mid_channels, self.num_anchors * 2, kernel_size=1, stride=1, padding=0
        )
        self.loc_conv = nn.Conv2d(
            mid_channels, self.num_anchors * 4, kernel_size=1, stride=1, padding=0
        )

        self.feat_stride = feat_stride
        self.mode = mode

        # RPN参数
        self.nms_iou = nms_iou
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def _filter_proposals(
        self,
        rois: torch.Tensor,
        scores: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        过滤和应用非极大抑制（NMS）来生成最终的建议框。

        参数：
        - rois (torch.Tensor): 调整后的候选框，形状为 [num_proposals, 4]。
        - scores (torch.Tensor): 每个候选框的分数，形状为 [num_proposals]。
        - img_size (Tuple[int, int]): 图像尺寸，格式为 (height, width)。

        返回：
        - final_rois (torch.Tensor): 经过过滤和NMS后的最终建议框，形状为 [n_post_nms, 4]。
        """
        # 防止建议框超出图像边缘
        rois[:, 0].clamp_(min=0, max=img_size[0])
        rois[:, 1].clamp_(min=0, max=img_size[1])
        rois[:, 2].clamp_(min=0, max=img_size[0])
        rois[:, 3].clamp_(min=0, max=img_size[1])

        # 过滤掉小于最小尺寸的建议框
        min_size_scaled = self.min_size
        keep = (rois[:, 2] - rois[:, 0] >= min_size_scaled) & (
            rois[:, 3] - rois[:, 1] >= min_size_scaled
        )
        rois, scores = rois[keep], scores[keep]

        # 根据得分排序并选择前 n_pre_nms 个建议框
        num_pre_nms = (
            self.n_train_pre_nms if self.mode == "training" else self.n_test_pre_nms
        )
        num_post_nms = (
            self.n_train_post_nms if self.mode == "training" else self.n_test_post_nms
        )

        if num_pre_nms > 0:
            scores, order = scores.topk(
                k=min(num_pre_nms, scores.size(0)), largest=True, sorted=True
            )
            rois = rois[order]

        # 非极大抑制
        keep = nms(rois, scores, self.nms_iou)
        keep = keep[:num_post_nms]

        # 如果保留的框不足，则进行补充
        if len(keep) < num_post_nms:
            extra = num_post_nms - len(keep)
            keep = torch.cat(
                [keep, keep.new_full((extra,), keep[-1].item(), dtype=torch.long)]
            )

        final_rois = rois[keep]
        return final_rois

    def forward(
        self, x: torch.Tensor, img_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数：
        - x (torch.Tensor): 输入特征图，形状为 [batch_size, in_channels, height, width]。
        - img_size (Tuple[int, int]): 图像尺寸，格式为 (height, width)。
        - scale (float): 缩放因子。

        返回：
        - rpn_locs (torch.Tensor): 回归预测，形状为 [batch_size, num_anchors, 4]。
        - rpn_scores (torch.Tensor): 分类预测，形状为 [batch_size, num_anchors, 2]。
        - rois (torch.Tensor): 建议框，形状为 [batch_size, n_post_nms, 4]。
        - roi_indices (torch.Tensor): 建议框对应的批次索引，形状为 [batch_size, n_post_nms]。
        - anchors (torch.Tensor): 生成的先验框，形状为 [1, num_anchors, 4]。
        """
        batch_size, _, height, width = x.shape

        # 特征整合
        feature = F.relu(self.conv(x))

        # 回归预测
        rpn_locs = self.loc_conv(feature)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        # 分类预测
        rpn_scores = self.score_conv(feature)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        # 获取包含物体的概率
        rpn_probs = F.softmax(rpn_scores, dim=-1)[:, :, 1]  # [batch_size, num_anchors]

        # 生成先验框
        anchors = _enumerate_shifted_anchor(
            self.anchor_base, self.feat_stride, height, width
        )  # [num_anchors, 4]
        anchors = torch.from_numpy(anchors).float().to(x.device)

        rois = []
        roi_indices = []

        for i in range(batch_size):
            # 将回归预测应用于先验框，生成候选框
            bbox = loc2bbox(anchors, rpn_locs[i])

            # 过滤和应用NMS
            final_rois = self._filter_proposals(bbox, rpn_probs[i], img_size)

            rois.append(final_rois.unsqueeze(0))
            roi_indices.append(
                torch.full(
                    (final_rois.size(0),), i, dtype=torch.float32, device=x.device
                ).unsqueeze(0)
            )

        rois = torch.cat(rois, dim=0)  # [batch_size, n_post_nms, 4]
        roi_indices = torch.cat(roi_indices, dim=0)  # [batch_size, n_post_nms, 1]

        return rpn_locs, rpn_scores, rois, roi_indices, anchors


if __name__ == "__main__":
    # 实例化 RPN
    rpn = RegionProposalNetwork(
        in_channels=512,
        mid_channels=512,
        ratios=[0.5, 1, 2],
        anchor_scales=[8, 16, 32],
        feat_stride=16,
        mode="training",
        nms_iou=0.7,
        n_train_pre_nms=12000,
        n_train_post_nms=600,
        n_test_pre_nms=3000,
        n_test_post_nms=300,
        min_size=16,
    )

    # 输入特征图（示例）
    x = torch.randn(2, 512, 38, 50)  # 示例特征图
    img_size = (600, 800)  # 示例图像尺寸
    scale = 1.0

    # 前向传播
    rpn_locs, rpn_scores, rois, roi_indices, anchors = rpn(x, img_size)

    print("RPN Locs Shape:", rpn_locs.shape)
    print("RPN Scores Shape:", rpn_scores.shape)
    print("ROIs Shape:", rois.shape)
    print("ROI Indices Shape:", roi_indices.shape)
    print("Anchors Shape:", anchors.shape)
