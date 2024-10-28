import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def bbox_iou(bbox_a: torch.Tensor, bbox_b: torch.Tensor) -> torch.Tensor:
    """
    计算两个边界框集合之间的交并比（IoU）。

    参数:
    - bbox_a (torch.Tensor): 第一个边界框集合，形状为 (N, 4)，格式为 [x1, y1, x2, y2]。
    - bbox_b (torch.Tensor): 第二个边界框集合，形状为 (M, 4)，格式为 [x1, y1, x2, y2]。

    返回:
    - iou (torch.Tensor): 交并比矩阵，形状为 (N, M)，每个元素表示对应边界框对的 IoU。
    """
    # 计算交集的左上角和右下角坐标
    inter_top_left = torch.max(bbox_a[:, None, :2], bbox_b[:, :2])  # (N, M, 2)
    inter_bottom_right = torch.min(bbox_a[:, None, 2:], bbox_b[:, 2:])  # (N, M, 2)

    # 计算交集的宽度和高度，确保非负
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)  # (N, M, 2)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # (N, M)

    # 计算每个边界框的面积
    area_a = (bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1])  # (N,)
    area_b = (bbox_b[:, 2] - bbox_b[:, 0]) * (bbox_b[:, 3] - bbox_b[:, 1])  # (M,)

    # 计算并集面积
    union_area = area_a[:, None] + area_b - inter_area  # (N, M)

    # 计算 IoU，避免除以零
    iou = torch.where(
        union_area > 0, inter_area / union_area, torch.zeros_like(inter_area)
    )

    return iou


def loc2bbox(src_bbox: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """
    将定位偏移量转换为边界框坐标。

    参数:
    - src_bbox (torch.Tensor): 源边界框，形状为 (N, 4)，格式为 [x1, y1, x2, y2]。
    - loc (torch.Tensor): 定位偏移量，形状为 (N, 4)，格式为 [dx, dy, dw, dh]。

    返回:
    - dst_bbox (torch.Tensor): 解码后的边界框，形状为 (N, 4)。
    """
    if src_bbox.size(0) == 0:
        return torch.zeros_like(loc)

    # 计算源边界框的宽度、高度和中心坐标
    src_width = (src_bbox[:, 2] - src_bbox[:, 0]).clamp(min=1e-6)
    src_height = (src_bbox[:, 3] - src_bbox[:, 1]).clamp(min=1e-6)
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    # 获取定位偏移量
    dx = loc[:, 0]
    dy = loc[:, 1]
    dw = loc[:, 2]
    dh = loc[:, 3]

    # 计算目标边界框的中心坐标、宽度和高度
    dst_ctr_x = dx * src_width + src_ctr_x
    dst_ctr_y = dy * src_height + src_ctr_y
    dst_width = torch.exp(dw) * src_width
    dst_height = torch.exp(dh) * src_height

    # 计算目标边界框的左上角和右下角坐标
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0] = dst_ctr_x - 0.5 * dst_width  # x1
    dst_bbox[:, 1] = dst_ctr_y - 0.5 * dst_height  # y1
    dst_bbox[:, 2] = dst_ctr_x + 0.5 * dst_width  # x2
    dst_bbox[:, 3] = dst_ctr_y + 0.5 * dst_height  # y2

    return dst_bbox


def bbox2loc(src_bbox: torch.Tensor, dst_bbox: torch.Tensor) -> torch.Tensor:
    """
    将目标边界框转换为相对于源边界框的定位偏移量。

    参数:
    - src_bbox (torch.Tensor): 源边界框，形状为 (N, 4)，格式为 [x1, y1, x2, y2]。
    - dst_bbox (torch.Tensor): 目标边界框，形状为 (N, 4)，格式为 [x1, y1, x2, y2]。

    返回:
    - loc (torch.Tensor): 定位偏移量，形状为 (N, 4)，格式为 [dx, dy, dw, dh]。
    """
    if src_bbox.size() != dst_bbox.size():
        raise ValueError(
            f"src_bbox 和 dst_bbox 必须具有相同的形状，但得到的形状为 {src_bbox.shape} 和 {dst_bbox.shape}"
        )

    # 计算源边界框的宽度、高度和中心坐标
    src_width = (src_bbox[:, 2] - src_bbox[:, 0]).clamp(min=1e-6)
    src_height = (src_bbox[:, 3] - src_bbox[:, 1]).clamp(min=1e-6)
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    # 计算目标边界框的宽度、高度和中心坐标
    dst_width = (dst_bbox[:, 2] - dst_bbox[:, 0]).clamp(min=1e-6)
    dst_height = (dst_bbox[:, 3] - dst_bbox[:, 1]).clamp(min=1e-6)
    dst_ctr_x = dst_bbox[:, 0] + 0.5 * dst_width
    dst_ctr_y = dst_bbox[:, 1] + 0.5 * dst_height

    # 计算偏移量
    dx = (dst_ctr_x - src_ctr_x) / src_width
    dy = (dst_ctr_y - src_ctr_y) / src_height
    dw = torch.log(dst_width / src_width)
    dh = torch.log(dst_height / src_height)

    # 合并偏移量为 (N, 4) 的张量
    loc = torch.stack((dx, dy, dw, dh), dim=1)

    return loc


class AnchorTargetCreator:
    """
    Anchor目标创建器，用于生成训练所需的定位偏移和标签。
    """

    def __init__(
        self,
        n_sample: int = 256,
        pos_iou_thresh: float = 0.7,
        neg_iou_thresh: float = 0.3,
        pos_ratio: float = 0.5,
    ):
        """
        初始化参数。

        参数:
        - n_sample (int): 总样本数量。
        - pos_iou_thresh (float): 正样本的IoU阈值。
        - neg_iou_thresh (float): 负样本的IoU阈值。
        - pos_ratio (float): 正样本比例。
        """
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(
        self, bbox: torch.Tensor, anchor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成定位偏移和标签。

        参数:
        - bbox (torch.Tensor): 真实边界框，形状为 (num_gt, 4)。
        - anchor (torch.Tensor): 锚框，形状为 (num_anchors, 4)。

        返回:
        - loc (torch.Tensor): 定位偏移量，形状为 (num_anchors, 4)。
        - label (torch.Tensor): 标签，形状为 (num_anchors,)。
        """
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return torch.zeros_like(anchor), label

    def _calc_ious(
        self, anchor: torch.Tensor, bbox: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算anchor与bbox之间的IoU，并找到每个anchor和每个bbox的最佳匹配。

        参数:
        - anchor (torch.Tensor): 锚框，形状为 (num_anchors, 4)。
        - bbox (torch.Tensor): 真实边界框，形状为 (num_gt, 4)。

        返回:
        - argmax_ious (torch.Tensor): 每个anchor对应的最大的bbox的索引，形状为 (num_anchors,)。
        - max_ious (torch.Tensor): 每个anchor对应的最大的IoU，形状为 (num_anchors,)。
        - gt_argmax_ious (torch.Tensor): 每个bbox对应的最大的anchor的索引，形状为 (num_gt,)。
        """
        ious = bbox_iou(anchor, bbox)  # (num_anchors, num_gt)

        if bbox.numel() == 0:
            argmax_ious = torch.zeros(
                anchor.size(0), dtype=torch.int64, device=anchor.device
            )
            max_ious = torch.zeros(anchor.size(0), device=anchor.device)
            gt_argmax_ious = torch.zeros(0, dtype=torch.int64, device=anchor.device)
            return argmax_ious, max_ious, gt_argmax_ious

        argmax_ious = ious.argmax(dim=1)  # (num_anchors,)
        max_ious = ious.max(dim=1)[0]  # (num_anchors,)
        gt_argmax_ious = ious.argmax(dim=0)  # (num_gt,)

        # 确保每个真实框至少有一个对应的anchor
        argmax_ious[gt_argmax_ious] = torch.arange(
            gt_argmax_ious.size(0), device=anchor.device
        )

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(
        self, anchor: torch.Tensor, bbox: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据IoU生成标签，并进行正负样本的平衡。

        参数:
        - anchor (torch.Tensor): 锚框，形状为 (num_anchors, 4)。
        - bbox (torch.Tensor): 真实边界框，形状为 (num_gt, 4)。

        返回:
        - argmax_ious (torch.Tensor): 每个anchor对应的最大的bbox的索引，形状为 (num_anchors,)。
        - label (torch.Tensor): 标签，形状为 (num_anchors,)。
        """
        label = torch.full(
            (anchor.size(0),), -1, dtype=torch.int64, device=anchor.device
        )

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        # 标记负样本
        label[max_ious < self.neg_iou_thresh] = 0
        # 标记正样本
        label[max_ious >= self.pos_iou_thresh] = 1
        if gt_argmax_ious.numel() > 0:
            label[gt_argmax_ious] = 1

        # 限制正样本数量
        label = self._limit_positive_samples(label)

        # 平衡负样本数量
        label = self._balance_negative_samples(label)

        return argmax_ious, label

    def _limit_positive_samples(self, label: torch.Tensor) -> torch.Tensor:
        """
        限制正样本数量不超过指定比例。

        参数:
        - label (torch.Tensor): 当前标签数组，形状为 (num_anchors,)。

        返回:
        - label (torch.Tensor): 更新后的标签数组。
        """
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_indices = torch.nonzero(label == 1, as_tuple=False).squeeze(1)
        if pos_indices.numel() > n_pos:
            disable_indices = pos_indices[
                torch.randperm(pos_indices.numel())[: pos_indices.numel() - n_pos]
            ]
            label[disable_indices] = -1
        return label

    def _balance_negative_samples(self, label: torch.Tensor) -> torch.Tensor:
        """
        平衡负样本数量，使总样本数量保持为n_sample。

        参数:
        - label (torch.Tensor): 当前标签数组，形状为 (num_anchors,)。

        返回:
        - label (torch.Tensor): 更新后的标签数组。
        """
        n_neg = self.n_sample - (label == 1).sum().item()
        neg_indices = torch.nonzero(label == 0, as_tuple=False).squeeze(1)
        if neg_indices.numel() > n_neg:
            disable_indices = neg_indices[
                torch.randperm(neg_indices.numel())[: neg_indices.numel() - n_neg]
            ]
            label[disable_indices] = -1
        return label


class ProposalTargetCreator:
    """
    Proposal目标创建器，用于生成第二阶段（如Fast R-CNN）的训练所需的定位偏移和标签。
    """

    def __init__(
        self,
        n_sample: int = 128,
        pos_ratio: float = 0.5,
        pos_iou_thresh: float = 0.5,
        neg_iou_thresh_high: float = 0.5,
        neg_iou_thresh_low: float = 0.0,
        loc_normalize_std: Tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
    ):
        """
        初始化参数。

        参数:
        - n_sample (int): 总样本数量。
        - pos_ratio (float): 正样本比例。
        - pos_iou_thresh (float): 正样本的IoU下限阈值。
        - neg_iou_thresh_high (float): 负样本的IoU上限阈值。
        - neg_iou_thresh_low (float): 负样本的IoU下限阈值。
        - loc_normalize_std (Tuple[float, float, float, float]): 定位偏移量的标准化参数。
        """
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = int(round(self.n_sample * self.pos_ratio))
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low
        self.loc_normalize_std = torch.tensor(loc_normalize_std)

    def __call__(
        self,
        roi: torch.Tensor,
        bbox: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成第二阶段的定位偏移和标签。

        参数:
        - roi (torch.Tensor): 第一阶段生成的建议框，形状为 (num_roi, 4)。
        - bbox (torch.Tensor): 真实边界框，形状为 (num_gt, 4)。
        - label (torch.Tensor): 真实边界框的标签，形状为 (num_gt,)。

        返回:
        - loc_normalized (torch.Tensor): 标准化后的定位偏移量，形状为 (n_sample, 4)。
        - labels (torch.Tensor): 标签，形状为 (n_sample,)。
        - sampled_roi (torch.Tensor): 采样后的建议框，形状为 (n_sample, 4)。
        """
        # 将建议框与真实框合并
        combined_roi = torch.cat([roi, bbox], dim=0)  # (num_roi + num_gt, 4)

        # 计算IoU
        iou = bbox_iou(combined_roi, bbox)  # (num_combined_roi, num_gt)

        if bbox.numel() == 0:
            gt_assignment = torch.zeros(
                combined_roi.size(0), dtype=torch.int64, device=roi.device
            )
            max_iou = torch.zeros(combined_roi.size(0), device=roi.device)
            gt_roi_label = torch.zeros(
                combined_roi.size(0), dtype=torch.int64, device=roi.device
            )
        else:
            gt_assignment = iou.argmax(dim=1)  # (num_combined_roi,)
            max_iou, _ = iou.max(dim=1)  # (num_combined_roi,)
            gt_roi_label = label[gt_assignment] + 1  # (num_combined_roi,)

            # 标记负样本
            gt_roi_label[max_iou < self.neg_iou_thresh_low] = 0
            # 标记正样本
            gt_roi_label[max_iou >= self.pos_iou_thresh] = 1

        # 采样正负样本
        pos_indices = torch.nonzero(gt_roi_label == 1, as_tuple=False).squeeze(1)
        neg_indices = torch.nonzero(gt_roi_label == 0, as_tuple=False).squeeze(1)

        # 采样正样本
        num_pos = min(self.pos_roi_per_image, pos_indices.numel())
        if num_pos > 0:
            perm = torch.randperm(pos_indices.numel(), device=roi.device)
            pos_indices = pos_indices[perm[:num_pos]]
        else:
            pos_indices = torch.tensor([], dtype=torch.int64, device=roi.device)

        # 采样负样本
        num_neg = self.n_sample - num_pos
        num_neg = min(num_neg, neg_indices.numel())
        if num_neg > 0:
            perm = torch.randperm(neg_indices.numel(), device=roi.device)
            neg_indices = neg_indices[perm[:num_neg]]
        else:
            neg_indices = torch.tensor([], dtype=torch.int64, device=roi.device)

        # 合并正负样本索引
        sampled_indices = torch.cat([pos_indices, neg_indices], dim=0)

        # 采样后的建议框和标签
        sampled_roi = combined_roi[sampled_indices]
        labels = gt_roi_label[sampled_indices]

        # 计算定位偏移量
        if pos_indices.numel() > 0:
            assigned_gt = gt_assignment[sampled_indices[:num_pos]]
            assigned_bbox = bbox[assigned_gt]
            loc = bbox2loc(sampled_roi[:num_pos], assigned_bbox)
            loc_normalized = loc / self.loc_normalize_std.to(loc.device)
            # 对于负样本，定位偏移量设为0
            if num_neg > 0:
                loc_normalized = torch.cat(
                    [loc_normalized, torch.zeros(num_neg, 4, device=loc.device)], dim=0
                )
        else:
            loc_normalized = torch.zeros_like(sampled_roi)

        return loc_normalized, labels, sampled_roi


if __name__ == "__main__":
    # 示例锚框和真实框
    anchors = torch.tensor(
        [[10, 10, 20, 20], [15, 15, 25, 25], [20, 20, 30, 30], [30, 30, 40, 40]],
        dtype=torch.float32,
    )

    bboxes = torch.tensor([[12, 12, 22, 22], [28, 28, 38, 38]], dtype=torch.float32)

    labels = torch.tensor([1, 2], dtype=torch.int64)  # 假设类别标签为1和2

    # 创建AnchorTargetCreator实例
    anchor_target_creator = AnchorTargetCreator()

    # 生成定位偏移和标签
    loc, anchor_labels = anchor_target_creator(bboxes, anchors)
    print("Anchor Targets (loc):\n", loc)
    print("Anchor Labels:\n", anchor_labels)

    # 创建ProposalTargetCreator实例
    proposal_target_creator = ProposalTargetCreator()

    # 假设第一阶段生成的建议框
    rois = torch.tensor(
        [[11, 11, 21, 21], [16, 16, 26, 26], [25, 25, 35, 35], [35, 35, 45, 45]],
        dtype=torch.float32,
    )

    # 生成第二阶段的定位偏移和标签
    loc_normalized, proposal_labels, sampled_rois = proposal_target_creator(
        rois, bboxes, labels
    )
    print("Proposal Targets (loc_normalized):\n", loc_normalized)
    print("Proposal Labels:\n", proposal_labels)
    print("Sampled ROIs:\n", sampled_rois)
