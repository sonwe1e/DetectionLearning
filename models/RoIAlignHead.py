import torch.nn as nn
import torch
from torchvision.ops import RoIAlign
from typing import Tuple


class RoIAlignHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        output_size: int = 7,
        spatial_scale: float = 0.0675,
    ):
        """
        Faster R-CNN 头部，用于分类和回归任务。

        参数：
        - in_channels (int): RoIAlign 输出特征的通道数。
        - num_classes (int): 目标类别的数量（不包括背景类）。
        """
        super(RoIAlignHead, self).__init__()
        self.num_classes = num_classes

        # RoIAlign 层，用于提取固定大小的特征
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),  # 输出特征的空间尺寸
            spatial_scale=spatial_scale,  # 特征图与原图的比例
            sampling_ratio=-1,  # 采样点的数量，-1 表示自动计算
            aligned=True,
        )

        self.cls_head = nn.Linear(in_channels * output_size * output_size, num_classes)
        self.reg_head = nn.Linear(
            in_channels * output_size * output_size, num_classes * 4
        )

    def forward(
        self,
        features: torch.Tensor,
        rois: torch.Tensor,
        roi_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数：
        - features (torch.Tensor): 输入特征图，形状为 [batch_size, in_channels, height, width]。
        - rois (torch.Tensor): 建议框，形状为 [batch_size, num_rois, 4]，格式为 [x1, y1, x2, y2]。
        - roi_indices (torch.Tensor): RoI 对应的批次索引，形状为 [batch_size * num_rois, 1]。
        - img_size (Tuple[int, int]): 图像尺寸，格式为 (height, width)。

        返回：
        - cls_scores (torch.Tensor): 分类得分，形状为 [num_rois_total, num_classes]。
        - bbox_deltas (torch.Tensor): 边界框调整参数，形状为 [num_rois_total, num_classes * 4]。
        """

        # 将 rois 从 [batch_size, num_rois, 4] 转换为 [total_rois, 4]
        rois = rois.view(-1, 4)  # [total_rois, 4]
        roi_indices = roi_indices.view(-1, 1)  # [total_rois, 1]
        rois = torch.cat([roi_indices.float(), rois], dim=1)  # [total_rois, 5]

        # 使用 RoIAlign 提取固定大小的特征 [total_rois, in_channels, 7, 7]
        aligned_features = self.roi_align(features, rois)

        # 展平特征 [total_rois, in_channels * 7 * 7]
        flattened_features = aligned_features.view(aligned_features.size(0), -1)

        cls_scores = self.cls_head(flattened_features)  # [total_rois, num_classes]
        bbox_deltas = self.reg_head(flattened_features)  # [total_rois, num_classes * 4]

        return cls_scores, bbox_deltas


if __name__ == "__main__":
    head = RoIAlignHead(256, 20)
    features = torch.randn(2, 256, 50, 50)
    rois = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32)
    roi_indices = torch.tensor([[0], [0]], dtype=torch.float32)
    print(features.shape, rois.shape, roi_indices.shape)
    cls_scores, bbox_deltas = head(features, rois, roi_indices)
    print(cls_scores.shape, bbox_deltas.shape)
    # torch.Size([2, 20]) torch.Size([2, 80])
