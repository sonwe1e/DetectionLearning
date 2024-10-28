import timm
import torch
import torch.nn as nn
from .RegionProposalNetwork import RegionProposalNetwork
from .RoIAlignHead import RoIAlignHead


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet50", pretrained=True):
        super(FasterRCNN, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(-2,)
        )
        self.rpn = RegionProposalNetwork(
            in_channels=self.backbone.feature_info[-2]["num_chs"]
        )
        self.num_classes = num_classes
        self.roi_head = RoIAlignHead(
            self.backbone.feature_info[-2]["num_chs"], num_classes
        )

    def forward(self, x):
        features = self.backbone(x)[-1]
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(
            features, x.shape[-2:]
        )
        cls_scores, bbox_deltas = self.roi_head(features, rois, roi_indices)
        return (
            features,
            rpn_locs,
            rpn_scores,
            rois,
            roi_indices,
            anchors,
            cls_scores,
            bbox_deltas,
        )

    def get_rpn_results(self, x):
        features = self.backbone(x)[-1]
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(
            features, x.shape[-2:]
        )
        return features, rpn_locs, rpn_scores, rois, roi_indices, anchors

    def get_roi_results(self, features, rois, roi_indices):
        cls_scores, bbox_deltas = self.roi_head(features, rois, roi_indices)
        return cls_scores, bbox_deltas


if __name__ == "__main__":
    model = FasterRCNN(num_classes=20)
    x = torch.randn(1, 3, 224, 224)
    rpn_locs, rpn_scores, cls_scores, bbox_deltas, rois, roi_indices, anchors = model(x)
    print(
        rpn_locs.shape,
        rpn_scores.shape,
        cls_scores.shape,
        bbox_deltas.shape,
        rois.shape,
        roi_indices.shape,
        anchors.shape,
    )
