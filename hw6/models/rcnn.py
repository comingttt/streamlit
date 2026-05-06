"""
模块2: R-CNN系列目标检测模型
包含: R-CNN (简化版), Fast R-CNN (简化版), Faster R-CNN (torchvision预训练)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.ops import nms, roi_align, roi_pool
from typing import List, Dict, Tuple, Optional
import numpy as np


# ==================== R-CNN (简化演示版) ====================

class SimpleRCNN(nn.Module):
    """
    简化版R-CNN演示模型
    核心思想：
    1. 使用Selective Search生成候选区域（region proposals）
    2. 对每个区域用CNN提取特征
    3. 用SVM/分类器对每个区域分类
    4. NMS去除重复框

    简化实现：使用预训练ResNet提取特征 + 线性分类器
    """

    def __init__(self, num_classes: int = 21, pretrained_backbone: bool = True):
        super(SimpleRCNN, self).__init__()
        self.num_classes = num_classes

        # 使用ResNet50作为特征提取器
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        # 去掉最后的全连接层和池化层
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器：对每个proposal分类
        self.classifier = nn.Linear(2048, num_classes)

        # 边界框回归器：微调proposal位置
        self.bbox_regressor = nn.Linear(2048, num_classes * 4)

    def forward(self, x: torch.Tensor, proposals: List[torch.Tensor] = None):
        """
        Args:
            x: (B, 3, H, W) 输入图像
            proposals: 候选区域列表
        Returns:
            分类分数和边界框偏移
        """
        # 提取特征
        features = self.backbone(x)  # (B, 2048, H/32, W/32)

        # 如果没有proposals，返回特征图
        if proposals is None:
            return features

        return features, proposals


# ==================== Fast R-CNN (简化演示版) ====================

class SimpleFastRCNN(nn.Module):
    """
    简化版Fast R-CNN
    核心改进（相比R-CNN）：
    1. 整张图只过一次CNN，共享特征
    2. 使用RoI Pooling提取固定大小的区域特征
    3. 分类和回归使用全连接网络
    """

    def __init__(self, num_classes: int = 21, pretrained_backbone: bool = True):
        super(SimpleFastRCNN, self).__init__()
        self.num_classes = num_classes

        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # RoI Pooling: 将不同大小的proposal映射到固定尺寸
        self.roi_output_size = (7, 7)
        self.spatial_scale = 1.0 / 32.0  # 特征图下采样比例

        # 分类头和回归头
        self.fc1 = nn.Linear(2048 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        images: torch.Tensor,
        proposals: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W)
            proposals: List of (N_i, 4) tensors [x1, y1, x2, y2]
        Returns:
            cls_scores: 分类分数
            bbox_preds: 边界框偏移
        """
        # 共享特征提取
        features = self.backbone(images)

        # RoI Pooling
        roi_features = roi_pool(
            features, proposals, self.roi_output_size, self.spatial_scale
        )  # (total_rois, 2048, 7, 7)

        # 展平并通过全连接层
        x = roi_features.view(roi_features.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)

        return cls_scores, bbox_preds


# ==================== Faster R-CNN ====================

def get_faster_rcnn(
    num_classes: int = 91,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
) -> nn.Module:
    """
    获取Faster R-CNN模型（基于torchvision预训练）

    Faster R-CNN核心创新（相比Fast R-CNN）：
    1. 引入RPN（Region Proposal Network）
    2. RPN与检测网络共享卷积特征
    3. 实现端到端训练，无需外部proposal方法

    Args:
        num_classes: 类别数（含背景）
        pretrained: 是否使用COCO预训练权重
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes
        )
    return model


# ==================== 辅助函数 ====================

import torch.nn.functional as F


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: int = 100
) -> torch.Tensor:
    """应用NMS过滤重复框"""
    keep = nms(boxes, scores, iou_threshold)
    if len(keep) > max_detections:
        keep = keep[:max_detections]
    return keep


def generate_proposals_selective_search(
    image: np.ndarray, max_proposals: int = 200
) -> np.ndarray:
    """
    使用OpenCV的Selective Search生成候选区域
    这是R-CNN中使用的候选区域生成方法
    """
    import cv2

    # OpenCV Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    proposals = []

    for rect in rects[:max_proposals]:
        x, y, w, h = rect
        proposals.append([x, y, x + w, y + h])

    return np.array(proposals, dtype=np.float32)
