"""
模块3: Mask R-CNN 实例分割模型
使用torchvision预训练模型
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image


def get_mask_rcnn(
    num_classes: int = 91,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
) -> nn.Module:
    """
    获取Mask R-CNN模型

    Mask R-CNN核心创新（相比Faster R-CNN）：
    1. 添加Mask分支，与分类、回归并行
    2. 使用RoIAlign替代RoIPool，解决量化误差
    3. 解耦mask预测与类别预测

    Args:
        num_classes: 类别数（含背景）
        pretrained: 是否使用COCO预训练权重
    """
    if pretrained:
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        )
    else:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes
        )
    return model


class MaskRCNNPredictor:
    """
    Mask R-CNN推理封装器
    """

    def __init__(
        self,
        pretrained: bool = True,
        score_threshold: float = 0.5,
        device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = get_mask_rcnn(pretrained=pretrained).to(self.device)
        self.model.eval()

    def predict(
        self,
        image: np.ndarray,
        score_threshold: float = 0.5
    ) -> Dict:
        """
        对单张图像进行实例分割预测
        Args:
            image: (H, W, 3) numpy数组, RGB格式
            score_threshold: 置信度阈值
        Returns:
            {
                'boxes': (N, 4) 边界框
                'labels': (N,) 类别索引
                'scores': (N,) 置信度分数
                'masks': (N, H, W) 实例mask
            }
        """
        from torchvision import transforms as T

        # 预处理
        transform = T.Compose([
            T.ToTensor(),
        ])
        img_tensor = transform(image).to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model([img_tensor])

        pred = predictions[0]

        # 过滤低置信度结果
        keep = pred['scores'] > score_threshold

        result = {
            'boxes': pred['boxes'][keep].cpu().numpy(),
            'labels': pred['labels'][keep].cpu().numpy(),
            'scores': pred['scores'][keep].cpu().numpy(),
            'masks': pred['masks'][keep, 0].cpu().numpy(),
        }

        return result
