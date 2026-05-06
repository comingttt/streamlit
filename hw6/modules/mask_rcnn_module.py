"""
模块3: Mask R-CNN实例分割 - 高层封装
"""

import torch
import numpy as np
from PIL import Image
import os
from typing import Tuple, Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask_rcnn import MaskRCNNPredictor
from utils.visualization import (
    draw_instance_masks, COCO_CLASSES, Timer, count_parameters
)


class MaskRCNNSegmenter:
    """
    Mask R-CNN实例分割器
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.predictor = MaskRCNNPredictor(pretrained=pretrained, device=str(self.device))
        self.class_names = COCO_CLASSES

    def predict(
        self,
        image: np.ndarray,
        score_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        实例分割推理
        Args:
            image: (H, W, 3) numpy数组, RGB
            score_threshold: 置信度阈值
        Returns:
            result_image: 带实例分割结果的图像
            detections: 检测结果
        """
        result = self.predictor.predict(image, score_threshold=score_threshold)

        # 绘制结果
        result_image = draw_instance_masks(
            image,
            result['masks'],
            result['boxes'],
            result['labels'],
            result['scores'],
            class_names=self.class_names,
            score_threshold=score_threshold,
        )

        detections = {
            'boxes': result['boxes'],
            'labels': result['labels'],
            'scores': result['scores'],
            'num_instances': len(result['boxes']),
        }

        return result_image, detections

    def predict_pil(self, pil_image: Image.Image, score_threshold: float = 0.5):
        image_np = np.array(pil_image)
        return self.predict(image_np, score_threshold)

    def get_model_info(self) -> dict:
        return {
            'model_type': 'Mask R-CNN (ResNet50-FPN)',
            'num_params': count_parameters(self.predictor.model),
            'backbone': 'ResNet50 + FPN',
            'framework': 'torchvision',
            'capabilities': ['bounding box', 'instance mask', 'class label'],
        }
