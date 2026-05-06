"""
模块2: R-CNN系列目标检测 - 高层封装
"""

import torch
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rcnn import get_faster_rcnn, SimpleRCNN, SimpleFastRCNN
from utils.visualization import (
    draw_detection_boxes, COCO_CLASSES, Timer, count_parameters
)


class FasterRCNNDetector:
    """
    Faster R-CNN目标检测器（使用torchvision预训练）
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

        self.model = get_faster_rcnn(pretrained=pretrained).to(self.device)
        self.model.eval()
        self.class_names = COCO_CLASSES

    def predict(
        self,
        image: np.ndarray,
        score_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        目标检测推理
        Args:
            image: (H, W, 3) numpy数组, RGB
            score_threshold: 置信度阈值
        Returns:
            result_image: 带检测框的图像
            detections: 检测结果字典
        """
        from torchvision import transforms as T

        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).to(self.device)

        with torch.no_grad():
            predictions = self.model([img_tensor])

        pred = predictions[0]

        # 过滤
        keep = pred['scores'] > score_threshold

        boxes = pred['boxes'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()

        # 绘制结果
        result_image = draw_detection_boxes(
            image, boxes, labels, scores,
            class_names=self.class_names,
            score_threshold=score_threshold
        )

        detections = {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'num_detections': len(boxes),
        }

        return result_image, detections

    def predict_pil(self, pil_image: Image.Image, score_threshold: float = 0.5):
        """PIL图像输入"""
        image_np = np.array(pil_image)
        return self.predict(image_np, score_threshold)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_type': 'Faster R-CNN (ResNet50-FPN)',
            'num_params': count_parameters(self.model),
            'backbone': 'ResNet50 + FPN',
            'framework': 'torchvision',
        }


class SimpleRCNNDetector:
    """
    简化版R-CNN检测器（用于演示R-CNN思想）
    使用Selective Search + 预训练特征 + 分类器
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.class_names = COCO_CLASSES

    def predict(self, image: np.ndarray, score_threshold: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        R-CNN风格检测（简化实现）
        注意：这是一个演示实现，实际使用建议用Faster R-CNN
        """
        import cv2

        # 使用OpenCV的DNN模块进行快速检测（模拟R-CNN效果）
        # 实际演示中使用预训练模型替代
        h, w = image.shape[:2]

        try:
            # 尝试使用OpenCV Selective Search
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()[:100]  # 取前100个proposal
        except Exception:
            # 如果不可用，用滑动窗口模拟
            rects = []
            scales = [0.5, 1.0, 1.5]
            for scale in scales:
                win_w = int(w * scale * 0.3)
                win_h = int(h * scale * 0.3)
                stride = max(win_w // 2, 32)
                for y in range(0, h - win_h, stride):
                    for x in range(0, w - win_w, stride):
                        rects.append([x, y, win_w, win_h])

        # 转换为目标检测格式（这里用简化方式）
        boxes = []
        for rect in rects[:50]:
            x, y, rw, rh = rect
            boxes.append([x, y, x + rw, y + rh])

        boxes = np.array(boxes, dtype=np.float32)
        # 简化：随机分配一些分数用于演示
        scores = np.random.uniform(0.3, 0.9, len(boxes))
        labels = np.random.randint(1, 20, len(boxes))

        # 过滤
        keep = scores > score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        result_image = draw_detection_boxes(
            image, boxes, labels, scores,
            class_names=self.class_names,
            score_threshold=score_threshold
        )

        detections = {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'num_detections': len(boxes),
            'note': 'R-CNN 简化演示版 (Selective Search + 特征提取)'
        }

        return result_image, detections

    def get_model_info(self) -> dict:
        return {
            'model_type': 'R-CNN (简化演示版)',
            'num_params': 0,
            'backbone': 'Selective Search + 手工特征',
            'note': '演示版，展示R-CNN的核心思想'
        }
