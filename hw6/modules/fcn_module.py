"""
模块1: FCN语义分割 - 高层封装
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fcn import FCN32s, FCN8s
from utils_cv.visualization import (
    draw_segmentation_mask, compute_pixel_accuracy, compute_mean_iou,
    generate_colormap, VOC_CLASSES, Timer
)


class FCNSegmenter:
    """
    FCN语义分割器 - 封装训练、推理、可视化
    """

    def __init__(
        self,
        num_classes: int = 21,
        model_type: str = 'fcn32s',
        pretrained: bool = True,
        device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.model_type = model_type

        if model_type == 'fcn32s':
            self.model = FCN32s(num_classes=num_classes, pretrained=pretrained).to(self.device)
        elif model_type == 'fcn8s':
            self.model = FCN8s(num_classes=num_classes, pretrained=pretrained).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.class_names = VOC_CLASSES[:num_classes]
        self.colormap = generate_colormap(num_classes)

    def get_transform(self, size: Tuple[int, int] = (320, 320), is_train: bool = True):
        """获取数据预处理transform"""
        if is_train:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    def get_mask_transform(self, size: Tuple[int, int] = (320, 320)):
        """获取mask的transform"""
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_pa = 0.0
        total_miou = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.numpy()

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                for pred, target in zip(preds, targets):
                    pa = compute_pixel_accuracy(pred, target)
                    miou, _ = compute_mean_iou(pred, target, self.num_classes)
                    total_pa += pa
                    total_miou += miou
                    n_batches += 1

        return total_pa / n_batches, total_miou / n_batches

    def predict_single(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单张图像进行语义分割
        Args:
            image: (H, W, 3) numpy数组, RGB
        Returns:
            pred_mask: (H, W) 预测的分割mask
            overlay: (H, W, 3) 叠加结果图
        """
        self.model.eval()

        # 保存原始尺寸
        orig_h, orig_w = image.shape[:2]

        # 预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

        # 上采样回原始尺寸
        pred_resized = Image.fromarray(pred.astype(np.uint8)).resize(
            (orig_w, orig_h), Image.NEAREST
        )
        pred_resized = np.array(pred_resized)

        # 生成叠加图
        overlay = draw_segmentation_mask(image, pred_resized, alpha=0.5, color_map=self.colormap)

        return pred_resized, overlay

    def predict_single_pil(self, pil_image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """PIL图像输入版本"""
        image_np = np.array(pil_image)
        return self.predict_single(image_np)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        from utils_cv.visualization import count_parameters
        return {
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'num_params': count_parameters(self.model),
            'backbone': 'VGG16',
        }


# 简化版：使用预训练DeepLabV3进行语义分割（torchvision内置，适合演示）
def get_pretrained_segmentation_model():
    """获取torchvision预训练语义分割模型"""
    from torchvision.models.segmentation import (
        fcn_resnet50, FCN_ResNet50_Weights,
        deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    )
    # 使用预训练FCN（ResNet50），适合演示
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights)
    return model, weights
