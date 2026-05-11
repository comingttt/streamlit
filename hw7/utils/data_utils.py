"""
数据加载和预处理工具模块
- MNIST 数据集加载
- 旋转数据生成（自监督学习）
- 图像遮挡（MAE 自监督学习）
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image


def get_mnist_dataloader(batch_size=64, train=True):
    """加载 MNIST 数据集，返回 DataLoader"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(
        root='./data', train=train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


# ============================================================
# Rotation Prediction 数据生成器
# ============================================================

class RotationDataset(Dataset):
    """
    自监督旋转预测数据集
    对每张图像随机应用 0°, 90°, 180°, 270° 旋转
    标签为旋转角度类别：0, 1, 2, 3
    """
    def __init__(self, mnist_dataset):
        self.dataset = mnist_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # 忽略原始标签
        # 随机选择旋转角度：0°, 90°, 180°, 270°
        rotation_label = np.random.randint(0, 4)
        k = rotation_label  # 旋转 k * 90 度
        # torch.rot90 逆时针旋转，k 次即 k*90°
        rotated_img = torch.rot90(img, k, dims=[1, 2])
        return rotated_img, rotation_label


def get_rotation_dataloader(batch_size=64, train=True):
    """获取旋转预测任务的 DataLoader"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    mnist_dataset = torchvision.datasets.MNIST(
        root='./data', train=train, download=True, transform=transform
    )
    rotation_dataset = RotationDataset(mnist_dataset)
    return DataLoader(rotation_dataset, batch_size=batch_size, shuffle=train)


# ============================================================
# MAE 数据生成器 — 随机遮挡
# ============================================================

def mask_image(img_tensor, mask_ratio=0.5):
    """
    对单张图像张量 [1, H, W] 进行随机块状遮挡
    返回：(遮挡后图像, 遮挡图, mask 布尔矩阵)
    """
    _, h, w = img_tensor.shape
    masked_img = img_tensor.clone()
    mask = torch.zeros((h, w), dtype=torch.bool)

    # 块状遮挡：将图像划分为 patch，随机遮挡部分 patch
    patch_size = 7  # MNIST 28x28, 7x7 的 patch 共 16 个
    num_patches_h = h // patch_size   # 4
    num_patches_w = w // patch_size   # 4
    total_patches = num_patches_h * num_patches_w  # 16
    num_mask = int(total_patches * mask_ratio)

    # 随机选择要遮挡的 patch 索引
    mask_indices = np.random.choice(total_patches, num_mask, replace=False)

    for idx in mask_indices:
        row = idx // num_patches_w
        col = idx % num_patches_w
        r_start = row * patch_size
        c_start = col * patch_size
        masked_img[:, r_start:r_start+patch_size, c_start:c_start+patch_size] = 0
        mask[r_start:r_start+patch_size, c_start:c_start+patch_size] = True

    return masked_img, mask


class MAEDataset(Dataset):
    """
    MAE 自监督数据集：对每张图像进行随机遮挡
    返回：(遮挡图像, 原始图像) — 用于自编码器重建
    """
    def __init__(self, mnist_dataset, mask_ratio=0.5):
        self.dataset = mnist_dataset
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        masked_img, _ = mask_image(img, self.mask_ratio)
        return masked_img, img  # 输入遮挡图，目标是原图


def get_mae_dataloader(batch_size=64, train=True, mask_ratio=0.5):
    """获取 MAE 任务的 DataLoader"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    mnist_dataset = torchvision.datasets.MNIST(
        root='./data', train=train, download=True, transform=transform
    )
    mae_dataset = MAEDataset(mnist_dataset, mask_ratio=mask_ratio)
    return DataLoader(mae_dataset, batch_size=batch_size, shuffle=train)


def process_uploaded_image(uploaded_file):
    """
    处理用户上传的图像：转为 28x28 灰度图，返回 tensor [1, 1, 28, 28]
    """
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 1, 28, 28]
    return img_tensor
