"""
数据加载工具模块
- 从本地 MNIST 原始文件加载数据（IDX 格式）
- 自动解压 .gz 压缩文件
- 提供 DataLoader 创建功能
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
import gzip
import shutil


def _decompress_mnist(data_dir: str = "data"):
    """将 MNIST/raw/ 下的 .gz 文件解压为 .ubyte 文件"""
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    if not os.path.isdir(raw_dir):
        return

    for fname in os.listdir(raw_dir):
        if fname.endswith(".gz"):
            gz_path = os.path.join(raw_dir, fname)
            out_path = os.path.join(raw_dir, fname[:-3])  # 去掉 .gz
            if not os.path.exists(out_path):
                with gzip.open(gz_path, "rb") as f_in:
                    with open(out_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)


def get_mnist_loaders(batch_size: int = 128, data_dir: str = "data"):
    """
    加载 MNIST 数据集，返回训练和测试 DataLoader。
    优先使用本地文件，如果只有 .gz 压缩文件则自动解压。

    Args:
        batch_size: 批次大小
        data_dir: 数据根目录，期望结构为 {data_dir}/MNIST/raw/

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 尝试本地加载，找不到就尝试解压 .gz 再加载
    try:
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=False, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=False, transform=transform
        )
    except RuntimeError:
        _decompress_mnist(data_dir)
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=False, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=False, transform=transform
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def get_test_subset(test_loader: DataLoader, n_samples: int = 500):
    """
    从测试集 DataLoader 中提取前 n_samples 个样本，
    用于潜空间可视化（避免一次性展示全部 10000 个点导致散点图过于拥挤）。

    Returns:
        images: Tensor (n_samples, 1, 28, 28)
        labels: Tensor (n_samples,)
    """
    images_list, labels_list = [], []
    count = 0
    for x, y in test_loader:
        images_list.append(x)
        labels_list.append(y)
        count += x.size(0)
        if count >= n_samples:
            break

    images = torch.cat(images_list, dim=0)[:n_samples]
    labels = torch.cat(labels_list, dim=0)[:n_samples]
    return images, labels
