"""
Masked Autoencoder (MAE) 简化版
- 编码器：将遮挡图像压缩为低维表示
- 解码器：从低维表示重建原始图像
- 使用 MSE Loss 训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    """
    简单卷积自编码器，用于图像重建
    编码器逐步降采样，解码器逐步上采样恢复
    """
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # ---- 编码器 ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 7 -> 7
            nn.ReLU(),
        )
        # ---- 解码器 ----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            # 不加激活函数，让模型自由输出任意值以匹配归一化后的输入
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_mae_model(model, train_loader, val_loader, epochs=5, lr=0.001,
                    device='cpu', progress_callback=None):
    """
    训练 MAE（Masked Autoencoder）模型
    参数：
        model: SimpleAutoencoder 模型实例
        train_loader: 训练数据加载器（返回 masked_img, original_img）
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        progress_callback: 每 epoch 结束后的回调 (epoch, train_loss, val_loss, model)
    返回：
        history: 包含 train_losses, val_losses 的字典
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        model.train()
        running_loss = 0.0
        for masked_imgs, original_imgs in train_loader:
            masked_imgs = masked_imgs.to(device)
            original_imgs = original_imgs.to(device)

            optimizer.zero_grad()
            reconstructed = model(masked_imgs)
            loss = criterion(reconstructed, original_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- 验证阶段 ----
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for masked_imgs, original_imgs in val_loader:
                masked_imgs = masked_imgs.to(device)
                original_imgs = original_imgs.to(device)
                reconstructed = model(masked_imgs)
                loss = criterion(reconstructed, original_imgs)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if progress_callback:
            progress_callback(epoch + 1, avg_train_loss, avg_val_loss, model)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
