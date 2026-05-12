"""
训练工具模块
- train_autoencoder: 训练普通自编码器
- train_vae: 训练变分自编码器
- train_dcgan: 训练 DCGAN
- 提供模型保存/加载功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from model_defs import Autoencoder, VAE, DCGANGenerator, DCGANDiscriminator

# 模型保存目录
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_device():
    """获取可用设备（优先 GPU，否则 CPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# Autoencoder 训练
# ============================================================
def train_autoencoder(
    model: Autoencoder,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = None,
    progress_callback=None,
):
    """
    训练 Autoencoder。

    Returns:
        model: 训练好的模型
        losses: 每个 epoch 的平均 loss 列表
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_loss)

    return model, losses


# ============================================================
# VAE 训练
# ============================================================
def vae_loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    VAE 损失函数：
    - 重构损失：MSE（也可用 BCE）
    - KL 散度：D_KL(q(z|x) || p(z))，其中 p(z) = N(0, I)
    - beta 参数控制 KL 项的权重（beta-VAE）
    """
    # 重构损失
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    # KL 散度: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div / x.size(0)

    return recon_loss + beta * kl_div, recon_loss, kl_div


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: torch.device = None,
    progress_callback=None,
):
    """
    训练 VAE。

    Returns:
        model: 训练好的模型
        losses: 每个 epoch 的 (total_loss, recon_loss, kl_loss) 列表
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x)
            total_loss, recon_loss, kl_loss = vae_loss_function(recon, x, mu, logvar, beta)
            total_loss.backward()
            optimizer.step()
            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        n = len(train_loader)
        losses.append((epoch_total / n, epoch_recon / n, epoch_kl / n))

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_total / n)

    return model, losses


# ============================================================
# DCGAN 训练
# ============================================================
def train_dcgan(
    generator: DCGANGenerator,
    discriminator: DCGANDiscriminator,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 2e-4,
    device: torch.device = None,
    progress_callback=None,
):
    """
    训练 DCGAN。

    Returns:
        generator, discriminator: 训练好的模型
        g_losses, d_losses: 每个 epoch 的平均 loss 列表
    """
    if device is None:
        device = get_device()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    noise_dim = generator.noise_dim
    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = 0

        for real_imgs, _ in train_loader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 真实标签（平滑到 0.9）和假标签
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---- 训练判别器 ----
            d_optimizer.zero_grad()

            # 真实图像 loss
            real_output = discriminator(real_imgs)
            d_real_loss = criterion(real_output, real_labels)

            # 假图像 loss
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = generator(noise)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ---- 训练生成器 ----
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = generator(noise)
            fake_output = discriminator(fake_imgs)

            # 生成器希望判别器认为假图像是真的
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            n_batches += 1

        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_g, avg_d)

    return generator, discriminator, g_losses, d_losses


# ============================================================
# 模型保存/加载
# ============================================================
def save_model(model, name: str):
    """保存模型到 saved_models/ 目录"""
    path = os.path.join(SAVE_DIR, f"{name}.pth")
    torch.save(model.state_dict(), path)
    return path


def load_model(model, name: str, device: torch.device = None):
    """从 saved_models/ 目录加载模型"""
    if device is None:
        device = get_device()
    path = os.path.join(SAVE_DIR, f"{name}.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        return True
    return False


def model_exists(name: str) -> bool:
    """检查模型是否已保存"""
    return os.path.exists(os.path.join(SAVE_DIR, f"{name}.pth"))
