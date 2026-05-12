"""
模型定义模块
- Autoencoder（普通自编码器）
- Variational Autoencoder（变分自编码器，VAE）
- DCGAN（深度卷积生成对抗网络）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Autoencoder
# ============================================================
class Autoencoder(nn.Module):
    """
    普通自编码器：将 28×28 的 MNIST 图像压缩到低维潜空间，再重构回原始图像。
    架构：Conv2d 编码 + ConvTranspose2d 解码
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- 编码器 ----
        # 输入: (B, 1, 28, 28) -> (B, 32, 14, 14) -> (B, 64, 7, 7)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # ---- 解码器 ----
        # 输入: (B, latent_dim) -> (B, 1, 28, 28)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def encode(self, x):
        """仅编码，返回潜向量"""
        return self.encoder(x)

    def decode(self, z):
        """仅解码，从潜向量生成图像"""
        return self.decoder(z)


# ============================================================
# Variational Autoencoder (VAE)
# ============================================================
class VAE(nn.Module):
    """
    变分自编码器：编码器输出潜空间的均值 μ 和对数方差 log(σ²)，
    通过重参数化技巧采样潜向量 z = μ + σ·ε，其中 ε ~ N(0, I)。
    损失 = 重构损失(MSE) + KL 散度。
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- 编码器（输出 μ 和 logvar） ----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)       # 均值
        self.fc_logvar = nn.Linear(256, latent_dim)   # 对数方差

        # ---- 解码器 ----
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """编码：返回潜向量 z（采样后）及 μ、logvar"""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：z = μ + σ·ε
        使得梯度可以反向传播通过采样操作。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def decode(self, z):
        """从潜向量解码生成图像"""
        return self.decoder(z)

    def sample(self, num_samples: int, device: str = "cpu"):
        """
        从标准正态先验 N(0, I) 中采样潜向量并生成图像。
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


# ============================================================
# DCGAN
# ============================================================
class DCGANGenerator(nn.Module):
    """
    DCGAN 生成器：将随机噪声向量映射为 28×28 的 MNIST 风格图像。
    使用 ConvTranspose2d 逐步上采样。
    """

    def __init__(self, noise_dim: int = 100):
        super().__init__()
        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            # 输入: (B, noise_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=3, stride=1, padding=0),  # 1 -> 3
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 3 -> 7
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 7 -> 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 14 -> 28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        # z: (B, noise_dim) -> (B, noise_dim, 1, 1)
        z = z.view(z.size(0), self.noise_dim, 1, 1)
        return self.model(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN 判别器：判断输入图像是真实 MNIST 图像还是生成器伪造的。
    输出一个标量概率。
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 输入: (B, 1, 28, 28) -> (B, 64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 64, 14, 14) -> (B, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 128, 7, 7) -> (B, 256, 3, 3)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid(),  # 输出概率 [0, 1]
        )

    def forward(self, x):
        return self.model(x)
