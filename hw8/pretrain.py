"""
预训练脚本：提前训练好模型并保存到 saved_models/
- AE (latent_dim=2, epochs=5)
- VAE (latent_dim=2, epochs=5)  
- DCGAN (noise_dim=100, epochs=10)
进度写入 pretrain_log.txt
"""

import torch
import numpy as np
import time
import os
import sys

from data_utils import get_mnist_loaders
from model_defs import Autoencoder, VAE, DCGANGenerator, DCGANDiscriminator
from train_utils import get_device, train_autoencoder, train_vae, train_dcgan, save_model

# 日志文件
LOG_FILE = os.path.join(os.path.dirname(__file__), "pretrain_log.txt")

def log(msg):
    """同时打印到终端和日志文件"""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# 清空日志
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

device = get_device()
log(f"Device: {device}")
log(f"Save dir: {os.path.abspath('saved_models')}")
log("")

# 共享数据加载器
log("Loading MNIST data...")
train_loader, test_loader = get_mnist_loaders(batch_size=64)
log(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
log("")

# ============================================================
# 1. 训练 Autoencoder (latent_dim=2, 5 epochs)
# ============================================================
log("=" * 60)
log("1. Training Autoencoder (latent_dim=2, epochs=5)")
log("=" * 60)
ae = Autoencoder(latent_dim=2)

def ae_cb(ep, total, loss):
    log(f"  AE Epoch {ep}/{total} - Loss: {loss:.6f}")

t0 = time.time()
ae, ae_losses = train_autoencoder(ae, train_loader, epochs=5, device=device, progress_callback=ae_cb)
save_model(ae, "ae_ld2_ep5")
log(f"  AE done in {time.time()-t0:.1f}s, final loss: {ae_losses[-1]:.6f}")
log("")

# ============================================================
# 2. 训练 VAE (latent_dim=2, 5 epochs)
# ============================================================
log("=" * 60)
log("2. Training VAE (latent_dim=2, epochs=5)")
log("=" * 60)
vae = VAE(latent_dim=2)

def vae_cb(ep, total, loss):
    log(f"  VAE Epoch {ep}/{total} - Total Loss: {loss:.4f}")

t0 = time.time()
vae, vae_losses = train_vae(vae, train_loader, epochs=5, device=device, progress_callback=vae_cb)
save_model(vae, "vae_ld2_ep5")
log(f"  VAE done in {time.time()-t0:.1f}s, final loss: {vae_losses[-1][0]:.4f}")
log("")

# ============================================================
# 3. 训练 DCGAN (noise_dim=100, 10 epochs)
# ============================================================
log("=" * 60)
log("3. Training DCGAN (noise_dim=100, epochs=10)")
log("=" * 60)
gen = DCGANGenerator(noise_dim=100)
disc = DCGANDiscriminator()

def dcgan_cb(ep, total, g_loss, d_loss):
    log(f"  DCGAN Epoch {ep}/{total} - G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")

t0 = time.time()
gen, disc, g_losses, d_losses = train_dcgan(
    gen, disc, train_loader, epochs=10, device=device, progress_callback=dcgan_cb
)
save_model(gen, "dcgan_g_nd100_ep10")
save_model(disc, "dcgan_d_nd100_ep10")
log(f"  DCGAN done in {time.time()-t0:.1f}s")
log(f"  Final G Loss: {g_losses[-1]:.4f} | D Loss: {d_losses[-1]:.4f}")
log("")

log("=" * 60)
log("ALL MODELS TRAINED SUCCESSFULLY!")
log("=" * 60)
