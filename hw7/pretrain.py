"""
预训练脚本 — 训练并保存 Rotation CNN 和 MAE 模型
可以单独运行：python pretrain.py
"""

import os
import sys
import torch

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rotation_cnn import RotationCNN, train_rotation_model
from models.mae import SimpleAutoencoder, train_mae_model
from utils.data_utils import get_rotation_dataloader, get_mae_dataloader

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')


def ensure_pretrained_dir():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)


def pretrain_rotation(epochs=5, lr=0.001, device='cpu'):
    """
    预训练旋转预测模型
    返回：模型, 训练历史
    """
    print(f"[Rotation] Loading data...")
    train_loader = get_rotation_dataloader(batch_size=64, train=True)
    val_loader = get_rotation_dataloader(batch_size=64, train=False)

    model = RotationCNN(num_classes=4)

    def progress(epoch, train_loss, val_acc, m):
        print(f"  Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"[Rotation] Training on {device}...")
    history = train_rotation_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr,
        device=device, progress_callback=progress
    )

    # 保存模型
    ensure_pretrained_dir()
    model_path = os.path.join(PRETRAINED_DIR, 'rotation_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'epochs': epochs,
        'lr': lr,
    }, model_path)
    print(f"[Rotation] Model saved to {model_path}")
    return model, history


def pretrain_mae(epochs=5, lr=0.001, mask_ratio=0.5, device='cpu'):
    """
    预训练 MAE 模型
    返回：模型, 训练历史
    """
    print(f"[MAE] Loading data (mask_ratio={mask_ratio})...")
    train_loader = get_mae_dataloader(batch_size=64, train=True, mask_ratio=mask_ratio)
    val_loader = get_mae_dataloader(batch_size=64, train=False, mask_ratio=mask_ratio)

    model = SimpleAutoencoder()

    def progress(epoch, train_loss, val_loss, m):
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"[MAE] Training on {device}...")
    history = train_mae_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr,
        device=device, progress_callback=progress
    )

    # 保存模型
    ensure_pretrained_dir()
    model_path = os.path.join(PRETRAINED_DIR, f'mae_model_mask{mask_ratio}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'epochs': epochs,
        'lr': lr,
        'mask_ratio': mask_ratio,
    }, model_path)
    print(f"[MAE] Model saved to {model_path}")
    return model, history


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*50)
    print("Pre-training Rotation Prediction Model")
    print("="*50)
    pretrain_rotation(epochs=5, lr=0.001, device=device)

    print("\n" + "="*50)
    print("Pre-training MAE Models (different mask ratios)")
    print("="*50)
    for ratio in [0.25, 0.5, 0.75]:
        print(f"\n--- Mask Ratio: {ratio} ---")
        pretrain_mae(epochs=5, lr=0.001, mask_ratio=ratio, device=device)

    print("\nAll models pre-trained and saved!")
