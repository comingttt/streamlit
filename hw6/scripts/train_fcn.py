"""
FCN语义分割训练脚本（可选）
用于在Pascal VOC数据集上训练FCN模型

用法: python scripts/train_fcn.py --epochs 10 --batch_size 8
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fcn import FCN32s, FCN8s
from utils.visualization import compute_pixel_accuracy, compute_mean_iou


def get_voc_dataset(data_dir: str, image_set: str = 'train', size=(320, 320)):
    """获取VOC分割数据集"""
    from torchvision.datasets import VOCSegmentation

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

    def target_to_long(x):
        return (x.squeeze(0) * 255).long()

    dataset = VOCSegmentation(
        root=data_dir,
        year='2012',
        image_set=image_set,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Train FCN for Semantic Segmentation')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集目录')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--model_type', type=str, default='fcn32s', choices=['fcn32s', 'fcn8s'])
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    print("Loading dataset...")
    try:
        train_dataset = get_voc_dataset(args.data_dir, 'train')
        val_dataset = get_voc_dataset(args.data_dir, 'val')
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("尝试使用较小的数据集...")
        # 创建一个小的虚拟数据集用于演示
        class DummySegDataset(torch.utils.data.Dataset):
            def __init__(self, size=50, img_size=(320, 320)):
                self.size = size
                self.img_size = img_size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                img = torch.randn(3, *self.img_size)
                target = torch.randint(0, 21, self.img_size)
                return img, target

        train_dataset = DummySegDataset(100)
        val_dataset = DummySegDataset(20)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'fcn32s':
        model = FCN32s(num_classes=args.num_classes, pretrained=True)
    else:
        model = FCN8s(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        total_pa = 0.0
        total_miou = 0.0
        n = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.numpy()

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                for pred, target in zip(preds, targets):
                    pa = compute_pixel_accuracy(pred, target)
                    miou, _ = compute_mean_iou(pred, target, args.num_classes)
                    total_pa += pa
                    total_miou += miou
                    n += 1

        print(f"Validation - Pixel Accuracy: {total_pa / n:.4f}, mIoU: {total_miou / n:.4f}")

        # 保存检查点
        if epoch % 2 == 0:
            save_path = os.path.join(args.save_dir, f'{args.model_type}_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

    print("Training completed!")


if __name__ == '__main__':
    main()
