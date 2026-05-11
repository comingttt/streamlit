"""
旋转预测模型 (Rotation Prediction CNN)
- 输入：28x28 灰度图像（已随机旋转）
- 输出：4 分类（0°, 90°, 180°, 270°）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationCNN(nn.Module):
    """
    简单的 CNN 分类器，用于预测图像的旋转角度
    架构：Conv -> Conv -> FC -> FC
    """
    def __init__(self, num_classes=4):
        super(RotationCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)                              # 减半
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28 -> 14 -> 7
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)              # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_rotation_model(model, train_loader, val_loader, epochs=5, lr=0.001,
                         device='cpu', progress_callback=None):
    """
    训练旋转预测模型
    参数：
        model: RotationCNN 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        progress_callback: 每 epoch 结束后的回调函数 (epoch, train_loss, val_acc, model)
    返回：
        history: 包含 train_losses, val_accs 的字典
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- 验证阶段 ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        val_accs.append(val_acc)

        # 回调通知进度
        if progress_callback:
            progress_callback(epoch + 1, avg_train_loss, val_acc, model)

    return {
        'train_losses': train_losses,
        'val_accs': val_accs
    }
