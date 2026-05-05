"""
模块3: CNN模型训练与测试
=========================
使用PyTorch实现一个简化版LeNet-5风格的CNN，
在MNIST数据集上进行训练和评估。

LeNet-5 经典结构:
    Input(1x28x28) → Conv(6, 5x5) → ReLU → MaxPool(2x2)
    → Conv(16, 5x5) → ReLU → MaxPool(2x2)
    → Flatten → FC(120) → ReLU → FC(84) → ReLU → FC(10)

训练流程:
    1. 加载MNIST数据集
    2. 定义CNN模型
    3. 训练循环（前向+反向+优化）
    4. 测试评估
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time


# ============================================================
# 1. LeNet-5风格CNN模型定义
# ============================================================
class LeNet5(nn.Module):
    """
    简化版LeNet-5卷积神经网络

    结构说明:
    - Conv1: 1→6通道, 5x5卷积核, 提取低级特征（边缘、纹理）
    - Pool1: 2x2最大池化, 降采样, 保留主要特征
    - Conv2: 6→16通道, 5x5卷积核, 提取高级特征（形状、模式）
    - Pool2: 2x2最大池化
    - FC1: 展平后全连接层 16*4*4 → 120
    - FC2: 120 → 84
    - FC3: 84 → 10 (分类输出)

    参数量计算:
    - Conv1: (5*5*1+1)*6 = 156
    - Conv2: (5*5*6+1)*16 = 2416
    - FC1: 16*4*4*120 + 120 = 30840
    - FC2: 120*84 + 84 = 10164
    - FC3: 84*10 + 10 = 850
    - Total: ~44,426
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 特征提取器：卷积层 + 池化层
        self.features = nn.Sequential(
            # Conv1: 输入1通道 (灰度图), 输出6个特征图
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28→14

            # Conv2: 6→16通道
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14→10→5 (after 5x5 conv: 10-4=6, /2=3)
        )

        # 分类器：全连接层
        # 计算特征图尺寸: 28→Conv(5,p2)→28→Pool→14→Conv(5)→10→Pool→5
        # 所以展平后为 16 * 5 * 5 = 400
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        """
        前向传播
        x: (batch, 1, 28, 28)
        返回: (batch, 10) logits
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. 数据加载
# ============================================================
def load_mnist(batch_size=64, use_subset=False, subset_size=5000):
    """
    加载MNIST数据集

    MNIST: 70,000张28x28手写数字灰度图像 (10类)
    - 训练集: 60,000
    - 测试集: 10,000

    参数:
        batch_size: 批次大小
        use_subset: 是否使用子集（加速演示）
        subset_size: 子集大小
    """
    # 数据预处理：转Tensor + 归一化到[-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值与标准差
    ])

    # 下载/加载数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 可选：使用子集加速训练
    if use_subset:
        train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), subset_size // 5, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# 3. 训练与评估函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        # 梯度清零（PyTorch默认累积梯度）
        optimizer.zero_grad()

        # 前向传播
        output = model(data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播：自动计算梯度
        loss.backward()

        # 参数更新
        optimizer.step()

        # 统计
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，节省内存与加速
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# 4. 可视化函数
# ============================================================
def plot_training_curves_cnn(history):
    """绘制CNN训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, history['test_loss'], 'r-s', label='Test Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=4)
    ax2.plot(epochs, history['test_acc'], 'r-s', label='Test Accuracy', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_predictions(model, loader, device, n_examples=10):
    """可视化预测结果"""
    model.eval()
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            for i in range(min(n_examples, len(data))):
                if len(images_list) >= n_examples:
                    break
                images_list.append(data[i].cpu())
                labels_list.append(target[i].cpu().item())
                preds_list.append(pred[i].cpu().item())

            if len(images_list) >= n_examples:
                break

    cols = min(n_examples, 10)
    rows = (len(images_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images_list):
            img = images_list[idx].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            true_label = labels_list[idx]
            pred_label = preds_list[idx]
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'T:{true_label} P:{pred_label}', color=color, fontsize=10)
        ax.axis('off')

    fig.suptitle('CNN Predictions (Green=Correct, Red=Wrong)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_feature_maps(model, sample_img, device):
    """
    可视化卷积层的特征图
    展示CNN每层学到了什么特征
    """
    model.eval()

    # 获取各卷积层的输出
    conv1_out = None
    conv2_out = None

    def hook_conv1(module, input, output):
        nonlocal conv1_out
        conv1_out = output.detach().cpu()

    def hook_conv2(module, input, output):
        nonlocal conv2_out
        conv2_out = output.detach().cpu()

    hook1 = model.features[0].register_forward_hook(hook_conv1)
    hook2 = model.features[3].register_forward_hook(hook_conv2)

    with torch.no_grad():
        model(sample_img.unsqueeze(0).to(device))

    hook1.remove()
    hook2.remove()

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))

    # Conv1 特征图 (6通道)
    for i in range(min(6, 8)):
        axes[0, i].imshow(conv1_out[0, i].numpy(), cmap='viridis')
        axes[0, i].set_title(f'Conv1-ch{i}', fontsize=9)
        axes[0, i].axis('off')
    for i in range(6, 8):
        axes[0, i].axis('off')

    # Conv2 特征图 (16通道, 只显示前8个)
    for i in range(min(8, 16)):
        axes[1, i].imshow(conv2_out[0, i].numpy(), cmap='viridis')
        axes[1, i].set_title(f'Conv2-ch{i}', fontsize=9)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Conv1 (6 channels)', fontsize=11)
    axes[1, 0].set_ylabel('Conv2 (16 channels)', fontsize=11)
    fig.suptitle('CNN Feature Maps Visualization', fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================
# 5. 主流程函数
# ============================================================
def run_cnn_training(epochs=5, batch_size=64, learning_rate=0.001,
                     use_subset=False, subset_size=5000):
    """
    运行CNN训练流程

    参数:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        use_subset: 是否用小数据集
        subset_size: 子集大小
    返回:
        results: 包含所有结果的字典
    """
    results = {}

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results['device'] = str(device)

    # 加载数据
    train_loader, test_loader = load_mnist(
        batch_size=batch_size, use_subset=use_subset, subset_size=subset_size
    )
    results['n_train'] = len(train_loader.dataset)
    results['n_test'] = len(test_loader.dataset)

    # 创建模型
    model = LeNet5(num_classes=10).to(device)
    results['total_params'] = sum(p.numel() for p in model.parameters())
    results['trainable_params'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # 损失函数与优化器
    # CrossEntropyLoss: 内置了Softmax，所以模型输出不需要再加Softmax
    criterion = nn.CrossEntropyLoss()
    # Adam优化器：结合了动量和自适应学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    log_lines = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        log_line = (
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        log_lines.append(log_line)

    results['log'] = '\n'.join(log_lines)
    results['history'] = history
    results['final_train_acc'] = history['train_acc'][-1]
    results['final_test_acc'] = history['test_acc'][-1]

    # 可视化
    results['curve_fig'] = plot_training_curves_cnn(history)
    results['pred_fig'] = plot_predictions(model, test_loader, device)

    # 特征图
    sample_img, _ = next(iter(test_loader))
    results['feature_map_fig'] = plot_feature_maps(model, sample_img[0], device)

    # 保存模型用于预测
    results['model'] = model
    results['device_obj'] = device

    return results


def predict_image(model, image_tensor, device):
    """
    对单张图片进行预测

    参数:
        model: 训练好的模型
        image_tensor: (1, 28, 28) 或 (28, 28) 的张量
        device: torch device
    返回:
        predicted_class: 预测类别
        probabilities: 各类别概率
    """
    model.eval()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()

    return pred, probs.cpu().numpy().flatten()
