"""
模块4: ResNet预训练模型对比
============================
对比不同深度的ResNet预训练模型在图像分类任务上的性能：
- ResNet18 (残差网络18层)
- ResNet34 (残差网络34层)
- ResNet50 (残差网络50层, 使用Bottleneck结构)

ResNet核心思想：
- 残差连接 (Skip Connection): 输出 = F(x) + x
- 解决了深层网络的梯度消失问题
- 使训练上百层的网络成为可能

对比维度：
1. 分类准确率 (Top-1 Accuracy)
2. 推理时间 (Inference Time)
3. 模型参数量 (Number of Parameters)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import time
import pandas as pd


# ============================================================
# 1. 数据准备
# ============================================================
def prepare_mnist_data(batch_size=32, subset_size=2000):
    """
    准备MNIST数据集用于ResNet评估

    MNIST: 70,000张28x28灰度手写数字图像，10个类别(0-9)
    需要:
    - 将28x28缩放到224x224以适配ImageNet预训练的ResNet
    - 将单通道灰度图转换为3通道RGB（ResNet要求3通道输入）

    参数:
        batch_size: 批次大小
        subset_size: 每个模型使用的测试样本数
    """
    # 测试时的预处理：缩放 → 转3通道 → 转Tensor → 归一化
    transform = transforms.Compose([
        transforms.Resize(224),          # ResNet需要224x224输入
        transforms.Grayscale(3),         # 单通道 → 三通道 (复制灰度到RGB)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307, 0.1307, 0.1307],
                             std=[0.3081, 0.3081, 0.3081])  # MNIST的均值与标准差
    ])

    # 加载测试集
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 使用子集加速演示
    if subset_size and subset_size < len(test_dataset):
        indices = np.random.choice(len(test_dataset), subset_size, replace=False)
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # MNIST类别名称
    classes = tuple(str(i) for i in range(10))

    return test_loader, classes


# ============================================================
# 2. 模型加载与评估
# ============================================================
def load_pretrained_resnet(model_name='resnet18'):
    """
    加载预训练ResNet模型

    参数:
        model_name: 'resnet18' | 'resnet34' | 'resnet50'
    返回:
        model: 预训练模型
        model_info: 模型信息字典
    """
    model_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")

    # 加载预训练权重 (ImageNet)
    weights = 'IMAGENET1K_V1'
    model = model_map[model_name](weights=weights)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        'name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_millions': round(total_params / 1e6, 2)
    }

    return model, model_info


def evaluate_resnet(model, loader, device):
    """
    评估ResNet模型在给定数据集上的性能

    返回:
        accuracy: Top-1准确率
        avg_time: 每张图片的平均推理时间 (ms)
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    times = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # 测量推理时间
            start = time.time()
            output = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start

            times.append(elapsed)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    accuracy = correct / total
    # 平均每张图片推理时间（ms）
    avg_time_per_image = (sum(times) / total) * 1000

    return accuracy, avg_time_per_image


def compare_resnet_models(test_loader, device):
    """
    对比多个ResNet模型的性能

    参数:
        test_loader: 数据加载器
        device: 计算设备
    返回:
        results_df: 包含对比结果的DataFrame
        all_results: 详细结果列表
    """
    model_names = ['resnet18', 'resnet34', 'resnet50']
    all_results = []

    for name in model_names:
        model, info = load_pretrained_resnet(name)
        accuracy, avg_time = evaluate_resnet(model, test_loader, device)

        result = {
            'Model': name.upper(),
            'Accuracy': round(accuracy * 100, 2),
            'Inference Time (ms/img)': round(avg_time, 2),
            'Parameters (M)': info['params_millions'],
            'Layers': {
                'resnet18': 18, 'resnet34': 34, 'resnet50': 50
            }[name],
        }
        all_results.append(result)

    df = pd.DataFrame(all_results)
    return df, all_results


# ============================================================
# 3. 可视化函数
# ============================================================
def plot_comparison_charts(results_df):
    """
    绘制ResNet对比图表：准确率、推理时间、参数量
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = results_df['Model'].tolist()
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    # 准确率柱状图
    ax1 = axes[0]
    bars1 = ax1.bar(models, results_df['Accuracy'], color=colors, edgecolor='white')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Top-1 Accuracy Comparison')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, results_df['Accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontweight='bold')

    # 推理时间柱状图
    ax2 = axes[1]
    bars2 = ax2.bar(models, results_df['Inference Time (ms/img)'], color=colors, edgecolor='white')
    ax2.set_ylabel('Time (ms/image)')
    ax2.set_title('Inference Time Comparison')
    for bar, val in zip(bars2, results_df['Inference Time (ms/img)']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', fontweight='bold')

    # 参数量柱状图
    ax3 = axes[2]
    bars3 = ax3.bar(models, results_df['Parameters (M)'], color=colors, edgecolor='white')
    ax3.set_ylabel('Parameters (Millions)')
    ax3.set_title('Model Size Comparison')
    for bar, val in zip(bars3, results_df['Parameters (M)']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}M', ha='center', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_resnet_architecture():
    """
    绘制ResNet残差块结构示意图
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 用文本和箭头描绘残差块结构
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 残差块示意图
    ax.text(6, 7.5, 'Residual Block', fontsize=16, fontweight='bold',
            ha='center')

    # 输入
    ax.annotate('Input x', xy=(6, 6.5), fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'))

    # 主路径
    ax.annotate('', xy=(6, 5.5), xytext=(6, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(6, 5.5, 'Weight Layer\n(Conv + BN + ReLU)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    ax.annotate('', xy=(6, 3.5), xytext=(6, 5.0),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(6, 3.5, 'Weight Layer\n(Conv + BN)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    ax.annotate('', xy=(6, 2.0), xytext=(6, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(6, 2.0, 'F(x) + x', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))

    # 跳跃连接
    ax.annotate('', xy=(5.0, 6.2), xytext=(4.0, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(4.5, 6.5, 'Skip Connection\n(Identity)', fontsize=9, ha='center', color='red')

    ax.annotate('', xy=(5.0, 2.0), xytext=(4.0, 2.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    # 最终输出
    ax.annotate('', xy=(6, 1.0), xytext=(6, 1.7),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(6, 0.7, 'ReLU(F(x) + x)\n→ Output', fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral'))

    # 说明文字
    ax.text(6, 0.1, 'ResNet Core: F(x) = H(x) - x\n'
            'Skip connections enable direct gradient flow to shallower layers, mitigating the vanishing gradient issue',
            fontsize=10, ha='center', style='italic')

    plt.tight_layout()
    return fig


# ============================================================
# 4. 主流程函数
# ============================================================
def run_resnet_comparison(subset_size=1000, batch_size=16):
    """
    运行ResNet模型对比

    参数:
        subset_size: 测试样本数
        batch_size: 批次大小
    返回:
        results: 包含所有结果的字典
    """
    results = {}

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results['device'] = str(device)

    # 数据准备
    test_loader, classes = prepare_mnist_data(
        batch_size=batch_size, subset_size=subset_size
    )
    results['n_test'] = len(test_loader.dataset)

    # 模型对比
    df, all_results = compare_resnet_models(test_loader, device)
    results['dataframe'] = df
    results['all_results'] = all_results

    # 可视化
    results['comparison_fig'] = plot_comparison_charts(df)
    results['arch_fig'] = plot_resnet_architecture()

    return results
