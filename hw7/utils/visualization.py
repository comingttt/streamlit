"""
可视化工具模块
- Matplotlib 图像绘制
- Plotly 交互式图表
- 训练曲线绘制
"""

import matplotlib
matplotlib.use('Agg')  # 非交互后端，兼容 Streamlit
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Matplotlib 图像可视化
# ============================================================

def denormalize(tensor):
    """将归一化的 MNIST tensor 转为可显示图像 [0, 1]"""
    img = tensor.clone()
    # MNIST 归一化：mean=0.1307, std=0.3081
    img = img * 0.3081 + 0.1307
    img = img.clamp(0, 1)
    return img


def tensor_to_numpy(tensor):
    """将 tensor [1, H, W] 或 [H, W] 转为 numpy array"""
    t = tensor.detach().cpu()
    if t.dim() == 3:
        t = t.squeeze(0)
    return t.numpy()


def show_rotation_prediction(original_img, rotated_img, prediction, ground_truth=None):
    """
    显示旋转预测结果图
    返回 matplotlib figure
    """
    angles = ['0°', '90°', '180°', '270°']

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # 原图
    axes[0].imshow(tensor_to_numpy(denormalize(original_img)), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    # 旋转图 + 预测
    axes[1].imshow(tensor_to_numpy(denormalize(rotated_img)), cmap='gray')
    title = f'Rotated → Pred: {angles[prediction]}'
    if ground_truth is not None:
        title += f' | True: {angles[ground_truth]}'
        color = 'green' if prediction == ground_truth else 'red'
        axes[1].set_title(title, color=color)
    else:
        axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    return fig


def show_mae_reconstruction(original_img, masked_img, reconstructed_img, mask_ratio=0.5):
    """
    显示 MAE 重建结果：原图 / 遮挡图 / 重建图
    返回 matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    titles = ['Original', f'Masked ({int(mask_ratio*100)}%)', 'Reconstructed']
    imgs = [original_img, masked_img, reconstructed_img]

    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(tensor_to_numpy(denormalize(img)), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    return fig


def show_mask_viz(original_img, masked_img, mask_ratio=0.5):
    """显示原图和遮挡图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(tensor_to_numpy(denormalize(original_img)), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(tensor_to_numpy(denormalize(masked_img)), cmap='gray')
    axes[1].set_title(f'Masked ({int(mask_ratio*100)}%)')
    axes[1].axis('off')
    plt.tight_layout()
    return fig


def show_image_grid(images, titles=None, ncols=4, figsize=(12, 3)):
    """显示多张图像网格"""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(nrows * ncols):
        if i < n:
            axes[i].imshow(tensor_to_numpy(denormalize(images[i])), cmap='gray')
            if titles:
                axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    return fig


# ============================================================
# Plotly 交互式图表
# ============================================================

def create_loss_plot(train_losses, val_accs_or_losses=None, title='Training Loss',
                     x_label='Epoch', y1_label='Loss', y2_label=None):
    """
    使用 Plotly 创建交互式训练曲线图
    支持双 Y 轴
    """
    epochs = list(range(1, len(train_losses) + 1))

    if y2_label and val_accs_or_losses is not None:
        # 双 Y 轴
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, mode='lines+markers',
                       name='Train Loss', line=dict(color='#1f77b4', width=2)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_accs_or_losses, mode='lines+markers',
                       name=y2_label, line=dict(color='#ff7f0e', width=2)),
            secondary_y=True
        )
        fig.update_yaxes(title_text=y1_label, secondary_y=False)
        fig.update_yaxes(title_text=y2_label, secondary_y=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_losses, mode='lines+markers',
            name=y1_label, line=dict(color='#1f77b4', width=2)
        ))
        if val_accs_or_losses is not None:
            fig.add_trace(go.Scatter(
                x=epochs, y=val_accs_or_losses, mode='lines+markers',
                name='Val Loss', line=dict(color='#d62728', width=2)
            ))
        fig.update_yaxes(title_text=y1_label)

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def create_comparison_bar_chart(categories, values, title='Comparison', y_label='Value'):
    """创建对比柱状图"""
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=colors[:len(categories)],
        text=[f'{v:.4f}' for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
