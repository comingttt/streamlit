"""
自监督学习教学项目 — Streamlit 主应用
包含四个页面：
  1. Rotation Prediction（旋转预测）
  2. MAE Reconstruction（遮挡自编码器重建）
  3. Comparison（对比实验）
  4. About Project（项目说明）
"""

import sys
import os
import time

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入项目模块
from models.rotation_cnn import RotationCNN, train_rotation_model
from models.mae import SimpleAutoencoder, train_mae_model
from utils.data_utils import (
    get_rotation_dataloader, get_mae_dataloader,
    mask_image, process_uploaded_image
)
from utils.visualization import (
    show_rotation_prediction, show_mae_reconstruction,
    show_mask_viz, show_image_grid,
    create_loss_plot, create_comparison_bar_chart,
    denormalize, tensor_to_numpy
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="自监督学习教学项目",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 常量
# ============================================================
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ANGLES = ['0°', '90°', '180°', '270°']

# ============================================================
# 预训练模型加载与初始化
# ============================================================

@st.cache_resource
def load_pretrained_rotation_model():
    """加载或初始化 Rotation 模型"""
    model = RotationCNN(num_classes=4).to(DEVICE)
    path = os.path.join(PRETRAINED_DIR, 'rotation_model.pt')
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('history', None), True
    return model, None, False


@st.cache_resource
def load_pretrained_mae_model(mask_ratio=0.5):
    """加载或初始化 MAE 模型"""
    model = SimpleAutoencoder().to(DEVICE)
    path = os.path.join(PRETRAINED_DIR, f'mae_model_mask{mask_ratio}.pt')
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('history', None), True
    return model, None, False


def load_or_get_mae(mask_ratio):
    """从 session_state 获取或加载 MAE 模型"""
    key = f'mae_model_{mask_ratio}'
    history_key = f'mae_history_{mask_ratio}'
    pretrained_key = f'mae_pretrained_{mask_ratio}'

    if key not in st.session_state:
        model, history, pretrained = load_pretrained_mae_model(mask_ratio)
        st.session_state[key] = model
        st.session_state[history_key] = history
        st.session_state[pretrained_key] = pretrained
    return st.session_state[key], st.session_state[history_key], st.session_state[pretrained_key]


# ============================================================
# 初始化 Session State
# ============================================================

def init_session_state():
    """初始化所有 session state 变量"""
    defaults = {
        'rotation_model': None,
        'rotation_history': None,
        'rotation_pretrained': False,
        'rotation_trained': False,
        'mae_training': False,
        'rotation_training': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 初始化 rotation model
    if st.session_state['rotation_model'] is None:
        model, history, pretrained = load_pretrained_rotation_model()
        st.session_state['rotation_model'] = model
        st.session_state['rotation_history'] = history
        st.session_state['rotation_pretrained'] = pretrained
        st.session_state['rotation_trained'] = pretrained


# ============================================================
# 自定义 CSS 样式
# ============================================================

def inject_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
    }
    .divider {
        border-top: 2px solid #eee;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 页面：首页 / Rotation Prediction
# ============================================================

def page_rotation_prediction():
    """旋转预测自监督学习页面"""
    st.markdown('<p class="main-header">🔄 Rotation Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">自监督学习：预测图像的旋转角度（0°, 90°, 180°, 270°）</p>',
        unsafe_allow_html=True
    )

    # ---- 模型状态 ----
    model = st.session_state['rotation_model']
    trained = st.session_state['rotation_trained']
    history = st.session_state['rotation_history']

    # ---- 侧边栏控件 ----
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### ⚙️ 训练设置")

        epochs = st.slider("训练 Epochs", 1, 20, 5, key="rot_epochs")
        lr = st.selectbox("学习率", [0.01, 0.005, 0.001, 0.0005, 0.0001],
                          index=2, key="rot_lr")

        btn_train = st.button("🚀 开始训练", type="primary", use_container_width=True,
                              disabled=st.session_state.get('rotation_training', False))

        # 模型状态指示
        if trained:
            st.markdown('<div class="success-box">✅ 模型已训练</div>', unsafe_allow_html=True)
            if history:
                st.metric("最佳验证准确率", f"{max(history['val_accs'])*100:.1f}%")
        else:
            st.info("💡 点击「开始训练」来训练模型")

        # 上传图片测试
        st.markdown("---")
        st.markdown("### 🖼️ 上传图片测试")
        uploaded = st.file_uploader("上传 PNG/JPG 图片", type=['png', 'jpg', 'jpeg'],
                                    key="rot_upload")

    with col_main:
        # ---- 说明区域 ----
        if not trained:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 任务说明</h4>
            模型将学习识别图像被旋转的角度。这是一个典型的<strong>自监督学习</strong>任务：
            我们不需要人工标注，只需对图像随机旋转并记录角度作为伪标签即可训练。
            </div>
            """, unsafe_allow_html=True)

        # ---- 训练 ----
        if btn_train:
            st.session_state['rotation_training'] = True
            with st.spinner(f"正在 {DEVICE} 上训练 Rotation Prediction 模型..."):
                train_loader = get_rotation_dataloader(batch_size=64, train=True)
                val_loader = get_rotation_dataloader(batch_size=64, train=False)

                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_placeholder = st.empty()
                acc_placeholder = st.empty()

                train_loss_history = []
                val_acc_history = []

                def progress_callback(epoch, train_loss, val_acc, m):
                    train_loss_history.append(train_loss)
                    val_acc_history.append(val_acc)
                    pct = epoch / epochs
                    progress_bar.progress(pct)
                    status_text.text(f"Epoch {epoch}/{epochs}")
                    loss_placeholder.metric("Train Loss", f"{train_loss:.4f}")
                    acc_placeholder.metric("Val Accuracy", f"{val_acc*100:.1f}%")

                history = train_rotation_model(
                    model, train_loader, val_loader, epochs=epochs, lr=lr,
                    device=DEVICE, progress_callback=progress_callback
                )

                st.session_state['rotation_model'] = model
                st.session_state['rotation_history'] = history
                st.session_state['rotation_trained'] = True
                st.session_state['rotation_pretrained'] = True

                progress_bar.empty()
                status_text.empty()
                st.session_state['rotation_training'] = False
                st.rerun()

        # ---- 结果显示 ----
        if trained:
            st.markdown("### 📊 训练曲线")
            fig_curve = create_loss_plot(
                history['train_losses'], history['val_accs'],
                title='Rotation Prediction Training',
                y1_label='Loss', y2_label='Accuracy'
            )
            st.plotly_chart(fig_curve, width='stretch')

            # 拿一批验证数据展示
            val_loader = get_rotation_dataloader(batch_size=8, train=False)
            images, labels = next(iter(val_loader))
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            model.eval()
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

            st.markdown("### 🔍 预测示例")
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(2.5, 2.5))
                    ax.imshow(tensor_to_numpy(denormalize(images[i])), cmap='gray')
                    pred_angle = ANGLES[preds[i].item()]
                    true_angle = ANGLES[labels[i].item()]
                    correct = preds[i].item() == labels[i].item()
                    color = 'green' if correct else 'red'
                    ax.set_title(f'Pred: {pred_angle}\nTrue: {true_angle}',
                                color=color, fontsize=10)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)

        # ---- 上传图片测试 ----
        if uploaded is not None and trained:
            st.markdown("---")
            st.markdown("### 🧪 用户图片测试")

            img_tensor = process_uploaded_image(uploaded).to(DEVICE)

            # 随机旋转用户图片
            rand_angle = np.random.randint(0, 4)
            rotated = torch.rot90(img_tensor.squeeze(0), rand_angle, dims=[0, 1]).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(rotated)
                _, pred = torch.max(output, 1)

            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(3, 3))
                ax1.imshow(tensor_to_numpy(denormalize(img_tensor)), cmap='gray')
                ax1.set_title('Uploaded Image')
                ax1.axis('off')
                st.pyplot(fig1)
                plt.close(fig1)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(3, 3))
                ax2.imshow(tensor_to_numpy(denormalize(rotated)), cmap='gray')
                correct = pred.item() == rand_angle
                color = 'green' if correct else 'red'
                ax2.set_title(f'Rotated {ANGLES[rand_angle]} → Pred: {ANGLES[pred.item()]}',
                             color=color)
                ax2.axis('off')
                st.pyplot(fig2)
                plt.close(fig2)

            if correct:
                st.success(f"✅ 预测正确！模型识别出这是 {ANGLES[rand_angle]} 旋转")
            else:
                st.error(f"❌ 预测错误！真实: {ANGLES[rand_angle]}, 预测: {ANGLES[pred.item()]}")


# ============================================================
# 页面：MAE Reconstruction
# ============================================================

def page_mae_reconstruction():
    """MAE 遮挡自编码器页面"""
    st.markdown('<p class="main-header">🧩 MAE Reconstruction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Masked Autoencoder：从部分遮挡的图像重建原始图像</p>',
        unsafe_allow_html=True
    )

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### ⚙️ 设置")

        mask_ratio = st.selectbox("遮挡比例", [0.25, 0.5, 0.75],
                                  index=1, key="mae_mask_ratio",
                                  format_func=lambda x: f"{int(x*100)}%")
        epochs = st.slider("训练 Epochs", 1, 20, 5, key="mae_epochs")
        lr = st.selectbox("学习率", [0.01, 0.005, 0.001, 0.0005, 0.0001],
                          index=2, key="mae_lr")

        # 加载/获取模型
        model, history, pretrained = load_or_get_mae(mask_ratio)

        btn_train = st.button("🚀 开始训练", type="primary", use_container_width=True,
                              disabled=st.session_state.get('mae_training', False))

        # 状态
        if pretrained or st.session_state.get(f'mae_trained_{mask_ratio}', False):
            st.markdown('<div class="success-box">✅ 模型已训练</div>', unsafe_allow_html=True)
            if history:
                st.metric("最终验证 Loss", f"{history['val_losses'][-1]:.6f}")
        else:
            st.info("💡 点击「开始训练」训练此遮挡比例的模型")

        st.markdown("---")
        st.markdown("### 🖼️ 上传图片测试")
        uploaded = st.file_uploader("上传 PNG/JPG 图片", type=['png', 'jpg', 'jpeg'],
                                    key="mae_upload")

        # 测试用遮挡比例
        test_mask_ratio = st.slider("测试遮挡比例", 0.1, 0.9, mask_ratio, 0.05,
                                    key="mae_test_ratio")

    with col_main:
        trained = pretrained or st.session_state.get(f'mae_trained_{mask_ratio}', False)

        if not trained:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 任务说明</h4>
            <strong>Masked Autoencoder (MAE)</strong> 是自监督学习的经典方法：<br>
            1. 随机遮挡图像的一部分<br>
            2. 编码器将遮挡图像压缩为隐空间表示<br>
            3. 解码器从隐空间重建完整图像<br>
            4. 通过最小化重建误差（MSE Loss）来训练
            </div>
            """, unsafe_allow_html=True)

        # ---- 训练 ----
        if btn_train:
            st.session_state['mae_training'] = True
            with st.spinner(f"正在 {DEVICE} 上训练 MAE (遮挡 {int(mask_ratio*100)}%)..."):
                train_loader = get_mae_dataloader(batch_size=64, train=True, mask_ratio=mask_ratio)
                val_loader = get_mae_dataloader(batch_size=64, train=False, mask_ratio=mask_ratio)

                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_placeholder = st.empty()
                val_loss_placeholder = st.empty()

                def progress_callback(epoch, train_loss, val_loss, m):
                    pct = epoch / epochs
                    progress_bar.progress(pct)
                    status_text.text(f"Epoch {epoch}/{epochs}")
                    loss_placeholder.metric("Train Loss", f"{train_loss:.6f}")
                    val_loss_placeholder.metric("Val Loss", f"{val_loss:.6f}")

                history = train_mae_model(
                    model, train_loader, val_loader, epochs=epochs, lr=lr,
                    device=DEVICE, progress_callback=progress_callback
                )

                key = f'mae_model_{mask_ratio}'
                hist_key = f'mae_history_{mask_ratio}'
                pretrained_key = f'mae_pretrained_{mask_ratio}'
                trained_key = f'mae_trained_{mask_ratio}'
                st.session_state[key] = model
                st.session_state[hist_key] = history
                st.session_state[pretrained_key] = True
                st.session_state[trained_key] = True

                progress_bar.empty()
                status_text.empty()
                st.session_state['mae_training'] = False
                st.rerun()

        # ---- 结果显示 ----
        if trained:
            st.markdown("### 📊 训练曲线")
            fig_curve = create_loss_plot(
                history['train_losses'], history['val_losses'],
                title=f'MAE Training (Mask {int(mask_ratio*100)}%)',
                y1_label='MSE Loss'
            )
            st.plotly_chart(fig_curve, width='stretch')

            # 可视化重建效果
            st.markdown("### 🔍 重建示例")
            val_loader = get_mae_dataloader(batch_size=8, train=False, mask_ratio=mask_ratio)
            masked_imgs, original_imgs = next(iter(val_loader))
            masked_imgs = masked_imgs.to(DEVICE)
            original_imgs = original_imgs.to(DEVICE)

            model.eval()
            with torch.no_grad():
                reconstructed = model(masked_imgs)

            # 展示 4 个样本
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
                    axes[0].imshow(tensor_to_numpy(denormalize(original_imgs[i])), cmap='gray')
                    axes[0].set_title('Original', fontsize=8)
                    axes[0].axis('off')
                    axes[1].imshow(tensor_to_numpy(denormalize(masked_imgs[i])), cmap='gray')
                    axes[1].set_title('Masked', fontsize=8)
                    axes[1].axis('off')
                    axes[2].imshow(tensor_to_numpy(denormalize(reconstructed[i])), cmap='gray')
                    axes[2].set_title('Reconstructed', fontsize=8)
                    axes[2].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            # 展示遮挡效果对比
            st.markdown("### 🎭 遮挡效果可视化")
            demo_img = original_imgs[0:1]
            ratios = [0.25, 0.5, 0.75]
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            axes[0].imshow(tensor_to_numpy(denormalize(demo_img[0])), cmap='gray')
            axes[0].set_title('Original', fontsize=10)
            axes[0].axis('off')
            for j, r in enumerate(ratios):
                masked_demo, _ = mask_image(demo_img[0].cpu(), r)
                axes[j+1].imshow(tensor_to_numpy(denormalize(masked_demo)), cmap='gray')
                axes[j+1].set_title(f'Masked {int(r*100)}%', fontsize=10)
                axes[j+1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---- 用户上传图片测试 ----
        if uploaded is not None and trained:
            st.markdown("---")
            st.markdown("### 🧪 用户图片测试")

            img_tensor = process_uploaded_image(uploaded).to(DEVICE)
            # 对原图反归一化后再归一化 (保持一致性)
            masked_img, _ = mask_image(img_tensor.squeeze(0).cpu(), test_mask_ratio)
            masked_img = masked_img.unsqueeze(0).to(DEVICE)

            model.eval()
            with torch.no_grad():
                reconstructed = model(masked_img)

            cols = st.columns(3)
            with cols[0]:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(tensor_to_numpy(denormalize(img_tensor)), cmap='gray')
                ax.set_title('Uploaded Image')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            with cols[1]:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(tensor_to_numpy(denormalize(masked_img)), cmap='gray')
                ax.set_title(f'Masked ({int(test_mask_ratio*100)}%)')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            with cols[2]:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(tensor_to_numpy(denormalize(reconstructed)), cmap='gray')
                ax.set_title('Reconstructed')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            # 计算重建误差
            recon_error = torch.nn.functional.mse_loss(reconstructed, img_tensor).item()
            st.metric("重建 MSE Loss", f"{recon_error:.6f}")


# ============================================================
# 页面：Comparison
# ============================================================

def page_comparison():
    """对比实验页面"""
    st.markdown('<p class="main-header">📊 Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">不同设置下的模型效果对比</p>', unsafe_allow_html=True)

    # ---- Tab 1: MAE 遮挡比例对比 ----
    tab1, tab2, tab3 = st.tabs([
        "🔲 MAE 遮挡比例对比",
        "🔄 Rotation 训练前后对比",
        "📋 汇总表格"
    ])

    with tab1:
        st.markdown("### 不同遮挡比例下的重建效果对比")
        st.markdown("比较 25%、50%、75% 三种遮挡比例下的 MAE 重建质量")

        ratios = [0.25, 0.5, 0.75]
        final_losses = {}
        trained_all = True

        for r in ratios:
            model, history, pretrained = load_or_get_mae(r)
            key = f'mae_trained_{r}'
            if pretrained or st.session_state.get(key, False):
                if history:
                    final_losses[f'{int(r*100)}%'] = history['val_losses'][-1]
            else:
                trained_all = False

        if not trained_all:
            st.warning("⚠️ 部分遮挡比例的模型尚未训练，请先在 MAE Reconstruction 页面训练所有遮挡比例。")
        else:
            col_chart, col_img = st.columns([1, 1])

            with col_chart:
                # 柱状图对比
                fig_bar = create_comparison_bar_chart(
                    list(final_losses.keys()), list(final_losses.values()),
                    title='MAE 最终验证 Loss 对比',
                    y_label='MSE Loss'
                )
                st.plotly_chart(fig_bar, width='stretch')

            with col_img:
                # 加载实际重建效果对比
                val_loader_25 = get_mae_dataloader(batch_size=1, train=False, mask_ratio=0.25)
                val_loader_50 = get_mae_dataloader(batch_size=1, train=False, mask_ratio=0.50)
                val_loader_75 = get_mae_dataloader(batch_size=1, train=False, mask_ratio=0.75)

                masked_25, orig_25 = next(iter(val_loader_25))
                masked_50, orig_50 = next(iter(val_loader_50))
                masked_75, orig_75 = next(iter(val_loader_75))

                model_25, _, _ = load_or_get_mae(0.25)
                model_50, _, _ = load_or_get_mae(0.50)
                model_75, _, _ = load_or_get_mae(0.75)

                models_list = [model_25, model_50, model_75]
                masked_list = [masked_25, masked_50, masked_75]

                results = []
                for i, r in enumerate(ratios):
                    m = models_list[i]
                    m.eval()
                    with torch.no_grad():
                        recon = m(masked_list[i].to(DEVICE))
                        results.append(recon.cpu())

                fig, axes = plt.subplots(3, 3, figsize=(9, 9))
                row_labels = ['25%', '50%', '75%']
                col_labels = ['Original', 'Masked', 'Reconstructed']

                for r_idx in range(3):
                    # Original
                    axes[r_idx, 0].imshow(
                        tensor_to_numpy(denormalize([orig_25, orig_50, orig_75][r_idx][0])),
                        cmap='gray'
                    )
                    axes[r_idx, 0].set_ylabel(f'Mask {row_labels[r_idx]}', fontsize=12, fontweight='bold')
                    if r_idx == 0:
                        axes[r_idx, 0].set_title('Original')
                    axes[r_idx, 0].axis('off')

                    # Masked
                    axes[r_idx, 1].imshow(
                        tensor_to_numpy(denormalize(masked_list[r_idx][0])), cmap='gray'
                    )
                    if r_idx == 0:
                        axes[r_idx, 1].set_title('Masked')
                    axes[r_idx, 1].axis('off')

                    # Reconstructed
                    axes[r_idx, 2].imshow(
                        tensor_to_numpy(denormalize(results[r_idx][0])), cmap='gray'
                    )
                    if r_idx == 0:
                        axes[r_idx, 2].set_title('Reconstructed')
                    axes[r_idx, 2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # 分析文字
            st.markdown("### 📝 分析")
            sorted_losses = sorted(final_losses.items(), key=lambda x: x[1])
            st.markdown(f"""
            - **遮挡比例越低，重建效果越好**：遮挡 {sorted_losses[0][0]} 时 Loss 最低
            - **遮挡 {list(final_losses.keys())[-1]} 最具挑战性**：模型需要从极少的信息中推断完整图像
            - 这体现了 MAE 的核心思想：通过高比例遮挡迫使模型学习图像的本质结构
            """)

    # ---- Tab 2: Rotation 训练前后对比 ----
    with tab2:
        st.markdown("### Rotation Prediction 训练前后对比")

        model = st.session_state['rotation_model']
        trained = st.session_state['rotation_trained']
        history = st.session_state['rotation_history']

        if not trained:
            st.warning("⚠️ Rotation 模型尚未训练，请先在 Rotation Prediction 页面训练模型。")
        else:
            val_loader = get_rotation_dataloader(batch_size=500, train=False)
            images, labels = next(iter(val_loader))

            # 创建未训练模型
            untrained_model = RotationCNN(num_classes=4).to(DEVICE)

            # 评估两个模型
            def evaluate_rotation(model, imgs, lbls):
                model.eval()
                with torch.no_grad():
                    outputs = model(imgs.to(DEVICE))
                    _, preds = torch.max(outputs, 1)
                    acc = (preds.cpu() == lbls).float().mean().item()
                return acc, preds.cpu()

            trained_acc, trained_preds = evaluate_rotation(model, images, labels)
            untrained_acc, untrained_preds = evaluate_rotation(untrained_model, images, labels)

            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{trained_acc*100:.1f}%</div>
                    <div class="metric-label">训练后准确率</div>
                </div>
                """, unsafe_allow_html=True)
            with col_metric2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{untrained_acc*100:.1f}%</div>
                    <div class="metric-label">未训练准确率</div>
                </div>
                """, unsafe_allow_html=True)
            with col_metric3:
                improvement = (trained_acc - untrained_acc) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">+{improvement:.1f}%</div>
                    <div class="metric-label">准确率提升</div>
                </div>
                """, unsafe_allow_html=True)

            # 柱状图
            fig_bar = create_comparison_bar_chart(
                ['Untrained', 'Trained'],
                [untrained_acc, trained_acc],
                title='Rotation Prediction Accuracy',
                y_label='Accuracy'
            )
            st.plotly_chart(fig_bar, width='stretch')

            # 训练曲线
            if history:
                st.markdown("### 训练过程曲线")
                fig_curve = create_loss_plot(
                    history['train_losses'], history['val_accs'],
                    title='Training Progress',
                    y1_label='Loss', y2_label='Accuracy'
                )
                st.plotly_chart(fig_curve, width='stretch')

    # ---- Tab 3: 汇总表格 ----
    with tab3:
        st.markdown("### 📋 实验汇总")

        # MAE 结果汇总
        mae_data = []
        for r in [0.25, 0.5, 0.75]:
            model, history, pretrained = load_or_get_mae(r)
            key = f'mae_trained_{r}'
            if pretrained or st.session_state.get(key, False):
                if history:
                    mae_data.append({
                        'Model': f'MAE ({int(r*100)}% mask)',
                        'Final Train Loss': f"{history['train_losses'][-1]:.6f}",
                        'Final Val Loss': f"{history['val_losses'][-1]:.6f}",
                        'Epochs': len(history['train_losses']),
                    })
            else:
                mae_data.append({
                    'Model': f'MAE ({int(r*100)}% mask)',
                    'Final Train Loss': 'N/A',
                    'Final Val Loss': 'N/A',
                    'Epochs': 'N/A',
                })

        # Rotation 结果
        rot_model = st.session_state['rotation_model']
        rot_trained = st.session_state['rotation_trained']
        rot_history = st.session_state['rotation_history']

        rot_data = []
        if rot_trained and rot_history:
            rot_data.append({
                'Model': 'Rotation CNN',
                'Final Train Loss': f"{rot_history['train_losses'][-1]:.4f}",
                'Final Val Accuracy': f"{rot_history['val_accs'][-1]*100:.1f}%",
                'Epochs': len(rot_history['train_losses']),
            })
        else:
            rot_data.append({
                'Model': 'Rotation CNN',
                'Final Train Loss': 'N/A',
                'Final Val Accuracy': 'N/A',
                'Epochs': 'N/A',
            })

        st.markdown("#### MAE 模型汇总")
        st.table(mae_data)

        st.markdown("#### Rotation Prediction 模型汇总")
        st.table(rot_data)

        st.markdown("""
        ### 💡 要点总结
        - **自监督学习** 通过设计巧妙的 pretext task（如旋转预测、遮挡重建），无需人工标注即可学习有意义的特征表示
        - **Rotation Prediction** 是一个简单的判别式自监督任务，CNN 可以高效学习
        - **MAE** 通过高比例遮挡迫使模型理解图像结构，遮挡比例越高任务越难
        """)


# ============================================================
# 页面：About
# ============================================================

def page_about():
    """项目介绍页面"""
    st.markdown('<p class="main-header">📖 About This Project</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        ## 自监督学习（Self-Supervised Learning）教学项目

        ### 🎯 项目目标
        本项目通过两个经典的自监督学习任务，帮助理解 SSL 的核心思想：
        无需人工标注，通过设计 pretext task 从数据本身生成监督信号。

        ### 🧩 模块概览

        #### 1. Rotation Prediction（旋转预测）
        - **任务**：预测图像被旋转的角度（0°, 90°, 180°, 270°）
        - **方法**：对图像随机旋转，以旋转角度作为伪标签训练分类器
        - **模型**：简单 CNN（2 层卷积 + 2 层全连接）
        - **数据集**：MNIST（28×28 灰度手写数字）

        #### 2. MAE（Masked Autoencoder）
        - **任务**：从部分遮挡的图像重建完整图像
        - **方法**：随机遮挡图像 patch，用卷积自编码器重建原图
        - **模型**：卷积 Autoencoder（Encoder-Decoder 结构）
        - **数据集**：MNIST

        ### 🔬 关键发现
        1. **旋转预测准确率可达 90%+**：即使没有语义标签，CNN 也能学会识别旋转
        2. **遮挡比例越高，重建越难**：75% 遮挡的 Loss 明显高于 25%
        3. **自监督预训练是有效的**：学到的特征可以迁移到下游任务

        ### 🛠️ 技术栈
        - **深度学习框架**：PyTorch
        - **Web 框架**：Streamlit
        - **可视化**：Matplotlib + Plotly
        - **数据集**：MNIST（torchvision）

        ### 📚 参考文献
        - Gidaris et al., "Unsupervised Representation Learning by Predicting Image Rotations", ICLR 2018
        - He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
        - Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
        """)

    with col_right:
        st.markdown("### 🏗️ 项目结构")
        st.code("""
hw7/
├── app.py                 # 主程序
├── pretrain.py            # 预训练脚本
├── requirements.txt       # 依赖
├── README.md             # 说明文档
├── models/
│   ├── __init__.py
│   ├── rotation_cnn.py   # 旋转预测模型
│   └── mae.py            # MAE 模型
├── utils/
│   ├── __init__.py
│   ├── data_utils.py     # 数据处理
│   └── visualization.py  # 可视化工具
└── pretrained/           # 预训练模型
        """, language=None)

        st.markdown("### 🚀 快速启动")
        st.code("""
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行预训练（可选，首次运行 app 时会自动提示）
python pretrain.py

# 3. 启动 Web 应用
streamlit run app.py
        """, language='bash')


# ============================================================
# 主程序入口
# ============================================================

def main():
    init_session_state()
    inject_css()

    # ---- 侧边栏 ----
    with st.sidebar:
        st.markdown("## 🧠 自监督学习")
        st.markdown("---")

        page = st.radio(
            "📂 导航",
            ["🔄 Rotation Prediction", "🧩 MAE Reconstruction", "📊 Comparison", "📖 About Project"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown(f"⚡ 运行设备：**{DEVICE.upper()}**")
        st.markdown("---")

        # 模型状态概览
        st.markdown("### 📊 模型状态")

        rot_trained = st.session_state.get('rotation_trained', False)
        st.markdown(f"- Rotation: {'✅' if rot_trained else '⬜'}")

        for r in [0.25, 0.5, 0.75]:
            _, _, pretrained = load_or_get_mae(r)
            trained = pretrained or st.session_state.get(f'mae_trained_{r}', False)
            st.markdown(f"- MAE {int(r*100)}%: {'✅' if trained else '⬜'}")

        st.markdown("---")
        st.caption("© 2025 Self-Supervised Learning Demo")
        st.caption("Built with PyTorch + Streamlit")

    # ---- 页面路由 ----
    if "Rotation" in page:
        page_rotation_prediction()
    elif "MAE" in page:
        page_mae_reconstruction()
    elif "Comparison" in page:
        page_comparison()
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
