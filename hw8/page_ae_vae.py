"""
页面 1: Autoencoder vs VAE 重构对比
- 训练按钮，支持调整 epoch、batch size、latent dimension
- 展示原始图像、AE 重构、VAE 重构
- 重构误差热力图
- 训练 loss 曲线
- MSE 对比
"""

import streamlit as st
import font_setup
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_utils import get_mnist_loaders
from model_defs import Autoencoder, VAE
from train_utils import (
    get_device,
    train_autoencoder,
    train_vae,
    save_model,
    load_model,
    model_exists,
)


def show_ae_vae_page():
    """Autoencoder vs VAE 页面主函数"""
    st.title("🔍 Autoencoder vs VAE 重构对比")
    st.markdown("""
    对比**普通自编码器（AE）** 与**变分自编码器（VAE）** 在 MNIST 上的重构效果。
    - **AE**：将图像压缩到潜空间后直接解码，目标是精确重构。
    - **VAE**：学习潜空间的概率分布 $q(z|x)$，通过 KL 散度正则化使潜空间更加平滑连续。
    """)

    device = get_device()
    st.info(f"💻 当前运行设备: **{device}**")

    # ---- 侧边参数 ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 训练参数")

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=30, value=5, step=1)
    with col2:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    with col3:
        latent_dim = st.selectbox("Latent Dim", [2, 8, 16, 32], index=0)

    # ---- 训练按钮 ----
    train_btn = st.sidebar.button("🚀 开始训练 AE & VAE", type="primary", width="stretch")

    # ---- 初始化/加载 session state ----
    if "ae_model" not in st.session_state:
        st.session_state.ae_model = None
    if "vae_model" not in st.session_state:
        st.session_state.vae_model = None
    if "ae_losses" not in st.session_state:
        st.session_state.ae_losses = None
    if "vae_losses" not in st.session_state:
        st.session_state.vae_losses = None
    if "test_images" not in st.session_state:
        st.session_state.test_images = None
    if "test_labels" not in st.session_state:
        st.session_state.test_labels = None

    # ---- 训练逻辑 ----
    if train_btn:
        with st.status("⏳ 正在训练模型...", expanded=True) as status:
            # 加载数据
            st.write("📦 加载 MNIST 数据...")
            train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
            test_imgs, test_lbls = next(iter(test_loader))
            st.session_state.test_images = test_imgs
            st.session_state.test_labels = test_lbls

            # 训练 AE
            st.write("🔧 训练 Autoencoder...")
            ae = Autoencoder(latent_dim=latent_dim)
            progress_bar = st.progress(0, "AE 训练中...")

            def ae_callback(epoch, total, loss):
                progress_bar.progress(epoch / total, f"AE Epoch {epoch}/{total} - Loss: {loss:.4f}")

            ae, ae_losses = train_autoencoder(
                ae, train_loader, epochs=epochs, device=device, progress_callback=ae_callback
            )
            save_model(ae, f"ae_ld{latent_dim}_ep{epochs}")
            st.session_state.ae_model = ae
            st.session_state.ae_losses = ae_losses

            # 训练 VAE
            st.write("🔧 训练 VAE...")
            vae = VAE(latent_dim=latent_dim)
            progress_bar2 = st.progress(0, "VAE 训练中...")

            def vae_callback(epoch, total, loss):
                progress_bar2.progress(epoch / total, f"VAE Epoch {epoch}/{total} - Loss: {loss:.4f}")

            vae, vae_losses = train_vae(
                vae, train_loader, epochs=epochs, device=device, progress_callback=vae_callback
            )
            save_model(vae, f"vae_ld{latent_dim}_ep{epochs}")
            st.session_state.vae_model = vae
            st.session_state.vae_losses = vae_losses

            status.update(label="✅ 训练完成！", state="complete", expanded=False)

    # ---- 尝试加载已有模型 ----
    if st.session_state.ae_model is None and model_exists(f"ae_ld{latent_dim}_ep{epochs}"):
        ae = Autoencoder(latent_dim=latent_dim)
        if load_model(ae, f"ae_ld{latent_dim}_ep{epochs}", device):
            st.session_state.ae_model = ae
    if st.session_state.vae_model is None and model_exists(f"vae_ld{latent_dim}_ep{epochs}"):
        vae = VAE(latent_dim=latent_dim)
        if load_model(vae, f"vae_ld{latent_dim}_ep{epochs}", device):
            st.session_state.vae_model = vae

    # ---- 加载测试数据 ----
    if st.session_state.test_images is None:
        _, test_loader = get_mnist_loaders(batch_size=batch_size)
        test_imgs, test_lbls = next(iter(test_loader))
        st.session_state.test_images = test_imgs
        st.session_state.test_labels = test_lbls

    # ---- 结果展示 ----
    ae_model = st.session_state.ae_model
    vae_model = st.session_state.vae_model

    if ae_model is not None and vae_model is not None:
        ae_model.eval()
        vae_model.eval()
        test_imgs = st.session_state.test_images
        test_lbls = st.session_state.test_labels

        with torch.no_grad():
            test_imgs_dev = test_imgs.to(device)
            ae_recon, _ = ae_model(test_imgs_dev)
            vae_recon, _, _, _ = vae_model(test_imgs_dev)

            # 移回 CPU
            ae_recon = ae_recon.cpu()
            vae_recon = vae_recon.cpu()

            # 计算 MSE
            ae_mse = torch.mean((ae_recon - test_imgs) ** 2, dim=[1, 2, 3]).numpy()
            vae_mse = torch.mean((vae_recon - test_imgs) ** 2, dim=[1, 2, 3]).numpy()

        # ---- 制图函数 ----
        def img_to_numpy(tensor):
            """Tensor (1, H, W) -> numpy (H, W)"""
            return tensor.squeeze().numpy()

        # ==================== 1. 重构结果对比 ====================
        st.subheader("📸 重构结果对比")
        st.markdown("每行依次为：**原始图像** → **AE 重构** → **VAE 重构** → **AE 误差热力图** → **VAE 误差热力图**")

        n_display = min(8, test_imgs.size(0))
        fig, axes = plt.subplots(n_display, 5, figsize=(14, 2.6 * n_display))

        for i in range(n_display):
            orig = img_to_numpy(test_imgs[i])
            ae_r = img_to_numpy(ae_recon[i])
            vae_r = img_to_numpy(vae_recon[i])

            # 误差
            ae_diff = np.abs(orig - ae_r)
            vae_diff = np.abs(orig - vae_r)

            axes[i, 0].imshow(orig, cmap="gray")
            axes[i, 0].set_title(f"原始 ({test_lbls[i].item()})" if i == 0 else "")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(ae_r, cmap="gray")
            axes[i, 1].set_title("AE 重构" if i == 0 else "")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(vae_r, cmap="gray")
            axes[i, 2].set_title("VAE 重构" if i == 0 else "")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(ae_diff, cmap="hot")
            axes[i, 3].set_title("AE 误差" if i == 0 else "")
            axes[i, 3].axis("off")

            axes[i, 4].imshow(vae_diff, cmap="hot")
            axes[i, 4].set_title("VAE 误差" if i == 0 else "")
            axes[i, 4].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ==================== 2. Loss 曲线 ====================
        st.subheader("📉 训练 Loss 曲线")

        ae_losses = st.session_state.ae_losses
        vae_losses = st.session_state.vae_losses

        if ae_losses is not None and vae_losses is not None:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # AE Loss
            ax1.plot(range(1, len(ae_losses) + 1), ae_losses, "b-o", markersize=5, label="AE Loss (MSE)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Autoencoder Training Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # VAE Loss
            vae_total = [l[0] for l in vae_losses]
            vae_recon_l = [l[1] for l in vae_losses]
            vae_kl_l = [l[2] for l in vae_losses]
            ax2.plot(range(1, len(vae_total) + 1), vae_total, "r-o", markersize=5, label="VAE Total Loss")
            ax2.plot(range(1, len(vae_recon_l) + 1), vae_recon_l, "g--s", markersize=4, label="Recon Loss")
            ax2.plot(range(1, len(vae_kl_l) + 1), vae_kl_l, "m--^", markersize=4, label="KL Divergence")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("VAE Training Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ==================== 3. MSE 对比 ====================
        st.subheader("📊 重构误差 (MSE) 对比")

        fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))

        # 直方图
        ax3.hist(ae_mse, bins=30, alpha=0.6, label=f"AE (avg={ae_mse.mean():.4f})", color="blue")
        ax3.hist(vae_mse, bins=30, alpha=0.6, label=f"VAE (avg={vae_mse.mean():.4f})", color="red")
        ax3.set_xlabel("MSE")
        ax3.set_ylabel("频次")
        ax3.set_title("重构误差分布")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 箱线图
        ax4.boxplot([ae_mse, vae_mse], labels=["AE", "VAE"], patch_artist=True,
                     boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red"))
        ax4.set_ylabel("MSE")
        ax4.set_title("MSE 箱线图对比")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ---- 文字总结 ----
        st.subheader("📝 分析总结")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("AE 平均 MSE", f"{ae_mse.mean():.5f}")
        with col_b:
            st.metric("VAE 平均 MSE", f"{vae_mse.mean():.5f}")

        st.markdown(f"""
        - **AE** 的重构通常更精确，因为它直接学习确定性映射，目标是纯重构。
        - **VAE** 的潜空间经过 KL 正则化，重构可能稍模糊，但潜空间更加**平滑连续**，
          这对生成任务（如潜空间插值）非常有利。
        - 从误差热力图中可以看到 VAE 的误差分布更加均匀，而 AE 可能在细节处更精确。
        """)

    else:
        st.warning("⚠️ 尚未训练模型。请在左侧边栏调整参数后点击「开始训练 AE & VAE」。")
        st.markdown("""
        ### 💡 提示
        如果训练时间较长，可以：
        - 减少 **Epochs**（如 3-5）
        - 增大 **Batch Size**（如 128）
        - 使用较小的 **Latent Dim**（如 2）
        """)
