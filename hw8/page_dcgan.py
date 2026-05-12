"""
页面 3: DCGAN 图像生成
- 训练一个轻量 DCGAN 在 MNIST 上
- 展示随机噪声、生成图像网格
- 判别器分数
- Generator Loss & Discriminator Loss 曲线
"""

import streamlit as st
import font_setup
import torch
import matplotlib.pyplot as plt
import numpy as np

from data_utils import get_mnist_loaders
from model_defs import DCGANGenerator, DCGANDiscriminator
from train_utils import (
    get_device,
    train_dcgan,
    save_model,
    load_model,
    model_exists,
)


def show_dcgan_page():
    """DCGAN 页面"""
    st.title("🎨 DCGAN 图像生成")
    st.markdown("""
    **DCGAN (Deep Convolutional GAN)** 使用卷积神经网络构建生成器和判别器：
    - **生成器 (Generator)**：将随机噪声映射为逼真的 MNIST 风格图像
    - **判别器 (Discriminator)**：区分真实 MNIST 图像与生成的假图像
    - 两者在对抗训练中相互博弈，生成器逐渐学会生成更真实的图像
    """)

    device = get_device()
    st.info(f"💻 当前运行设备: **{device}**")

    # ---- 侧边参数 ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ DCGAN 参数")

    col_p1, col_p2, col_p3 = st.sidebar.columns(3)
    with col_p1:
        epochs = st.number_input("Epochs", min_value=5, max_value=50, value=10, step=5)
    with col_p2:
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
    with col_p3:
        noise_dim = st.selectbox("Noise Dim", [64, 100, 128], index=1)

    seed = st.sidebar.number_input("随机种子 (Seed)", min_value=0, max_value=9999, value=42, step=1)

    train_btn = st.sidebar.button("🚀 开始训练 DCGAN", type="primary", width="stretch")

    # ---- 初始化 session state ----
    if "dcgan_g" not in st.session_state:
        st.session_state.dcgan_g = None
    if "dcgan_d" not in st.session_state:
        st.session_state.dcgan_d = None
    if "dcgan_g_losses" not in st.session_state:
        st.session_state.dcgan_g_losses = None
    if "dcgan_d_losses" not in st.session_state:
        st.session_state.dcgan_d_losses = None
    if "dcgan_noise_dim" not in st.session_state:
        st.session_state.dcgan_noise_dim = noise_dim

    # ---- 训练 ----
    if train_btn:
        torch.manual_seed(seed)
        np.random.seed(seed)

        with st.status("⏳ 正在训练 DCGAN...", expanded=True) as status:
            st.write("📦 加载 MNIST 数据...")
            train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

            generator = DCGANGenerator(noise_dim=noise_dim)
            discriminator = DCGANDiscriminator()

            progress_bar = st.progress(0, "DCGAN 训练中...")

            def dcgan_callback(epoch, total, g_loss, d_loss):
                progress_bar.progress(
                    epoch / total,
                    f"Epoch {epoch}/{total} - G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}"
                )

            generator, discriminator, g_losses, d_losses = train_dcgan(
                generator, discriminator, train_loader,
                epochs=epochs, device=device, progress_callback=dcgan_callback
            )

            save_model(generator, f"dcgan_g_nd{noise_dim}_ep{epochs}")
            save_model(discriminator, f"dcgan_d_nd{noise_dim}_ep{epochs}")

            st.session_state.dcgan_g = generator
            st.session_state.dcgan_d = discriminator
            st.session_state.dcgan_g_losses = g_losses
            st.session_state.dcgan_d_losses = d_losses
            st.session_state.dcgan_noise_dim = noise_dim

            status.update(label="✅ DCGAN 训练完成！", state="complete", expanded=False)

    # ---- 尝试加载已有模型 ----
    if st.session_state.dcgan_g is None:
        best_ep = None
        for ep in [50, 30, 20, 10, 5]:
            if model_exists(f"dcgan_g_nd{noise_dim}_ep{ep}"):
                best_ep = ep
                break
        if best_ep:
            gen = DCGANGenerator(noise_dim=noise_dim)
            disc = DCGANDiscriminator()
            if load_model(gen, f"dcgan_g_nd{noise_dim}_ep{best_ep}", device) and \
               load_model(disc, f"dcgan_d_nd{noise_dim}_ep{best_ep}", device):
                st.session_state.dcgan_g = gen
                st.session_state.dcgan_d = disc
                st.session_state.dcgan_noise_dim = noise_dim
                st.sidebar.success(f"✅ 已加载预训练 DCGAN（epochs={best_ep}）")

    # ---- 结果展示 ----
    generator = st.session_state.dcgan_g
    discriminator = st.session_state.dcgan_d

    if generator is not None and discriminator is not None:
        generator.eval()
        discriminator.eval()
        current_noise_dim = st.session_state.dcgan_noise_dim

        # ==================== 生成图像网格 ====================
        st.subheader("🖼️ 生成图像网格")

        n_gen = st.slider("生成图像数量", min_value=16, max_value=64, value=32, step=16)

        if st.button("🎲 重新采样生成", type="primary", key="resample_gan"):
            st.rerun()

        # 使用 session_state 中的固定噪声或重新生成
        gen_key = f"dcgan_noise_{n_gen}_{current_noise_dim}"
        if gen_key not in st.session_state or st.button("🔄 刷新噪声"):
            torch.manual_seed(np.random.randint(0, 99999))
            st.session_state[gen_key] = torch.randn(n_gen, current_noise_dim)

        with torch.no_grad():
            noise = st.session_state[gen_key].to(device)
            fake_imgs = generator(noise).cpu()

        # 显示网格
        cols = 8
        rows = (n_gen + cols - 1) // cols
        fig_gen, axes_gen = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
        axes_gen = axes_gen.flatten() if hasattr(axes_gen, "flatten") else [axes_gen]

        for i in range(n_gen):
            img = (fake_imgs[i].squeeze().numpy() + 1) / 2  # Tanh [-1,1] -> [0,1]
            img = np.clip(img, 0, 1)
            axes_gen[i].imshow(img, cmap="gray")
            axes_gen[i].axis("off")
        for i in range(n_gen, len(axes_gen)):
            axes_gen[i].axis("off")

        plt.suptitle("DCGAN Generated MNIST-style Images", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_gen)
        plt.close()

        # ==================== 噪声向量可视化 ====================
        st.subheader("🔢 随机噪声向量示例")
        st.markdown("展示部分噪声向量的数值（前 10 维），帮助理解生成器的输入。")

        noise_sample = st.session_state[gen_key][:4]  # 前 4 个
        fig_noise, ax_noise = plt.subplots(figsize=(10, 3))
        im = ax_noise.imshow(noise_sample[:, :16].numpy(), cmap="coolwarm", aspect="auto")
        ax_noise.set_xlabel("Noise Dimension")
        ax_noise.set_ylabel("Sample Index")
        ax_noise.set_title("Noise Vector (first 16 dims)")
        plt.colorbar(im, ax=ax_noise)
        st.pyplot(fig_noise)
        plt.close()

        # ==================== 判别器分数 ====================
        st.subheader("📊 判别器分数")

        st.markdown("判别器对真实 MNIST 样本和生成样本分别打分（越接近 1 表示越真实）。")

        col_score1, col_score2 = st.columns(2)

        with col_score1:
            if st.button("📈 评估判别器", type="primary"):
                _, test_loader = get_mnist_loaders(batch_size=batch_size)
                real_batch, _ = next(iter(test_loader))

                with torch.no_grad():
                    real_batch = real_batch.to(device)
                    real_scores = discriminator(real_batch).cpu().numpy().flatten()

                    noise_eval = torch.randn(real_batch.size(0), current_noise_dim, device=device)
                    fake_eval = generator(noise_eval)
                    fake_scores = discriminator(fake_eval).cpu().numpy().flatten()

                st.session_state.dcgan_real_scores = real_scores
                st.session_state.dcgan_fake_scores = fake_scores

        if "dcgan_real_scores" in st.session_state:
            real_scores = st.session_state.dcgan_real_scores
            fake_scores = st.session_state.dcgan_fake_scores

            fig_score, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(12, 4))

            ax_s1.hist(real_scores, bins=20, alpha=0.7, color="green", label=f"Real (avg={real_scores.mean():.3f})")
            ax_s1.hist(fake_scores, bins=20, alpha=0.7, color="red", label=f"Fake (avg={fake_scores.mean():.3f})")
            ax_s1.set_xlabel("Discriminator Score")
            ax_s1.set_ylabel("Count")
            ax_s1.set_title("Discriminator Score Distribution")
            ax_s1.legend()
            ax_s1.grid(True, alpha=0.3)

            categories = ["Real", "Fake"]
            means = [real_scores.mean(), fake_scores.mean()]
            ax_s2.bar(categories, means, color=["green", "red"], alpha=0.7)
            ax_s2.set_ylabel("Average Score")
            ax_s2.set_title("Discriminator Avg Score Comparison")
            for i, v in enumerate(means):
                ax_s2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig_score)
            plt.close()

            if real_scores.mean() > fake_scores.mean():
                st.success("✅ 判别器仍有区分能力——真实样本得分高于生成样本。")
            else:
                st.warning("⚠️ 判别器难以区分——生成器可能已经很强了！")

        # ==================== Loss 曲线 ====================
        st.subheader("📉 Generator Loss vs Discriminator Loss")

        g_losses = st.session_state.dcgan_g_losses
        d_losses = st.session_state.dcgan_d_losses

        if g_losses is not None and d_losses is not None:
            fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
            epochs_range = range(1, len(g_losses) + 1)
            ax_loss.plot(epochs_range, g_losses, "b-o", markersize=5, label="Generator Loss")
            ax_loss.plot(epochs_range, d_losses, "r-s", markersize=5, label="Discriminator Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss (BCE)")
            ax_loss.set_title("DCGAN Training Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
            plt.close()

            st.markdown("""
            > 💡 **GAN 训练的特点**：
            > - G Loss 和 D Loss 通常不会像普通分类任务那样稳定下降
            > - 理想情况下两者会达到某种**动态平衡**
            > - 如果 D Loss 趋近于 0，说明判别器太强，生成器学不到东西
            > - 如果 G Loss 远大于 D Loss，说明生成器难以欺骗判别器
            """)

    else:
        st.warning("⚠️ 尚未训练 DCGAN 模型。请在左侧边栏调整参数后点击「开始训练 DCGAN」。")
        st.markdown("""
        ### 💡 提示
        - DCGAN 训练较慢，建议先用 **5-10 epochs** 快速测试
        - **Noise Dim** 越大，生成多样性越高，但训练也更慢
        - 设置 **Seed** 可以复现训练结果
        """)
