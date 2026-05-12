"""
页面 2: VAE 潜空间可视化与交互
- 测试集编码到二维潜空间散点图（按类别着色）
- 点击潜空间位置生成图像
- 两样本间线性插值
- 随机采样生成
"""

import streamlit as st
import font_setup
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from data_utils import get_mnist_loaders, get_test_subset
from model_defs import VAE
from train_utils import get_device, load_model, model_exists


def show_latent_explorer_page():
    """潜空间探索页面"""
    st.title("🧭 VAE 潜空间可视化与交互")
    st.markdown("""
    将测试集图像通过 VAE 编码器映射到**二维潜空间**，探索潜空间的结构：
    - 📌 潜空间中相近的点对应语义相似的图像
    - 🔄 在潜空间中做线性插值，观察图像的平滑过渡
    - 🎲 从标准正态分布随机采样潜向量并生成新图像
    """)

    device = get_device()
    st.info(f"💻 当前运行设备: **{device}**")

    # ---- 加载/训练 VAE ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ VAE 设置")
    latent_dim = st.sidebar.selectbox("Latent Dimension", [2], index=0, disabled=True,
                                       help="潜空间可视化需要使用 2 维")
    batch_size = st.sidebar.selectbox("Batch Size", [64, 128, 256], index=1)

    # 尝试加载已保存的 VAE 模型
    vae = None
    for ep in [5, 10, 15, 20]:
        if model_exists(f"vae_ld2_ep{ep}"):
            vae = VAE(latent_dim=2)
            if load_model(vae, f"vae_ld2_ep{ep}", device):
                st.sidebar.success(f"✅ 已加载预训练 VAE（epochs={ep}）")
                break

    if vae is None:
        # 尝试其他 latent dim 的模型
        all_loaded = False
        for ld in [2, 8, 16]:
            for ep in [5, 10, 15, 20]:
                if model_exists(f"vae_ld{ld}_ep{ep}"):
                    vae = VAE(latent_dim=ld)
                    if load_model(vae, f"vae_ld{ld}_ep{ep}", device):
                        st.sidebar.success(f"✅ 已加载预训练 VAE（latent_dim={ld}, epochs={ep}）")
                        st.sidebar.warning(f"⚠️ 当前模型潜空间为 {ld} 维，可视化将使用 PCA 降维到 2D")
                        all_loaded = True
                        break
            if all_loaded:
                break

    if vae is None:
        st.warning("⚠️ 未找到已训练的 VAE 模型，请先在「Autoencoder vs VAE」页面训练 VAE。")
        return

    vae.eval()

    # ---- 编码测试集 ----
    _, test_loader = get_mnist_loaders(batch_size=batch_size)
    test_imgs, test_lbls = get_test_subset(test_loader, n_samples=1000)

    @st.cache_data(show_spinner="正在编码测试集到潜空间...")
    def encode_test_set(_vae, _images, _labels, _device):
        """将测试集编码到潜空间"""
        _vae.eval()
        all_z = []
        all_labels = []
        bs = 256
        with torch.no_grad():
            for i in range(0, len(_images), bs):
                x = _images[i:i + bs].to(_device)
                mu, _ = _vae.encode(x)
                all_z.append(mu.cpu())
                all_labels.append(_labels[i:i + bs])
        z_all = torch.cat(all_z, dim=0)
        lbl_all = torch.cat(all_labels, dim=0)
        return z_all.numpy(), lbl_all.numpy()

    z_np, lbl_np = encode_test_set(vae, test_imgs, test_lbls, device)

    # 如果 latent_dim > 2，使用 PCA 降到 2D
    if z_np.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_np)
        st.info(f"📐 潜空间维度为 {z_np.shape[1]}，使用 PCA 降维到 2D 进行可视化。")
        # 对于解码，需要使用原始维度 —— 对 PCA 2D 坐标做逆变换
        def pca_to_full(z2d):
            return pca.inverse_transform(z2d)
    else:
        z_2d = z_np

        def pca_to_full(z2d):
            return z2d

    # ==================== 1. 潜空间散点图 ====================
    st.subheader("📍 潜空间散点图（按数字类别着色）")

    # 使用 Plotly 实现交互
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    digit_names = [str(i) for i in range(10)]

    fig = go.Figure()
    for digit in range(10):
        mask = lbl_np == digit
        fig.add_trace(go.Scatter(
            x=z_2d[mask, 0],
            y=z_2d[mask, 1],
            mode="markers",
            name=f"数字 {digit}",
            marker=dict(size=4, color=colors[digit], opacity=0.6),
            hovertemplate=f"数字 {digit}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="VAE 潜空间分布（鼠标悬停可查看坐标）",
        xaxis_title="Latent Dim 1",
        yaxis_title="Latent Dim 2",
        width=700,
        height=600,
        legend=dict(itemsizing="constant"),
        hovermode="closest",
    )

    # 捕获点击事件
    selected = st.plotly_chart(fig, key="latent_scatter", on_select="rerun", width="stretch")

    # ---- 处理点击：从潜空间生成图像 ----
    st.subheader("🎯 潜空间点 → 图像生成")

    col_manual, col_click = st.columns(2)

    with col_manual:
        st.markdown("**手动输入潜空间坐标**")

        # 检查是否有用户从散点图选中的待用坐标
        init_z1 = st.session_state.pop("pending_z1", 0.0)
        init_z2 = st.session_state.pop("pending_z2", 0.0)

        z1 = st.slider("Latent Dim 1",
                       float(z_2d[:, 0].min()) - 0.5,
                       float(z_2d[:, 0].max()) + 0.5,
                       init_z1, 0.05)
        z2 = st.slider("Latent Dim 2",
                       float(z_2d[:, 1].min()) - 0.5,
                       float(z_2d[:, 1].max()) + 0.5,
                       init_z2, 0.05)

    with col_click:
        st.markdown("**或从散点图选择点**")
        if selected and selected.get("selection", {}).get("points"):
            point = selected["selection"]["points"][0]
            click_z1 = point["x"]
            click_z2 = point["y"]
            st.info(f"已选择: ({click_z1:.3f}, {click_z2:.3f})")
            if st.button("使用选中坐标", key="use_selected"):
                st.session_state.pending_z1 = click_z1
                st.session_state.pending_z2 = click_z2
                st.rerun()

    # 生成图像
    gen_btn = st.button("🎨 生成图像", type="primary")
    if gen_btn:
        z_input = torch.tensor([[z1, z2]], dtype=torch.float32)
        if z_np.shape[1] > 2:
            z_input = torch.tensor(pca_to_full(np.array([[z1, z2]])), dtype=torch.float32)
        z_input = z_input.to(device)

        with torch.no_grad():
            generated = vae.decode(z_input).cpu().squeeze().numpy()

        fig_gen, ax_gen = plt.subplots(figsize=(3, 3))
        ax_gen.imshow(generated, cmap="gray")
        ax_gen.set_title(f"Generated @ ({z1:.2f}, {z2:.2f})")
        ax_gen.axis("off")
        st.pyplot(fig_gen)
        plt.close()

    # ==================== 2. 线性插值 ====================
    st.markdown("---")
    st.subheader("🔄 潜空间线性插值")

    st.markdown("""
    在潜空间中，对两个样本的潜向量做**线性插值**，观察生成图像从数字 A 到数字 B 的平滑过渡。
    这体现了 VAE 潜空间的**连续性和平滑性**。
    """)

    n_samples = len(test_imgs)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        idx1 = st.number_input("样本 A 索引", min_value=0, max_value=n_samples - 1, value=0)
    with col_i2:
        idx2 = st.number_input("样本 B 索引", min_value=0, max_value=n_samples - 1, value=n_samples - 1)

    n_interp = st.slider("插值步数", min_value=4, max_value=16, value=8)

    if st.button("🔄 执行线性插值", type="primary"):
        with torch.no_grad():
            x1 = test_imgs[idx1:idx1 + 1].to(device)
            x2 = test_imgs[idx2:idx2 + 1].to(device)

            mu1, _ = vae.encode(x1)
            mu2, _ = vae.encode(x2)

            alphas = np.linspace(0, 1, n_interp)
            interpolated = []
            for alpha in alphas:
                z_interp = mu1 * (1 - alpha) + mu2 * alpha
                img = vae.decode(z_interp).cpu().squeeze().numpy()
                interpolated.append(img)

        # 显示插值序列
        fig_interp, axes_interp = plt.subplots(1, n_interp + 2, figsize=(3 * (n_interp + 2), 3))

        # 原始图像 A
        axes_interp[0].imshow(test_imgs[idx1].squeeze().numpy(), cmap="gray")
        axes_interp[0].set_title(f"Original A\nDigit {test_lbls[idx1].item()}")
        axes_interp[0].axis("off")

        # 插值序列
        for j, img in enumerate(interpolated):
            axes_interp[j + 1].imshow(img, cmap="gray")
            axes_interp[j + 1].set_title(f"α={alphas[j]:.2f}")
            axes_interp[j + 1].axis("off")

        # 原始图像 B
        axes_interp[-1].imshow(test_imgs[idx2].squeeze().numpy(), cmap="gray")
        axes_interp[-1].set_title(f"Original B\nDigit {test_lbls[idx2].item()}")
        axes_interp[-1].axis("off")

        plt.tight_layout()
        st.pyplot(fig_interp)
        plt.close()

        st.markdown("""
        > 💡 **观察**：插值过程中，图像从数字 A 逐渐变形为数字 B。
        > 如果两个数字形态接近（如 3→8），过渡会很自然；
        > 如果差异较大（如 1→0），中间会出现模糊的混合形态。
        > 这正是 VAE 潜空间**连续且平滑**的体现！
        """)

    # ==================== 3. 随机采样 ====================
    st.markdown("---")
    st.subheader("🎲 随机采样生成")

    st.markdown("""
    从标准正态分布 $\\mathcal{N}(0, I)$ 中随机采样潜向量，通过解码器生成新图像。
    这模拟了 VAE 的**生成过程**——从先验分布中采样并解码。
    """)

    n_random = st.slider("随机采样数量", min_value=4, max_value=32, value=16)
    if st.button("🎲 随机采样生成", type="primary"):
        with torch.no_grad():
            if z_np.shape[1] > 2:
                # 使用 PCA 逆变换
                z_rand_2d = np.random.randn(n_random, 2) * 2  # 在 2D 空间采样
                z_rand = torch.tensor(pca_to_full(z_rand_2d), dtype=torch.float32).to(device)
            else:
                z_rand = torch.randn(n_random, latent_dim, device=device) * 2.0

            generated_imgs = vae.decode(z_rand).cpu()

        # 显示采样图像网格
        cols_grid = 8
        rows_grid = (n_random + cols_grid - 1) // cols_grid
        fig_rand, axes_rand = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 1.5, rows_grid * 1.5))
        axes_rand = axes_rand.flatten() if hasattr(axes_rand, "flatten") else [axes_rand]

        for i in range(n_random):
            axes_rand[i].imshow(generated_imgs[i].squeeze().numpy(), cmap="gray")
            axes_rand[i].axis("off")
        for i in range(n_random, len(axes_rand)):
            axes_rand[i].axis("off")

        plt.suptitle("Random Latent → Generated Images", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig_rand)
        plt.close()

        st.markdown("""
        > 💡 **观察**：从先验分布采样的图像质量取决于 VAE 的训练程度。
        > 训练良好的 VAE 生成的图像应该清晰可辨；
        > 如果图像模糊，可能是 KL 正则化权重过高或训练不足。
        """)

    # ==================== 知识卡片 ====================
    st.markdown("---")
    st.subheader("📚 知识卡片：VAE 潜空间")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("""
        **潜空间 (Latent Space)**
        - 数据压缩后的低维表示空间
        - VAE 学习将数据映射到平滑的概率分布
        - 相近的潜向量生成相似的图像
        """)
    with col_k2:
        st.markdown("""
        **为什么 VAE 的潜空间更平滑？**
        - KL 散度正则化迫使 $q(z|x)$ 接近 $\\mathcal{N}(0,I)$
        - 避免潜空间出现"空洞"
        - 任意采样的潜向量都能解码为合理图像
        """)
