"""
页面 5: About Project - 项目介绍
"""

import streamlit as st


def show_about_page():
    """关于页面"""
    st.title("📖 关于本项目")

    st.markdown("""
    ## 生成模型与潜空间可视化

    本项目是一个基于 **Streamlit** 的交互式深度学习教学演示平台，涵盖了生成模型领域的核心技术。
    """)

    # ---- 项目概述 ----
    st.markdown("---")
    st.subheader("🎯 项目目标")

    st.markdown("""
    通过可视化和交互式操作，帮助理解以下生成模型的核心概念：

    1. **自编码器 (Autoencoder)**：学习数据的低维表示并重建
    2. **变分自编码器 (VAE)**：学习连续平滑的潜空间概率分布
    3. **生成对抗网络 (DCGAN)**：通过对抗训练生成逼真图像
    4. **扩散模型 (Diffusion Models)**：从文本描述生成图像
    """)

    # ---- 技术栈 ----
    st.markdown("---")
    st.subheader("🛠️ 技术栈")

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        st.markdown("""
        **深度学习框架**
        - 🐍 Python 3.9+
        - 🔥 PyTorch
        - 👁️ torchvision
        """)

    with col_t2:
        st.markdown("""
        **可视化与界面**
        - 🎈 Streamlit
        - 📊 Matplotlib
        - 📈 Plotly
        - 🖼️ Pillow
        """)

    with col_t3:
        st.markdown("""
        **生成模型库**
        - 🤗 Diffusers
        - 🤗 Transformers
        """)

    # ---- 模块说明 ----
    st.markdown("---")
    st.subheader("📦 项目模块")

    with st.expander("🔍 Autoencoder vs VAE 重构对比", expanded=False):
        st.markdown("""
        - 使用 MNIST 数据集训练普通 AE 和 VAE
        - 对比两种模型的重构质量
        - 展示重构误差热力图
        - 对比训练 Loss 曲线和 MSE 分布
        - **教学重点**：理解确定性编码 vs 概率编码的区别
        """)

    with st.expander("🧭 VAE 潜空间可视化与交互", expanded=False):
        st.markdown("""
        - 将测试集编码到二维潜空间并可视化
        - 交互式探索潜空间的语义结构
        - 线性插值展示潜空间的连续性
        - 随机采样生成新图像
        - **教学重点**：感受潜空间的平滑性和连续性
        """)

    with st.expander("🎨 DCGAN 图像生成", expanded=False):
        st.markdown("""
        - 训练轻量 DCGAN 生成 MNIST 风格图像
        - 可视化随机噪声和生成结果
        - 展示判别器对真假样本的评分
        - 观察 GAN 训练的对抗过程
        - **教学重点**：理解生成对抗训练的博弈过程
        """)

    with st.expander("🖼️ 文本到图像生成", expanded=False):
        st.markdown("""
        - 加载 Stable Diffusion Turbo 轻量模型
        - 支持调整 prompt、guidance scale、steps 等参数
        - 展示扩散模型的文本到图像生成能力
        - **教学重点**：了解扩散模型的去噪生成过程
        """)

    # ---- 运行说明 ----
    st.markdown("---")
    st.subheader("🚀 运行说明")

    st.code("""
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行 Streamlit 应用
streamlit run app.py

# 3. 在浏览器中打开 http://localhost:8501
    """, language="bash")

    st.markdown("""
    > ⚠️ **注意**：
    > - 首次训练模型会较慢（CPU），建议先用较小的 epoch（3-5）测试
    > - 训练好的模型会自动保存到 `saved_models/` 目录
    > - Text-to-Image 页面默认使用演示模式，加载真实模型需要 5-7GB 磁盘空间
    """)

    # ---- 项目结构 ----
    st.markdown("---")
    st.subheader("📁 项目结构")

    st.code("""
hw8/
├── app.py                    # Streamlit 主入口
├── model_defs.py             # 模型定义 (AE, VAE, DCGAN)
├── train_utils.py            # 训练工具与模型保存
├── data_utils.py             # 数据加载工具
├── page_ae_vae.py            # 页面1: AE vs VAE
├── page_latent.py            # 页面2: 潜空间探索
├── page_dcgan.py             # 页面3: DCGAN 生成
├── page_text2img.py          # 页面4: 文本到图像
├── page_about.py             # 页面5: 关于项目
├── saved_models/             # 训练好的模型文件
├── data/MNIST/raw/           # MNIST 数据集
└── requirements.txt          # 依赖列表
    """, language="text")

    st.markdown("---")
    st.markdown("""
    ### 👨‍💻 开发者
    深度学习课程作业 — 生成模型与潜空间可视化

    *Made with ❤️ using Streamlit, PyTorch & Diffusers*
    """)
