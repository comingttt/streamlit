"""
生成模型与潜空间可视化 — Streamlit 主入口
============================================
项目包含四个核心模块，通过侧边栏导航：
1. Autoencoder vs VAE 重构对比
2. Latent Space Explorer 潜空间探索
3. DCGAN Generator 图像生成
4. Text-to-Image 文本到图像生成
5. About Project 关于本项目
"""

import font_setup  # 必须在 matplotlib 使用前导入，配置中文字体

import streamlit as st

# 页面配置（必须是第一个 Streamlit 命令）
st.set_page_config(
    page_title="生成模型与潜空间可视化",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 导入各页面模块
from page_ae_vae import show_ae_vae_page
from page_latent import show_latent_explorer_page
from page_dcgan import show_dcgan_page
from page_text2img import show_text2img_page
from page_about import show_about_page


def main():
    """主函数：构建侧边栏导航和页面路由"""

    # ---- 侧边栏 ----
    with st.sidebar:
        st.title("🧠 生成模型可视化")
        st.markdown("---")

        # 导航菜单
        page = st.radio(
            "📋 导航菜单",
            options=[
                "🔍 Autoencoder vs VAE",
                "🧭 Latent Space Explorer",
                "🎨 DCGAN Generator",
                "🖼️ Text-to-Image",
                "📖 About Project",
            ],
        )

        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #888; font-size: 0.8em;'>
        PyTorch + Streamlit<br>
        生成模型课程作业
        </div>
        """, unsafe_allow_html=True)

    # ---- 页面路由 ----
    if page == "🔍 Autoencoder vs VAE":
        show_ae_vae_page()
    elif page == "🧭 Latent Space Explorer":
        show_latent_explorer_page()
    elif page == "🎨 DCGAN Generator":
        show_dcgan_page()
    elif page == "🖼️ Text-to-Image":
        show_text2img_page()
    elif page == "📖 About Project":
        show_about_page()


if __name__ == "__main__":
    main()
