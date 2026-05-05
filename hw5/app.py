"""
=============================================================================
计算机视觉教学项目 - Streamlit 主应用
=============================================================================
整合4个教学模块:
  1. HOG + Bag of Words + SVM 图像分类
  2. 神经网络反向传播演示
  3. CNN模型训练与测试
  4. ResNet预训练模型对比

运行方式:
    streamlit run app.py
=============================================================================
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# 必须在 import streamlit 之后、其他 st 调用之前设置页面配置
import streamlit as st
st.set_page_config(
    page_title="计算机视觉教学项目",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 抑制 Streamlit 内部的 ScriptRunContext 警告（线程池等场景的正常行为）
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import io
import time

# 导入各模块
from modules.hog_svm import run_hog_svm_pipeline
from modules.backprop import run_backprop_demo
from modules.cnn_train import run_cnn_training, LeNet5
from modules.resnet_compare import run_resnet_comparison
from utils.helpers import fig_to_base64, fig_to_pil, numpy_to_pil

# ============================================================
# 自定义CSS样式
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #666;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 标题
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 计算机视觉教学项目</h1>
    <p>传统方法与深度学习：从HOG+SVM到ResNet</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏 - 模块选择
# ============================================================
st.sidebar.title("📚 导航菜单")

module = st.sidebar.selectbox(
    "选择教学模块",
    [
        "🏠 项目概述",
        "📐 模块1: HOG + BoW + SVM 图像分类",
        "🧠 模块2: 神经网络反向传播演示",
        "🔬 模块3: CNN模型训练与测试",
        "🔄 模块4: ResNet预训练模型对比",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 关于本项目")
st.sidebar.info(
    "本项目覆盖了计算机视觉的核心知识点：\n\n"
    "- **传统方法**: HOG特征 + 视觉词袋 + SVM\n"
    "- **基础深度学习**: 手动实现反向传播\n"
    "- **CNN**: LeNet-5卷积网络训练\n"
    "- **现代架构**: ResNet预训练模型对比\n\n"
    "所有代码均从头实现或基于标准库，适合教学与演示。"
)

# ============================================================
# 页面路由
# ============================================================

# ---- 项目概述 ----
if module == "🏠 项目概述":
    st.markdown("## 🏠 项目概述")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="value">4</div>
            <div class="label">教学模块</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="value">HOG+SVM</div>
            <div class="label">传统方法</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="value">PyTorch</div>
            <div class="label">深度学习框架</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="value">Streamlit</div>
            <div class="label">交互界面</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### 🎯 学习目标

    通过本项目，你将掌握：

    1. **传统计算机视觉流程**
       - HOG特征提取的原理与实现
       - KMeans聚类构建视觉词袋
       - SVM分类器的使用

    2. **神经网络基础**
       - 前向传播的数学原理
       - 反向传播与链式法则
       - 梯度下降优化

    3. **卷积神经网络**
       - CNN的基本结构（卷积层、池化层、全连接层）
       - PyTorch训练流程
       - 特征图可视化

    4. **现代深度架构**
       - ResNet残差连接原理
       - 预训练模型的使用
       - 模型性能对比分析

    ### 🚀 快速开始

    从左侧边栏选择一个模块开始学习！每个模块都支持参数调节和交互式运行。
    """)

# ---- 模块1: HOG + SVM ----
elif module == "📐 模块1: HOG + BoW + SVM 图像分类":
    st.markdown("## 📐 模块1: HOG + Bag of Words + SVM 图像分类")
    st.markdown("""
    <div class="info-box">
    <strong>学习要点：</strong> HOG特征提取 → KMeans视觉词袋 → SVM分类器的完整传统CV分类流程。
    使用 sklearn digits 数据集（8×8手写数字）。
    </div>
    """, unsafe_allow_html=True)

    # 参数面板
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("KMeans聚类数（词袋大小）", min_value=10, max_value=100, value=50, step=10,
                               help="词袋中视觉单词的数量。越多越精细，但计算量也越大。")
    with col2:
        orientations = st.slider("HOG方向bin数", min_value=4, max_value=18, value=9, step=1,
                                 help="梯度方向直方图的bin数量，通常取9（0-180度，每20度一个bin）。")

    if st.button("🚀 运行 HOG+SVM 分类流程", type="primary", use_container_width=True):
        with st.spinner("正在运行 HOG+SVM 流程... 请稍候"):
            try:
                results = run_hog_svm_pipeline(
                    n_clusters=n_clusters,
                    orientations=orientations
                )

                st.success(f"✅ 运行完成！测试准确率: {results['accuracy']:.4f}")

                # 数据概览
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("训练样本", results['n_train'])
                with col2:
                    st.metric("测试样本", results['n_test'])
                with col3:
                    st.metric("HOG特征维度", results['hog_dim'])
                with col4:
                    st.metric("BoW特征维度", results['bow_dim'])

                # 准确率
                st.markdown(f"### 🎯 分类准确率: **{results['accuracy']:.2%}**")

                # 样本图像
                st.markdown("### 📷 数据集样本")
                st.pyplot(results['sample_fig'])

                # HOG可视化
                st.markdown("### 🔍 HOG特征可视化")
                st.markdown(
                    "HOG将图像划分为cell，在每个cell内计算梯度方向直方图，"
                    "反映了图像的边缘方向和强度分布。"
                )
                st.pyplot(results['hog_fig'])

                # 混淆矩阵
                st.markdown("### 📊 混淆矩阵")
                st.pyplot(results['cm_fig'])

                # 预测示例
                st.markdown("### ✅ 预测结果示例")
                st.pyplot(results['pred_samples_fig'])

            except Exception as e:
                st.error(f"运行出错: {str(e)}")
                st.exception(e)

# ---- 模块2: 反向传播 ----
elif module == "🧠 模块2: 神经网络反向传播演示":
    st.markdown("## 🧠 模块2: 神经网络反向传播演示")
    st.markdown("""
    <div class="info-box">
    <strong>学习要点：</strong> 从零开始用NumPy实现2层神经网络，手动完成前向传播、交叉熵损失和反向传播。
    使用2D数据以便可视化决策边界。
    </div>
    """, unsafe_allow_html=True)

    # 参数面板
    col1, col2 = st.columns(2)
    with col1:
        dataset_type = st.selectbox(
            "数据集类型",
            ['moons', 'circles', 'blobs'],
            help="moons=月牙形(非线性), circles=同心圆, blobs=高斯聚类"
        )
        hidden_size = st.slider("隐藏层神经元数", 5, 50, 20, 5,
                                help="隐藏层神经元越多，模型容量越大，但也更容易过拟合。")

    with col2:
        learning_rate = st.slider("学习率", 0.01, 2.0, 0.5, 0.01,
                                  help="控制参数更新的步长。太大可能震荡，太小收敛慢。")
        epochs = st.slider("训练轮数", 100, 5000, 1000, 100,
                           help="更多轮数可能提高准确率，但边际收益递减。")
        reg_lambda = st.slider("L2正则化系数", 0.0, 0.1, 0.01, 0.001,
                               help="防止过拟合。越大对权重的惩罚越强。")

    if st.button("🚀 运行反向传播演示", type="primary", use_container_width=True):
        with st.spinner("正在训练神经网络... 请稍候"):
            try:
                results = run_backprop_demo(
                    dataset_type=dataset_type,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    reg_lambda=reg_lambda
                )

                st.success(f"✅ 训练完成！验证准确率: {results['final_val_acc']:.4f}")

                # 指标
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最终损失", f"{results['final_loss']:.4f}")
                with col2:
                    st.metric("训练准确率", f"{results['final_train_acc']:.4f}")
                with col3:
                    st.metric("验证准确率", f"{results['final_val_acc']:.4f}")

                # 训练曲线
                st.markdown("### 📈 训练曲线")
                st.pyplot(results['curve_fig'])

                # 决策边界
                st.markdown("### 🗺️ 决策边界可视化")
                st.markdown(
                    "决策边界展示了神经网络学到的非线性分类边界。"
                    "ReLU激活函数使网络能够学习复杂的非线性决策边界。"
                )
                st.pyplot(results['boundary_fig'])

                # 梯度流
                st.markdown("### 🌊 梯度流分析")
                st.markdown(
                    "梯度流图展示了各层梯度的平均大小。如果某层梯度接近0，"
                    "说明存在梯度消失问题；如果梯度很大，说明可能有梯度爆炸。"
                )
                st.pyplot(results['grad_flow_fig'])

                # 训练日志
                with st.expander("📋 查看训练日志"):
                    st.code(results['log'])

                # 模型参数信息
                with st.expander("ℹ️ 模型参数信息"):
                    params = results['params']
                    st.write(f"- 输入维度: {results['input_dim']}")
                    st.write(f"- 隐藏层大小: {params['hidden_size']}")
                    st.write(f"- 输出类别数: {results['n_classes']}")
                    st.write(f"- 总参数量: {params['total_params']}")
                    st.write(f"- W1: {params['W1_shape']}, W2: {params['W2_shape']}")

            except Exception as e:
                st.error(f"运行出错: {str(e)}")
                st.exception(e)

# ---- 模块3: CNN训练 ----
elif module == "🔬 模块3: CNN模型训练与测试":
    st.markdown("## 🔬 模块3: CNN模型训练与测试")
    st.markdown("""
    <div class="info-box">
    <strong>学习要点：</strong> 使用PyTorch实现LeNet-5风格CNN，在MNIST数据集上训练和评估。
    包含特征图可视化，展示CNN每层学到的特征。
    </div>
    """, unsafe_allow_html=True)

    # 参数面板
    st.markdown("### ⚙️ 训练参数")

    col1, col2 = st.columns(2)
    with col1:
        epochs_cnn = st.slider("训练轮数 (Epochs)", 1, 10, 5, 1,
                               help="完整遍历数据集的次数。MNIST通常5轮就有不错效果。")
        batch_size = st.selectbox("批次大小 (Batch Size)", [32, 64, 128], index=1,
                                  help="每批处理的样本数。越大占用内存越多，但梯度估计更稳定。")

    with col2:
        learning_rate_cnn = st.selectbox("学习率 (Learning Rate)",
                                         [0.01, 0.001, 0.0001], index=1,
                                         help="Adam优化器的学习率。默认0.001通常效果很好。")
        use_subset = st.checkbox("使用子集（加速演示）", value=True,
                                 help="使用5000张训练图片加速演示，关闭则使用全部60000张。")

    if st.button("🚀 开始CNN训练", type="primary", use_container_width=True):
        with st.spinner("正在训练CNN... 这可能需要几分钟"):
            try:
                results = run_cnn_training(
                    epochs=epochs_cnn,
                    batch_size=batch_size,
                    learning_rate=learning_rate_cnn,
                    use_subset=use_subset,
                    subset_size=5000
                )

                st.success(
                    f"✅ 训练完成！测试准确率: {results['final_test_acc']:.4f} "
                    f"| 设备: {results['device']}"
                )

                # 指标
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("训练样本", results['n_train'])
                with col2:
                    st.metric("测试样本", results['n_test'])
                with col3:
                    st.metric("模型参数", f"{results['trainable_params']:,}")

                # 准确率
                st.markdown(f"### 🎯 测试准确率: **{results['final_test_acc']:.2%}**")

                # 训练曲线
                st.markdown("### 📈 训练曲线")
                st.pyplot(results['curve_fig'])

                # 预测结果
                st.markdown("### 🔍 预测结果可视化")
                st.pyplot(results['pred_fig'])

                # 特征图
                st.markdown("### 🎨 卷积特征图可视化")
                st.markdown(
                    "**Conv1**: 6个特征图，主要检测边缘、方向和简单纹理。\n\n"
                    "**Conv2**: 16个特征图，检测更高级的形状和模式组合。\n\n"
                    "越深的层学到的特征越抽象、越语义化。"
                )
                st.pyplot(results['feature_map_fig'])

                # 训练日志
                with st.expander("📋 查看训练日志"):
                    st.code(results['log'])

            except Exception as e:
                st.error(f"运行出错: {str(e)}")
                st.exception(e)

# ---- 模块4: ResNet对比 ----
elif module == "🔄 模块4: ResNet预训练模型对比":
    st.markdown("## 🔄 模块4: ResNet预训练模型对比")
    st.markdown("""
    <div class="info-box">
    <strong>学习要点：</strong> 加载ImageNet预训练的ResNet18/34/50，在MNIST上对比准确率、
    推理速度和参数量。MNIST灰度图自动转为3通道以适配ResNet。
    </div>
    """, unsafe_allow_html=True)

    # 参数面板
    col1, col2 = st.columns(2)
    with col1:
        subset_size = st.slider("测试样本数", 200, 5000, 1000, 200,
                                help="用于评估的样本数。更多样本结果更可靠，但更耗时。")
    with col2:
        batch_size_resnet = st.selectbox("批次大小", [8, 16, 32], index=1,
                                         help="ResNet模型较大，建议使用较小的batch size。")

    st.markdown("### 🏗️ ResNet残差块结构")
    # 静态展示残差块结构
    from modules.resnet_compare import plot_resnet_architecture
    arch_fig = plot_resnet_architecture()
    st.pyplot(arch_fig)

    if st.button("🚀 运行ResNet对比", type="primary", use_container_width=True):
        with st.spinner("正在加载预训练模型并进行对比... 这可能需要几分钟（首次运行需下载模型和数据集）"):
            try:
                results = run_resnet_comparison(
                    subset_size=subset_size,
                    batch_size=batch_size_resnet
                )

                st.success(f"✅ 对比完成！设备: {results['device']}")

                # 结果表格
                st.markdown("### 📊 模型对比结果")
                st.dataframe(
                    results['dataframe'].style.highlight_max(subset=['Accuracy'], color='#c8e6c9')
                                         .highlight_min(subset=['Inference Time (ms/img)'], color='#c8e6c9')
                                         .format({'Accuracy': '{:.2f}%', 'Inference Time (ms/img)': '{:.2f}'}),
                    use_container_width=True
                )

                # 柱状图对比
                st.markdown("### 📈 可视化对比")
                st.pyplot(results['comparison_fig'])

                # 分析说明
                st.markdown("### 📝 结果分析")
                best_acc = results['dataframe'].loc[results['dataframe']['Accuracy'].idxmax()]
                fastest = results['dataframe'].loc[results['dataframe']['Inference Time (ms/img)'].idxmin()]
                smallest = results['dataframe'].loc[results['dataframe']['Parameters (M)'].idxmin()]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="success-box">
                    <strong>🏆 最高准确率</strong><br>
                    {best_acc['Model']}: {best_acc['Accuracy']:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="success-box">
                    <strong>⚡ 最快推理</strong><br>
                    {fastest['Model']}: {fastest['Inference Time (ms/img)']:.1f} ms/img
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="success-box">
                    <strong>📦 最小模型</strong><br>
                    {smallest['Model']}: {smallest['Parameters (M)']}M params
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box">
                <strong>💡 关键发现：</strong><br>
                - ResNet50使用Bottleneck结构，层数更深但参数不一定更多<br>
                - 更深模型通常准确率更高，但推理时间也更长<br>
                - 在实际应用中需要在准确率和效率之间权衡
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"运行出错: {str(e)}")
                st.exception(e)

# ============================================================
# 页脚
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>计算机视觉教学项目 | Built with Streamlit & PyTorch</small>",
    unsafe_allow_html=True
)
