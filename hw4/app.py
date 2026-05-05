"""
机器学习关键算法与可视化 - Streamlit 应用
========================================
包含：最小二乘回归、KNN分类、线性分类器(SGD vs Momentum)、损失函数演示、模板图像展示

运行方式：streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from utils import (
    generate_regression_data, least_squares_fit, predict_linear,
    load_cifar10_or_fallback, get_cifar10_subset, normalize_cifar10,
    KNNClassifier, LinearClassifier,
    compute_loss_demo,
    plot_sample_images, plot_confusion_matrix,
    CIFAR10_CLASSES, CIFAR10_CLASSES_EN
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="ML算法可视化平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 缓存数据加载
# ============================================================
CIFAR_CACHE_DIR = str(Path(__file__).parent / "cifar10_cache")


@st.cache_data(show_spinner="正在加载 CIFAR-10 数据集...")
def load_cifar10_cached():
    """带缓存的CIFAR-10加载（含合成数据降级）"""
    result = load_cifar10_or_fallback(CIFAR_CACHE_DIR)
    return result  # (X_train, y_train, X_test, y_test, is_synthetic)


@st.cache_data(show_spinner="正在准备数据子集...")
def prepare_cifar10_subset(classes, num_train_per_class, num_test_per_class, seed=42):
    """带缓存的CIFAR-10子集准备"""
    X_train_full, y_train_full, X_test_full, y_test_full, _ = load_cifar10_cached()
    return get_cifar10_subset(
        X_train_full, y_train_full, X_test_full, y_test_full,
        classes=classes, num_train_per_class=num_train_per_class,
        num_test_per_class=num_test_per_class, seed=seed
    )


@st.cache_data(show_spinner="正在运行KNN分类...")
def run_knn_cached(X_train, y_train, X_test, y_test, k_values):
    """带缓存的KNN运行"""
    results = {}
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = np.mean(preds == y_test)
        results[k] = {'predictions': preds, 'accuracy': acc}
    return results


@st.cache_data(show_spinner="正在训练线性分类器...")
def train_linear_classifier_cached(X_train, y_train, X_test, y_test,
                                    loss_type, lr, epochs, optimizer,
                                    momentum, batch_size, reg, _seed=42):
    """带缓存的线性分类器训练"""
    np.random.seed(_seed)
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    
    clf = LinearClassifier(n_classes=n_classes, n_features=n_features, loss_type=loss_type)
    loss_hist, acc_hist = clf.train(
        X_train, y_train, lr=lr, epochs=epochs,
        optimizer=optimizer, momentum=momentum,
        batch_size=batch_size, reg=reg, verbose=False
    )
    test_acc = clf.score(X_test, y_test)
    return loss_hist, acc_hist, test_acc, clf.W, clf.b


# ============================================================
# 侧边栏 - 导航
# ============================================================
st.sidebar.title("🤖 ML算法可视化")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航菜单",
    [
        "📈 最小二乘回归",
        "🔍 KNN可视化",
        "⚡ 线性分类器训练对比",
        "📉 损失函数演示",
        "🖼️ 模板图像展示"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于")
st.sidebar.info(
    "本应用展示了机器学习中的几个经典算法，"
    "包含最小二乘回归、K近邻分类、线性分类器"
    "（Softmax/SVM）等算法的交互式可视化。"
)

st.sidebar.markdown("### 依赖安装")
st.sidebar.code(
    "pip install streamlit numpy matplotlib scikit-learn requests",
    language="bash"
)

# ============================================================
# 页面1: 最小二乘回归
# ============================================================
def show_regression_page():
    st.header("📈 最小二乘法线性回归")
    st.markdown("""
    本页面演示了**最小二乘法(Ordinary Least Squares)** 线性回归的基本原理。
    通过调节噪声水平和数据点数量，直观观察拟合效果的变化。
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("参数控制")
        n_points = st.slider("数据点数量", min_value=10, max_value=200, value=60, step=5)
        noise_level = st.slider("噪声水平", min_value=0.0, max_value=3.0, value=0.5, step=0.05)
        x_range = st.slider("X轴范围", min_value=-5.0, max_value=5.0, value=(-3.0, 3.0), step=0.5)
        seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)
        
        st.markdown("---")
        st.markdown("#### 拟合结果")
    
    # 生成数据
    X, y_true, y = generate_regression_data(
        n_points=n_points, noise_level=noise_level,
        x_range=x_range, seed=int(seed)
    )
    
    # 最小二乘拟合
    theta = least_squares_fit(X, y)
    y_pred = predict_linear(X, theta)
    
    # 计算误差
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    with col1:
        st.metric("截距 (θ₀)", f"{theta[0]:.4f}")
        st.metric("斜率 (θ₁)", f"{theta[1]:.4f}")
        st.metric("均方误差 (MSE)", f"{mse:.4f}")
        st.metric("R² 决定系数", f"{r2:.4f}")
        
        st.markdown("#### 真实函数")
        st.latex(r"y = 1.2x + 0.8")
        
        if st.checkbox("显示理论值"):
            st.info(f"理论截距: 0.8\n理论斜率: 1.2\n拟合截距: {theta[0]:.4f}\n拟合斜率: {theta[1]:.4f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 散点图（带噪声的数据）
        ax.scatter(X, y, alpha=0.6, label='Observed data (noisy)', color='steelblue', s=40)
        
        # True function
        ax.plot(X, y_true, 'g--', linewidth=2, label='True function: y = 1.2x + 0.8', alpha=0.8)
        
        # Fitted line
        ax.plot(X, y_pred, 'r-', linewidth=2.5, label=f'Fitted line: y = {theta[1]:.3f}x + {theta[0]:.3f}')
        
        # 绘制残差
        for i in range(len(X)):
            ax.plot([X[i], X[i]], [y[i], y_pred[i]], 'gray', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Least Squares Linear Regression (noise={noise_level}, n={n_points})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    # 残差分析
    with st.expander("📊 残差分析", expanded=False):
        residuals = y - y_pred
        
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 残差图
        axes[0].scatter(X, residuals, alpha=0.7, color='coral', s=40)
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[0].set_xlabel('X', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residual Plot', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
        axes[1].set_xlabel('Residuals', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residual Histogram', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ============================================================
# 页面2: KNN可视化
# ============================================================
def show_knn_page():
    st.header("🔍 KNN 图像分类可视化 (CIFAR-10)")
    st.markdown("""
    本页面使用 **K近邻(K-Nearest Neighbors)** 算法在 CIFAR-10 数据集上进行图像分类。
    通过调节K值，观察分类效果的变化。
    """)
    
    # 参数设置
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("数据参数")
        
        # 选择类别（默认选前5个类别以减少计算量）
        selected_classes = st.multiselect(
            "选择类别",
            options=list(range(10)),
            format_func=lambda x: f"{CIFAR10_CLASSES[x]} ({CIFAR10_CLASSES_EN[x]})",
            default=[0, 1, 2, 3, 4]
        )
        
        if not selected_classes:
            st.warning("请至少选择一个类别")
            st.stop()
        
        num_train = st.slider("每类训练样本数", min_value=20, max_value=200, value=80, step=10)
        num_test = st.slider("每类测试样本数", min_value=10, max_value=100, value=30, step=5)
        
        # K值选择
        st.subheader("K值参数")
        selected_ks = st.multiselect(
            "选择要比较的K值",
            options=[1, 3, 5, 7, 10, 15, 20],
            default=[1, 3, 5, 10]
        )
        
        if not selected_ks:
            st.warning("请至少选择一个K值")
            st.stop()
        
        run_button = st.button("🚀 运行KNN分类", type="primary")
    
    with col2:
        # 加载数据
        with st.spinner("正在准备CIFAR-10数据..."):
            X_train, y_train, X_test, y_test = prepare_cifar10_subset(
                classes=selected_classes,
                num_train_per_class=num_train,
                num_test_per_class=num_test,
                seed=42
            )
        
        # 归一化
        X_train_norm, X_test_norm = normalize_cifar10(X_train, X_test)
        
        st.info(f"✅ 数据准备完成: 训练集 {X_train.shape[0]} 张, 测试集 {X_test.shape[0]} 张, 类别数 {len(selected_classes)}")
        
        if run_button:
            # 运行KNN
            results = run_knn_cached(X_train_norm, y_train, X_test_norm, y_test, selected_ks)
            
            # 准确率柱状图
            fig, ax = plt.subplots(figsize=(10, 5))
            k_list = sorted(results.keys())
            acc_list = [results[k]['accuracy'] for k in k_list]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(k_list)))
            bars = ax.bar([str(k) for k in k_list], acc_list, color=colors, width=0.6, edgecolor='gray')
            
            # 在柱子上显示数值
            for bar, acc in zip(bars, acc_list):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_xlabel('K Value', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Accuracy Comparison across K Values', fontsize=14)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
            plt.close()
            
            # 显示最佳K值
            best_k = k_list[np.argmax(acc_list)]
            best_acc = max(acc_list)
            st.success(f"🏆 最佳K值: K={best_k}, 准确率: {best_acc:.2%}")
            
            # 展示部分测试图片的预测结果
            st.subheader("测试样本预测结果展示")
            
            # 让用户选择K值查看具体预测
            view_k = st.selectbox("选择K值查看预测结果", k_list, index=k_list.index(best_k) if best_k in k_list else 0)
            
            # 显示一些随机样本
            n_show = min(10, len(X_test))
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(X_test), n_show, replace=False)
            
            sample_images = X_test[sample_idx]
            sample_labels = y_test[sample_idx]
            sample_preds = results[view_k]['predictions'][sample_idx]
            
            fig2 = plot_sample_images(
                sample_images, sample_labels, sample_preds,
                CIFAR10_CLASSES_EN, num_samples=n_show
            )
            st.pyplot(fig2)
            plt.close()
            
            # Confusion matrix (optional)
            with st.expander("📊 View Confusion Matrix", expanded=False):
                view_k_cm = st.selectbox("Select K for confusion matrix", k_list, key="cm_select")
                fig3 = plot_confusion_matrix(
                    y_test, results[view_k_cm]['predictions'],
                    [CIFAR10_CLASSES_EN[c] for c in selected_classes],
                    title=f"Confusion Matrix (K={view_k_cm})"
                )
                st.pyplot(fig3)
                plt.close()
        else:
            st.info('👆 点击"运行KNN分类"按钮开始分类')


# ============================================================
# 页面3: 线性分类器训练对比
# ============================================================
def show_classifier_page():
    st.header("⚡ 线性分类器训练对比: SGD vs Momentum")
    st.markdown("""
    本页面比较 **SGD (随机梯度下降)** 与 **Momentum (动量更新)** 两种优化算法
    在 CIFAR-10 数据集上训练线性分类器的效果。
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("训练参数")
        
        # 分类器类型
        loss_type = st.radio(
            "损失函数类型",
            options=['softmax', 'svm'],
            format_func=lambda x: 'Softmax (交叉熵损失)' if x == 'softmax' else 'SVM (合页损失)',
            index=0
        )
        
        # 数据参数
        selected_classes = st.multiselect(
            "选择类别（建议选3-5个类别以加快训练）",
            options=list(range(10)),
            format_func=lambda x: f"{CIFAR10_CLASSES[x]} ({CIFAR10_CLASSES_EN[x]})",
            default=[0, 1, 2, 3, 4]
        )
        
        if not selected_classes:
            st.warning("请至少选择一个类别")
            st.stop()
        
        num_per_class = st.slider("每类样本数", min_value=50, max_value=500, value=200, step=50)
        
        # 训练超参数
        st.subheader("超参数")
        learning_rate = st.select_slider(
            "学习率 (Learning Rate)",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.001,
            format_func=lambda x: f'{x:.4f}'
        )
        
        epochs = st.slider("训练轮数 (Epochs)", min_value=10, max_value=200, value=50, step=10)
        batch_size = st.slider("批次大小 (Batch Size)", min_value=16, max_value=256, value=64, step=16)
        reg = st.slider("L2正则化强度", min_value=0.0, max_value=1.0, value=0.001, step=0.001,
                        format="%.3f")
        
        momentum = st.slider("动量系数 (Momentum)", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
        
        train_button = st.button("🚀 开始训练对比", type="primary")
    
    with col2:
        # 加载数据
        with st.spinner("正在准备CIFAR-10数据..."):
            X_train, y_train, X_test, y_test = prepare_cifar10_subset(
                classes=selected_classes,
                num_train_per_class=num_per_class,
                num_test_per_class=min(50, num_per_class // 2),
                seed=42
            )
        
        # 归一化
        X_train_norm, X_test_norm = normalize_cifar10(X_train, X_test)
        
        st.info(f"✅ 数据准备完成: 训练集 {X_train.shape[0]} 张, 测试集 {X_test.shape[0]} 张")
        
        if train_button:
            with st.spinner("正在训练SGD分类器..."):
                loss_sgd, acc_sgd, test_acc_sgd, W_sgd, b_sgd = train_linear_classifier_cached(
                    X_train_norm, y_train, X_test_norm, y_test,
                    loss_type=loss_type, lr=learning_rate, epochs=epochs,
                    optimizer='sgd', momentum=momentum,
                    batch_size=batch_size, reg=reg
                )
            
            with st.spinner("正在训练Momentum分类器..."):
                loss_mom, acc_mom, test_acc_mom, W_mom, b_mom = train_linear_classifier_cached(
                    X_train_norm, y_train, X_test_norm, y_test,
                    loss_type=loss_type, lr=learning_rate, epochs=epochs,
                    optimizer='momentum', momentum=momentum,
                    batch_size=batch_size, reg=reg
                )
            
            # 显示结果
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("SGD 测试准确率", f"{test_acc_sgd:.2%}")
            with col_res2:
                st.metric("Momentum 测试准确率", f"{test_acc_mom:.2%}")
            with col_res3:
                diff = test_acc_mom - test_acc_sgd
                st.metric("准确率差值 (Momentum - SGD)", f"{diff:+.2%}",
                         delta=f"{diff:+.2%}" if abs(diff) > 0.001 else None)
            
            # 损失曲线对比
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 损失曲线
            axes[0].plot(range(1, epochs + 1), loss_sgd, 'b-', linewidth=2, label='SGD', alpha=0.8)
            axes[0].plot(range(1, epochs + 1), loss_mom, 'r-', linewidth=2, label='Momentum', alpha=0.8)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            loss_name = 'Cross-Entropy' if loss_type == 'softmax' else 'Hinge'
            axes[0].set_title(f'Loss Curve Comparison ({loss_name})', fontsize=13)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy curves
            axes[1].plot(range(1, epochs + 1), acc_sgd, 'b-', linewidth=2, label='SGD', alpha=0.8)
            axes[1].plot(range(1, epochs + 1), acc_mom, 'r-', linewidth=2, label='Momentum', alpha=0.8)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Training Accuracy Comparison', fontsize=13)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 分析说明
            with st.expander("📖 结果分析", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**SGD (随机梯度下降)**")
                    st.markdown(f"""
                    - 最终损失: {loss_sgd[-1]:.4f}
                    - 最终准确率: {acc_sgd[-1]:.2%}
                    - 测试准确率: {test_acc_sgd:.2%}
                    - 特点: 收敛路径可能震荡，但计算简单
                    """)
                with col_b:
                    st.markdown("**Momentum (动量更新)**")
                    st.markdown(f"""
                    - 最终损失: {loss_mom[-1]:.4f}
                    - 最终准确率: {acc_mom[-1]:.2%}
                    - 测试准确率: {test_acc_mom:.2%}
                    - 特点: 加速收敛，减少震荡，通常更快达到最优
                    """)
                
                if test_acc_mom > test_acc_sgd:
                    st.success("✅ 在本实验中，Momentum优化器表现优于SGD，说明动量项有助于加速收敛并提高最终性能。")
                elif test_acc_mom < test_acc_sgd:
                    st.info("ℹ️ 在本实验中，SGD表现略优。这可能是因为动量系数设置、学习率或数据子集特性导致的。")
                else:
                    st.info("ℹ️ 两者表现相近。")
        else:
            st.info('👆 调整参数后点击"开始训练对比"')


# ============================================================
# 页面4: 损失函数演示
# ============================================================
def show_loss_page():
    st.header("📉 损失函数演示: Cross-Entropy vs Hinge Loss")
    st.markdown("""
    本页面直观对比 **交叉熵损失 (Cross-Entropy Loss)** 和 **合页损失 (Hinge Loss)** 
    的计算过程。你可以手动调整各类别的分数，观察损失值的变化。
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 模拟分数输入")
        st.markdown("为10个类别输入原始分数（logits），观察不同损失函数的计算结果。")
        
        # 创建随机生成按钮
        if st.button("🎲 随机生成分数"):
            rand_scores = np.random.randn(10) * 3 + 1
            st.session_state.scores = rand_scores.tolist()
        
        # 初始化session state
        if 'scores' not in st.session_state:
            st.session_state.scores = [2.5, 1.8, 0.5, -0.3, 1.0, -1.5, 0.2, -0.8, 0.7, -0.1]
        
        # 分数输入
        scores_input = []
        for i in range(10):
            val = st.number_input(
                f"{CIFAR10_CLASSES[i]} ({CIFAR10_CLASSES_EN[i]})",
                value=float(st.session_state.scores[i]),
                step=0.1,
                key=f"score_{i}",
                format="%.2f"
            )
            scores_input.append(val)
        
        scores = np.array(scores_input)
        true_class = st.selectbox(
            "真实类别 (True Class)",
            options=list(range(10)),
            format_func=lambda x: f"{CIFAR10_CLASSES[x]} ({CIFAR10_CLASSES_EN[x]})",
            index=0
        )
    
    with col2:
        st.subheader("📊 损失计算对比")
        
        # 计算两种损失
        ce_loss, _ = compute_loss_demo(scores.reshape(1, -1), np.array([true_class]), 'softmax')
        svm_loss, _ = compute_loss_demo(scores.reshape(1, -1), np.array([true_class]), 'svm')
        
        # 显示损失值（大号醒目显示）
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                "交叉熵损失",
                f"{ce_loss:.4f}",
                delta=None,
                delta_color="normal",
                help="Cross-Entropy Loss —— 越小越好，完美预测时为0"
            )
        with col_m2:
            st.metric(
                "合页损失",
                f"{svm_loss:.4f}",
                delta=None,
                delta_color="normal",
                help="Hinge Loss —— 越小越好，完美预测时为0"
            )
        st.markdown("---")


# ============================================================
# 页面5: 模板图像展示
# ============================================================
def show_templates_page():
    st.header("🖼️ 线性分类器模板图像展示")
    st.markdown("""
    线性分类器学习到的权重矩阵可以视为每个类别的"模板图像"。
    将权重矩阵的每一列 reshape 回 32×32×3 的图像，即可看到模型
    对每个类别的"印象"——即什么样的像素组合对该类别的判定最重要。
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("训练参数")
        
        loss_type = st.radio(
            "损失函数",
            options=['softmax', 'svm'],
            format_func=lambda x: 'Softmax (交叉熵)' if x == 'softmax' else 'SVM (合页损失)',
            index=0
        )
        
        optimizer = st.radio(
            "优化器",
            options=['sgd', 'momentum'],
            format_func=lambda x: 'SGD' if x == 'sgd' else 'Momentum',
            index=1
        )
        
        num_per_class = st.slider("每类样本数", min_value=100, max_value=1000, value=500, step=100)
        epochs = st.slider("训练轮数", min_value=10, max_value=100, value=30, step=5)
        lr = st.select_slider("学习率", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001,
                             format_func=lambda x: f'{x:.4f}')
        
        train_button = st.button("🔄 训练并生成模板图像", type="primary")
    
    with col2:
        # 加载全部10个类别的数据
        with st.spinner("正在加载CIFAR-10数据..."):
            X_train, y_train, X_test, y_test = prepare_cifar10_subset(
                classes=list(range(10)),
                num_train_per_class=num_per_class,
                num_test_per_class=min(50, num_per_class // 4),
                seed=42
            )
        
        X_train_norm, X_test_norm = normalize_cifar10(X_train, X_test)
        
        st.info(f"✅ 数据准备完成: 10个类别, 训练集 {X_train.shape[0]} 张")
        
        if train_button:
            with st.spinner("正在训练线性分类器..."):
                loss_hist, acc_hist, test_acc, W, b = train_linear_classifier_cached(
                    X_train_norm, y_train, X_test_norm, y_test,
                    loss_type=loss_type, lr=lr, epochs=epochs,
                    optimizer=optimizer, momentum=0.9,
                    batch_size=64, reg=0.001
                )
            
            st.success(f"✅ 训练完成! 测试准确率: {test_acc:.2%}")
            
            # 生成模板图像
            n_classes = 10
            templates = []
            for i in range(n_classes):
                w = W[:, i]
                w_min, w_max = w.min(), w.max()
                if w_max > w_min:
                    w_norm = (w - w_min) / (w_max - w_min)
                else:
                    w_norm = np.zeros_like(w)
                template = w_norm.reshape(3, 32, 32).transpose(1, 2, 0)
                templates.append(template)
            
            # 显示模板图像
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            
            for i in range(10):
                axes[i].imshow(templates[i])
                axes[i].set_title(f"{CIFAR10_CLASSES_EN[i]}", 
                                fontsize=11, fontweight='bold')
                axes[i].axis('off')
            
            plt.suptitle(f'Linear Classifier Template Images ({loss_type.upper()} loss, {optimizer.upper()} optimizer, LR={lr})',
                        fontsize=14, y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            **Template Image Interpretation:**
            - Each class template shows the "typical" pixel-space features for that class
            - Bright regions indicate positive weights (pushing toward that class)
            - Templates act as the model's "prototype" or "average impression" of each class
            - Comparing classes reveals how the model distinguishes them (e.g., blue backgrounds for airplane/ship, tire patterns for automobile/truck)
            """)
            
            # Loss curve
            st.subheader("Training Loss Curve")
            fig2, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(loss_hist) + 1), loss_hist, 'b-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss Curve', fontsize=13)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
            
        else:
            # 显示一个预生成的示例（使用随机权重可视化）
            st.info('👆 点击"训练并生成模板图像"以查看结果')
            
            # 显示一个样例说明
            st.markdown("""
            ### 什么是模板图像？
            
            线性分类器对每个类别学习一个权重向量 \\(w_c\\)，分类时计算:
            
            $$f_c(x) = x^T w_c + b_c$$
            
            其中 \\(w_c\\) 维度与输入图像相同（32×32×3=3072）。
            将 \\(w_c\\) 归一化后 reshape 回图像，就得到了该类别的"模板图像"。
            
            **模板图像的可视化意义：**
            - 🟢 亮色区域 → 该类别的正相关特征
            - 🔴 暗色区域 → 该类别的负相关特征
            - 例如，"船"类别的模板可能在蓝色（海水）区域有特殊权重分布
            """)


# ============================================================
# 主页面路由
# ============================================================
if page == "📈 最小二乘回归":
    show_regression_page()
elif page == "🔍 KNN可视化":
    show_knn_page()
elif page == "⚡ 线性分类器训练对比":
    show_classifier_page()
elif page == "📉 损失函数演示":
    show_loss_page()
elif page == "🖼️ 模板图像展示":
    show_templates_page()

# ============================================================
# 页脚
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "ML算法可视化平台 v1.0<br>"
    "Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
