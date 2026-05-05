"""
模块1: HOG + Bag of Words + SVM 图像分类
==========================================
传统计算机视觉分类流程：
1. HOG（Histogram of Oriented Gradients）特征提取
2. Bag of Words（基于KMeans聚类）构建视觉词袋
3. SVM分类器训练与评估

数据集: sklearn digits (8x8手写数字，简单快速)
"""
import os
# 修复 Windows 下 MKL/threadpoolctl DLL 冲突问题
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
import io

# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_data():
    """
    加载sklearn digits数据集
    返回: X_train, X_test, y_train, y_test, images (原始8x8图像)
    """
    digits = load_digits()
    X = digits.images       # shape: (1797, 8, 8) 原始图像
    y = digits.target       # shape: (1797,) 标签0-9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# 2. HOG特征提取
# ============================================================
def extract_hog_features(images, orientations=9, pixels_per_cell=(4, 4),
                          cells_per_block=(1, 1)):
    """
    对每张图像提取HOG特征。

    HOG原理简述：
    - 将图像划分为小的cell
    - 在每个cell内计算梯度方向直方图（orientation个bin）
    - 在block内做归一化以增强光照不变性
    - 拼接所有block的直方图作为最终特征向量

    参数:
        images: 图像数组 (n, h, w)
        orientations: 梯度方向bin数量
        pixels_per_cell: 每个cell的像素大小
        cells_per_block: 每个block包含的cell数
    返回:
        features: HOG特征矩阵 (n, feature_dim)
        hog_images: HOG可视化图像列表
    """
    features = []
    hog_images = []

    for img in images:
        # 提取HOG特征与可视化图像
        fd, hog_img = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True
        )
        features.append(fd)
        hog_images.append(hog_img)

    return np.array(features), hog_images


# ============================================================
# 3. Bag of Words (视觉词袋) 构建
# ============================================================
def build_bow_features(hog_features, n_clusters=50):
    """
    使用KMeans聚类构建视觉词袋(BoW)特征。

    原理：
    1. 对所有训练图像的HOG局部特征进行KMeans聚类
    2. 每个聚类中心代表一个"视觉单词"
    3. 将每张图像的特征分配到最近的聚类中心
    4. 统计每个视觉单词出现的频率 → 得到BoW直方图

    由于HOG特征已经是全局描述子，我们将其分块或直接用KMeans
    对特征空间进行聚类来构建词袋表示。

    参数:
        hog_features: 原始HOG特征 (n, d)
        n_clusters: 聚类中心数量（词袋大小）
    返回:
        bow_features: BoW特征 (n, n_clusters)
        kmeans: 训练好的KMeans模型
    """
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(hog_features)

    # KMeans聚类构建视觉词典
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=100,
        n_init=3
    )
    kmeans.fit(scaled_features)

    # 计算每个样本到聚类中心的距离，构建BoW表示
    distances = kmeans.transform(scaled_features)

    # 使用软分配：距离越近权重越大
    # 这里采用指数衰减的软分配方式
    bow_features = np.exp(-distances)
    bow_features = bow_features / bow_features.sum(axis=1, keepdims=True)

    return bow_features, kmeans, scaler


# ============================================================
# 4. SVM分类器训练
# ============================================================
def train_svm(X_train_bow, y_train):
    """
    使用RBF核SVM训练分类器。

    RBF核的优点：可以处理非线性可分的数据
    """
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_bow, y_train)
    return svm


# ============================================================
# 5. 可视化函数
# ============================================================
def plot_sample_images(images, labels, n_samples=10):
    """展示样本原始图像"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    fig.suptitle('Sample Images from Digits Dataset', fontsize=14)
    plt.tight_layout()
    return fig


def plot_hog_visualization(images, hog_images, n_samples=4):
    """并排展示原始图像与HOG可视化图像"""
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 6))
    for i in range(n_samples):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f'Original (label: {np.argmax(images[i].sum())})')
        axes[0, i].axis('off')

        axes[1, i].imshow(hog_images[i], cmap='gray')
        axes[1, i].set_title('HOG Visualization')
        axes[1, i].axis('off')
    fig.suptitle('Original Image vs HOG Feature Visualization', fontsize=14)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, classes):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        xlabel='Predicted',
        ylabel='True',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # 在每个格子中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


# ============================================================
# 6. 主流程函数 - 供Streamlit调用
# ============================================================
def run_hog_svm_pipeline(n_clusters=50, orientations=9):
    """
    运行完整的 HOG + BoW + SVM 分类流程。

    参数:
        n_clusters: KMeans聚类数（词袋大小）
        orientations: HOG方向bin数
    返回:
        results: 包含所有结果和图像的字典
    """
    results = {}

    # --- 加载数据 ---
    X_train, X_test, y_train, y_test = load_data()
    results['n_train'] = len(X_train)
    results['n_test'] = len(X_test)
    results['n_classes'] = 10

    # --- 样本图像可视化 ---
    results['sample_fig'] = plot_sample_images(X_train, y_train)

    # --- HOG特征提取 ---
    train_hog, train_hog_imgs = extract_hog_features(
        X_train, orientations=orientations
    )
    test_hog, test_hog_imgs = extract_hog_features(
        X_test, orientations=orientations
    )
    results['hog_dim'] = train_hog.shape[1]
    results['hog_fig'] = plot_hog_visualization(X_train, train_hog_imgs)

    # --- BoW特征构建 ---
    bow_train, kmeans, scaler = build_bow_features(train_hog, n_clusters=n_clusters)

    # 对测试集用同样的pipeline
    bow_test = scaler.transform(test_hog)
    bow_test = np.exp(-kmeans.transform(bow_test))
    bow_test = bow_test / bow_test.sum(axis=1, keepdims=True)

    results['bow_dim'] = bow_train.shape[1]

    # --- SVM训练 ---
    svm = train_svm(bow_train, y_train)

    # --- 预测与评估 ---
    y_pred = svm.predict(bow_test)
    acc = accuracy_score(y_test, y_pred)
    results['accuracy'] = acc
    results['report'] = classification_report(y_test, y_pred, output_dict=True)

    # --- 混淆矩阵 ---
    cm = confusion_matrix(y_test, y_pred)
    results['cm_fig'] = plot_confusion_matrix(cm, classes=list(range(10)))

    # --- 预测示例 ---
    n_display = 8
    results['pred_samples_fig'] = plot_prediction_samples(
        X_test[:n_display], y_test[:n_display], y_pred[:n_display]
    )

    return results


def plot_prediction_samples(images, true_labels, pred_labels):
    """展示预测结果样本"""
    n = len(images)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i < n:
            ax.imshow(images[i], cmap='gray')
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            ax.set_title(f'T:{true_labels[i]} P:{pred_labels[i]}', color=color, fontsize=10)
        ax.axis('off')

    fig.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    return fig
