"""
Machine Learning 算法可视化工具函数
包含：最小二乘回归、CIFAR-10数据加载、KNN分类器、线性分类器
"""

import numpy as np
import pickle
import os
import tarfile
import urllib.request
import hashlib
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# 全局配置
# ============================================================
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']
CIFAR10_CLASSES_EN = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================
# 1. 最小二乘法线性回归
# ============================================================
def generate_regression_data(n_points=50, noise_level=0.5, x_range=(-3, 3), seed=42):
    """
    生成线性回归合成数据
    真实函数: y = 1.2x + 0.8
    """
    rng = np.random.RandomState(seed)
    X = np.linspace(x_range[0], x_range[1], n_points)
    y_true = 1.2 * X + 0.8
    noise = rng.randn(n_points) * noise_level
    y = y_true + noise
    return X, y_true, y


def least_squares_fit(X, y):
    """
    最小二乘法拟合线性回归参数
    返回: [intercept, slope]
    """
    X_b = np.c_[np.ones_like(X), X]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta


def predict_linear(X, theta):
    """使用拟合的参数进行预测"""
    X_b = np.c_[np.ones_like(X), X]
    return X_b @ theta


# ============================================================
# 2. CIFAR-10 数据加载
# ============================================================
# 多个下载源（主站 + 镜像）
CIFAR10_DOWNLOAD_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://github.com/YashAcharya01/CIFAR-10/raw/master/cifar-10-python.tar.gz",
    "https://objectstorage.ap-tokyo-1.oraclecloud.com/n/nrh4nwfssnqn/b/cifar-10/o/cifar-10-python.tar.gz",
]


def _check_cache_exists(cache_dir):
    """检查CIFAR-10缓存是否存在且完整"""
    data_path = Path(cache_dir) / "cifar-10-batches-py"
    if not data_path.exists():
        return False
    required_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    for f in required_files:
        if not (data_path / f).exists():
            return False
    return True


def _download_with_requests(url, save_path, timeout=120):
    """使用requests下载文件，支持进度显示"""
    import requests
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    downloaded = 0
    chunk_size = 8192
    with open(save_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total
                    progress_bar.progress(min(pct, 1.0))
                    status_text.text(f"下载中: {downloaded//1024//1024}MB / {total//1024//1024}MB ({pct*100:.1f}%)")
    
    progress_bar.progress(1.0)
    status_text.text("下载完成！")
    return True


def _download_cifar10(cache_dir):
    """尝试从多个源下载并解压CIFAR-10数据集"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = cache_path / "cifar-10-python.tar.gz"
    
    # 如果已经有tar文件但之前解压失败，清理重试
    if tar_path.exists():
        tar_path.unlink()
    
    # 尝试所有下载源
    download_success = False
    for url_idx, url in enumerate(CIFAR10_DOWNLOAD_URLS):
        try:
            st.info(f"正在从镜像 {url_idx + 1}/{len(CIFAR10_DOWNLOAD_URLS)} 下载 CIFAR-10 数据集...")
            _download_with_requests(url, tar_path)
            st.success("下载完成！正在解压...")
            download_success = True
            break
        except Exception as e:
            st.warning(f"镜像 {url_idx + 1} 下载失败: {str(e)[:80]}")
            # 清理失败的下载
            if tar_path.exists():
                tar_path.unlink()
            continue
    
    if not download_success:
        st.error("所有CIFAR-10下载源均不可用，将使用合成数据替代。")
        return False
    
    # 解压
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=cache_path)
        st.success("数据集解压完成！")
        return True
    except Exception as e:
        st.error(f"解压失败: {e}")
        return False


def load_cifar10(cache_dir="./cifar10_cache"):
    """
    加载CIFAR-10数据集
    返回: (X_train, y_train, X_test, y_test) 或 None（如果加载失败）
    X: shape (N, 3072), 像素值范围 0-255
    y: shape (N,), 类别标签 0-9
    """
    if not _check_cache_exists(cache_dir):
        success = _download_cifar10(cache_dir)
        if not success:
            return None
    
    try:
        data_path = Path(cache_dir) / "cifar-10-batches-py"
        
        # 加载训练集
        X_train_list = []
        y_train_list = []
        for i in range(1, 6):
            batch_path = data_path / f"data_batch_{i}"
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            X_train_list.append(batch[b'data'])
            y_train_list.append(np.array(batch[b'labels']))
        
        X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # 加载测试集
        test_path = data_path / "test_batch"
        with open(test_path, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')
        X_test = test_batch[b'data'].astype(np.float32)
        y_test = np.array(test_batch[b'labels'])
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        st.error(f"加载CIFAR-10文件失败: {e}")
        return None


def generate_synthetic_cifar10(num_train=5000, num_test=1000, seed=42):
    """
    当CIFAR-10下载失败时，生成合成图像数据作为替代
    生成10个类别的32x32x3彩色图像，每个类别有特定的颜色/模式特征
    """
    rng = np.random.RandomState(seed)
    img_size = 32
    n_channels = 3
    n_pixels = img_size * img_size * n_channels
    n_classes = 10
    
    # 每个类别的特征颜色 (R, G, B)
    class_colors = [
        (0.8, 0.2, 0.2),  # 0: 红色 - airplane
        (0.2, 0.8, 0.2),  # 1: 绿色 - automobile
        (0.2, 0.2, 0.8),  # 2: 蓝色 - bird
        (0.8, 0.8, 0.2),  # 3: 黄色 - cat
        (0.8, 0.2, 0.8),  # 4: 紫色 - deer
        (0.2, 0.8, 0.8),  # 5: 青色 - dog
        (0.8, 0.5, 0.2),  # 6: 橙色 - frog
        (0.5, 0.2, 0.8),  # 7: 靛色 - horse
        (0.5, 0.8, 0.2),  # 8: 黄绿 - ship
        (0.8, 0.5, 0.5),  # 9: 粉红 - truck
    ]
    
    def _generate_class_samples(n_samples, class_id, rng):
        color = class_colors[class_id]
        samples = []
        for _ in range(n_samples):
            img = rng.randn(img_size, img_size, n_channels) * 0.1
            # 添加类别特定的颜色块
            block_size = rng.randint(8, 20)
            x = rng.randint(0, img_size - block_size)
            y = rng.randint(0, img_size - block_size)
            # 主色块
            img[x:x+block_size, y:y+block_size, 0] += color[0] * 0.6
            img[x:x+block_size, y:y+block_size, 1] += color[1] * 0.6
            img[x:x+block_size, y:y+block_size, 2] += color[2] * 0.6
            # 一些随机纹理
            img += rng.randn(img_size, img_size, n_channels) * 0.05
            img = np.clip(img, 0, 1)
            samples.append(img.flatten())
        return np.array(samples)
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    samples_per_class_train = num_train // n_classes
    samples_per_class_test = num_test // n_classes
    
    for cls in range(n_classes):
        X_train_list.append(_generate_class_samples(samples_per_class_train, cls, rng))
        y_train_list.append(np.full(samples_per_class_train, cls))
        X_test_list.append(_generate_class_samples(samples_per_class_test, cls, rng))
        y_test_list.append(np.full(samples_per_class_test, cls))
    
    X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test_list, axis=0)
    
    # 缩放到0-255范围以模拟真实CIFAR-10
    X_train = (X_train * 255).astype(np.float32)
    X_test = (X_test * 255).astype(np.float32)
    
    # 打乱
    train_idx = rng.permutation(len(X_train))
    test_idx = rng.permutation(len(X_test))
    
    return X_train[train_idx], y_train[train_idx], X_test[test_idx], y_test[test_idx]


def load_cifar10_or_fallback(cache_dir="./cifar10_cache"):
    """
    加载CIFAR-10，如果失败则使用合成数据
    返回: (X_train, y_train, X_test, y_test, is_synthetic)
    """
    result = load_cifar10(cache_dir)
    if result is not None:
        return (*result, False)
    
    st.warning("⚠️ CIFAR-10 下载失败，使用合成数据替代演示。算法原理相同，但数据为生成的彩色图像。")
    X_train, y_train, X_test, y_test = generate_synthetic_cifar10()
    return X_train, y_train, X_test, y_test, True


def get_cifar10_subset(X_train, y_train, X_test, y_test,
                       classes=None, num_train_per_class=100,
                       num_test_per_class=20, seed=42):
    """
    从CIFAR-10中提取子集
    """
    rng = np.random.RandomState(seed)
    
    if classes is None:
        classes = list(range(10))
    
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_train_idx = np.where(y_train == cls)[0]
        cls_test_idx = np.where(y_test == cls)[0]
        
        chosen_train = rng.choice(cls_train_idx, 
                                  min(num_train_per_class, len(cls_train_idx)),
                                  replace=False)
        chosen_test = rng.choice(cls_test_idx,
                                 min(num_test_per_class, len(cls_test_idx)),
                                 replace=False)
        
        train_indices.extend(chosen_train)
        test_indices.extend(chosen_test)
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    
    return (X_train[train_indices], y_train[train_indices],
            X_test[test_indices], y_test[test_indices])


def normalize_cifar10(X_train, X_test):
    """对CIFAR-10数据进行归一化（减去均值，除以标准差）"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


# ============================================================
# 3. KNN 分类器
# ============================================================
class KNNClassifier:
    """K近邻分类器（使用L2距离）"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """
        预测类别
        使用向量化L2距离计算
        """
        # dists[i, j] = ||X[i] - X_train[j]||^2
        dists = (
            np.sum(X ** 2, axis=1, keepdims=True) +
            np.sum(self.X_train ** 2, axis=1) -
            2 * X @ self.X_train.T
        )
        dists = np.maximum(dists, 0)  # 消除数值误差
        
        k_nearest = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        k_labels = self.y_train[k_nearest]
        
        predictions = []
        for lbls in k_labels:
            counts = np.bincount(lbls, minlength=10)
            predictions.append(np.argmax(counts))
        
        return np.array(predictions)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    def predict_with_neighbors(self, X, max_k=10):
        """
        预测并返回直到max_k的所有K值结果
        用于批量比较不同K值
        """
        dists = (
            np.sum(X ** 2, axis=1, keepdims=True) +
            np.sum(self.X_train ** 2, axis=1) -
            2 * X @ self.X_train.T
        )
        dists = np.maximum(dists, 0)
        
        sorted_indices = np.argsort(dists, axis=1)
        
        results = {}
        for k in [1, 3, 5, 10]:
            if k > max_k:
                continue
            k_indices = sorted_indices[:, :k]
            k_labels = self.y_train[k_indices]
            predictions = []
            for lbls in k_labels:
                counts = np.bincount(lbls, minlength=10)
                predictions.append(np.argmax(counts))
            results[k] = np.array(predictions)
        
        return results


# ============================================================
# 4. 线性分类器（手动实现）
# ============================================================
class LinearClassifier:
    """
    线性分类器，支持 Softmax (Cross-Entropy) 和 SVM (Hinge) 损失
    支持 SGD 和 Momentum 优化器
    """
    
    def __init__(self, n_classes=10, n_features=3072, loss_type='softmax'):
        self.n_classes = n_classes
        self.n_features = n_features
        self.loss_type = loss_type
        
        # He初始化
        self.W = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / n_features)
        self.b = np.zeros(n_classes)
        
        # 训练历史
        self.loss_history = []
        self.acc_history = []
    
    def reset_weights(self):
        """重置权重"""
        self.W = np.random.randn(self.n_features, self.n_classes) * np.sqrt(2.0 / self.n_features)
        self.b = np.zeros(self.n_classes)
        self.loss_history = []
        self.acc_history = []
    
    def softmax_loss(self, scores, y_batch):
        """计算Softmax (Cross-Entropy) 损失和梯度"""
        # 数值稳定性：减去最大值
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        batch_size = scores.shape[0]
        correct_log_probs = -np.log(probs[np.arange(batch_size), y_batch] + 1e-15)
        loss = np.mean(correct_log_probs)
        
        # 梯度
        dscores = probs.copy()
        dscores[np.arange(batch_size), y_batch] -= 1
        dscores /= batch_size
        
        return loss, dscores
    
    def svm_loss(self, scores, y_batch, delta=1.0):
        """计算SVM (Hinge) 损失和梯度"""
        batch_size = scores.shape[0]
        correct_scores = scores[np.arange(batch_size), y_batch].reshape(-1, 1)
        
        margins = np.maximum(0, scores - correct_scores + delta)
        margins[np.arange(batch_size), y_batch] = 0
        
        loss = np.sum(margins) / batch_size
        
        # 梯度
        dscores = np.zeros_like(scores)
        dscores[margins > 0] = 1
        dscores[np.arange(batch_size), y_batch] = -np.sum(margins > 0, axis=1)
        dscores /= batch_size
        
        return loss, dscores
    
    def train(self, X, y, lr=1e-3, epochs=100, optimizer='sgd',
              momentum=0.9, batch_size=64, reg=0.0, verbose=True):
        """
        训练线性分类器
        
        参数:
            X: 训练数据, shape (N, D)
            y: 标签, shape (N,)
            lr: 学习率
            epochs: 训练轮数
            optimizer: 'sgd' 或 'momentum'
            momentum: 动量系数
            batch_size: 批次大小
            reg: L2正则化强度
            verbose: 是否显示进度
        """
        n_samples = X.shape[0]
        self.loss_history = []
        self.acc_history = []
        
        # 动量缓存
        v_W = np.zeros_like(self.W)
        v_b = np.zeros_like(self.b)
        
        if verbose:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for epoch in range(epochs):
            # 打乱数据
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                curr_batch_size = X_batch.shape[0]
                
                # 前向传播
                scores = X_batch @ self.W + self.b
                
                # 计算损失和梯度
                if self.loss_type == 'softmax':
                    loss, dscores = self.softmax_loss(scores, y_batch)
                else:
                    loss, dscores = self.svm_loss(scores, y_batch)
                
                # 加上正则化项
                loss += 0.5 * reg * np.sum(self.W * self.W)
                
                # 梯度
                dW = X_batch.T @ dscores + reg * self.W
                db = np.sum(dscores, axis=0)
                
                # 参数更新
                if optimizer == 'sgd':
                    self.W -= lr * dW
                    self.b -= lr * db
                elif optimizer == 'momentum':
                    v_W = momentum * v_W + lr * dW
                    v_b = momentum * v_b + lr * db
                    self.W -= v_W
                    self.b -= v_b
                
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)
            
            # 计算训练准确率
            train_preds = self.predict(X)
            acc = np.mean(train_preds == y)
            self.acc_history.append(acc)
            
            if verbose:
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
        
        if verbose:
            status_text.text("训练完成！")
        
        return self.loss_history, self.acc_history
    
    def predict(self, X):
        """预测类别"""
        scores = X @ self.W + self.b
        return np.argmax(scores, axis=1)
    
    def score(self, X, y):
        """计算准确率"""
        preds = self.predict(X)
        return np.mean(preds == y)
    
    def get_template_images(self):
        """
        获取每个类别的模板图像
        将权重矩阵的每一列（对应一个类别）reshape回32x32x3
        """
        templates = []
        for i in range(self.n_classes):
            w = self.W[:, i]
            # 归一化到[0, 1]
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w_norm = (w - w_min) / (w_max - w_min)
            else:
                w_norm = np.zeros_like(w)
            # reshape为32x32x3
            template = w_norm.reshape(3, 32, 32).transpose(1, 2, 0)
            templates.append(template)
        return templates


# ============================================================
# 5. 损失函数可视化工具
# ============================================================
def compute_loss_demo(scores, y_true, loss_type='softmax', delta=1.0):
    """
    计算给定分数和真实标签的损失值（用于演示）
    scores: shape (N, C) 的原始分数
    y_true: shape (N,) 的真实标签
    """
    if loss_type == 'softmax':
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        batch_size = scores.shape[0]
        loss = -np.mean(np.log(probs[np.arange(batch_size), y_true] + 1e-15))
        return loss, probs
    else:  # svm
        batch_size = scores.shape[0]
        correct_scores = scores[np.arange(batch_size), y_true].reshape(-1, 1)
        margins = np.maximum(0, scores - correct_scores + delta)
        margins[np.arange(batch_size), y_true] = 0
        loss = np.sum(margins) / batch_size
        # 计算每个样本的合页损失用于展示
        per_sample_loss = np.sum(margins, axis=1)
        return loss, per_sample_loss


def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    """Plot confusion matrix"""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    
    # 在格子中显示数字
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_sample_images(images, labels, predictions, class_names, num_samples=10):
    """Plot sample images with their predictions"""
    n = min(num_samples, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(n):
        # CIFAR-10 image format: (3072,) -> reshape to (3, 32, 32) -> transpose to (32, 32, 3)
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[i].imshow(img)
        true_label = class_names[labels[i]]
        if predictions is not None:
            pred_label = class_names[predictions[i]]
            color = 'green' if labels[i] == predictions[i] else 'red'
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", 
                            fontsize=8, color=color)
        else:
            axes[i].set_title(f"Label: {true_label}", fontsize=9)
        axes[i].axis('off')
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
