"""
模块2: 神经网络反向传播演示
=============================
从零开始实现一个2层前馈神经网络（仅使用NumPy），
手动实现前向传播、损失计算、反向传播的完整流程。

网络结构: Input → Hidden (ReLU) → Output (Softmax + CrossEntropy)

核心概念：
- 前向传播: 逐层计算激活值
- 损失函数: 交叉熵损失衡量预测与真实分布的差异
- 反向传播: 链式法则计算梯度，从输出层反向传播到输入层
- 梯度下降: 沿负梯度方向更新参数
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. 神经网络类（纯NumPy实现）
# ============================================================
class TwoLayerNet:
    """
    两层全连接神经网络

    结构:
        Input (d) → Hidden (h, ReLU) → Output (k, Softmax)

    数学原理：
    - 前向传播:
        Z1 = X @ W1 + b1        (线性变换)
        A1 = ReLU(Z1)            (激活函数: max(0, z))
        Z2 = A1 @ W2 + b2        (第二层线性变换)
        A2 = Softmax(Z2)         (输出概率分布)

    - 交叉熵损失:
        L = -Σ y_i * log(p_i)    (p_i是Softmax输出)

    - 反向传播（链式法则）:
        dZ2 = A2 - Y             (Softmax+CE的联合梯度)
        dW2 = A1.T @ dZ2         (W2的梯度)
        db2 = sum(dZ2)           (b2的梯度)
        dA1 = dZ2 @ W2.T         (传播到第一层)
        dZ1 = dA1 * (Z1 > 0)     (ReLU的梯度)
        dW1 = X.T @ dZ1          (W1的梯度)
        db1 = sum(dZ1)           (b1的梯度)
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1,
                 reg_lambda=0.01):
        """
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层神经元数
            output_size: 输出类别数
            learning_rate: 学习率
            reg_lambda: L2正则化强度
        """
        self.lr = learning_rate
        self.reg_lambda = reg_lambda

        # He初始化：适用于ReLU激活函数
        # 权重从 N(0, sqrt(2/fan_in)) 采样，有助于保持梯度稳定
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 用于记录训练历史
        self.history = {
            'loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def _relu(self, Z):
        """ReLU激活函数: f(z) = max(0, z)"""
        return np.maximum(0, Z)

    def _softmax(self, Z):
        """
        Softmax函数: 将logits转为概率分布
        p_i = exp(z_i) / Σ exp(z_j)

        使用数值稳定技巧：减去每行最大值防止溢出
        """
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        """
        前向传播

        参数:
            X: 输入数据 (N, input_size)
        返回:
            A2: Softmax输出概率 (N, output_size)
            cache: 中间计算结果（用于反向传播）
        """
        # 第一层：线性变换 + ReLU激活
        Z1 = X @ self.W1 + self.b1       # (N, hidden_size)
        A1 = self._relu(Z1)               # (N, hidden_size)

        # 第二层：线性变换 + Softmax
        Z2 = A1 @ self.W2 + self.b2       # (N, output_size)
        A2 = self._softmax(Z2)             # (N, output_size)

        # 缓存中间值用于反向传播
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, A2, Y):
        """
        计算交叉熵损失（含L2正则化）

        L = -(1/N) * Σ Σ y_ij * log(p_ij) + (λ/2) * (||W1||² + ||W2||²)

        交叉熵衡量两个概率分布之间的差异：
        - 预测正确且置信度高 → loss小
        - 预测错误 → loss大

        参数:
            A2: Softmax概率输出 (N, output_size)
            Y: one-hot真实标签 (N, output_size)
        返回:
            loss: 标量损失值
        """
        N = A2.shape[0]

        # 交叉熵损失（加小量防止log(0)）
        cross_entropy = -np.sum(Y * np.log(A2 + 1e-8)) / N

        # L2正则化项
        l2_reg = (self.reg_lambda / 2) * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
        )

        return cross_entropy + l2_reg

    def backward(self, cache, Y):
        """
        反向传播：计算所有参数的梯度

        核心：链式法则 (Chain Rule)
        从输出层开始，逐层向前计算梯度

        关键推导：
        - Softmax + 交叉熵的联合梯度非常简洁: dL/dZ2 = A2 - Y
          （这是Softmax+CE组合被广泛使用的原因之一）

        - ReLU的梯度: d(ReLU(z))/dz = 1 if z > 0 else 0

        参数:
            cache: 前向传播的缓存 (X, Z1, A1, Z2, A2)
            Y: one-hot真实标签 (N, output_size)
        返回:
            grads: 包含所有参数梯度的字典
        """
        X, Z1, A1, Z2, A2 = cache
        N = X.shape[0]

        # ---- 输出层梯度 ----
        # dL/dZ2 = Softmax - Y  (Softmax+交叉熵的优雅联合梯度)
        dZ2 = A2 - Y                              # (N, output_size)

        # dL/dW2 = A1^T @ dZ2 / N + λ * W2
        dW2 = (A1.T @ dZ2) / N + self.reg_lambda * self.W2   # (hidden, output)

        # dL/db2 = mean(dZ2, axis=0)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N          # (1, output)

        # ---- 隐藏层梯度 ----
        # 梯度从输出层传播到隐藏层
        dA1 = dZ2 @ self.W2.T                     # (N, hidden)

        # ReLU的梯度: 当Z1 > 0时为1，否则为0
        dZ1 = dA1 * (Z1 > 0)                      # (N, hidden)

        # dL/dW1 = X^T @ dZ1 / N + λ * W1
        dW1 = (X.T @ dZ1) / N + self.reg_lambda * self.W1     # (input, hidden)

        # dL/db1 = mean(dZ1, axis=0)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N          # (1, hidden)

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }
        return grads

    def update_params(self, grads):
        """
        梯度下降更新参数
        参数更新公式: θ = θ - lr * ∇θ

        lr（学习率）控制每次更新的步长：
        - 太大：可能震荡或发散
        - 太小：收敛慢
        """
        self.W1 -= self.lr * grads['W1']
        self.b1 -= self.lr * grads['b1']
        self.W2 -= self.lr * grads['W2']
        self.b2 -= self.lr * grads['b2']

    def train_step(self, X, Y):
        """单步训练：前向 → 计算损失 → 反向传播 → 更新参数"""
        A2, cache = self.forward(X)
        loss = self.compute_loss(A2, Y)
        grads = self.backward(cache, Y)
        self.update_params(grads)
        return loss, A2

    def predict(self, X):
        """预测：前向传播后取argmax"""
        A2, _ = self.forward(X)
        return np.argmax(A2, axis=1)


# ============================================================
# 2. 数据生成与训练
# ============================================================
def generate_data(dataset_type='moons', n_samples=500, noise=0.2):
    """
    生成2D分类数据用于可视化演示。

    参数:
        dataset_type: 'moons' | 'circles' | 'blobs'
        n_samples: 样本数量
        noise: 噪声水平
    返回:
        X_train, X_val, y_train, y_val (one-hot格式)
    """
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                          random_state=42, cluster_std=1.5)

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 转换为one-hot编码
    n_classes = len(np.unique(y))
    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_oh = encoder.transform(y_val.reshape(-1, 1))

    return X_train, X_val, y_train_oh, y_val_oh, y_train, y_val, n_classes


def train_network(X_train, y_train_oh, X_val, y_val_oh, y_val,
                  hidden_size=20, learning_rate=0.5, epochs=1000,
                  reg_lambda=0.01, output_callback=None):
    """
    训练神经网络

    参数:
        X_train, y_train_oh: 训练数据和one-hot标签
        X_val, y_val_oh, y_val: 验证数据
        hidden_size: 隐藏层大小
        learning_rate: 学习率
        epochs: 训练轮数
        reg_lambda: L2正则化系数
        output_callback: 每N轮回调函数 callback(epoch, loss, train_acc, val_acc)
    返回:
        model: 训练好的模型
    """
    input_dim = X_train.shape[1]
    output_dim = y_train_oh.shape[1]

    model = TwoLayerNet(
        input_size=input_dim,
        hidden_size=hidden_size,
        output_size=output_dim,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda
    )

    # 学习率衰减调度
    initial_lr = learning_rate

    for epoch in range(epochs):
        # 学习率指数衰减
        model.lr = initial_lr * (0.999 ** epoch)

        # 训练一步
        loss, A2_train = model.train_step(X_train, y_train_oh)

        # 计算准确率
        train_pred = np.argmax(A2_train, axis=1)
        train_acc = np.mean(train_pred == np.argmax(y_train_oh, axis=1))

        A2_val, _ = model.forward(X_val)
        val_pred = np.argmax(A2_val, axis=1)
        val_acc = np.mean(val_pred == y_val)

        # 记录历史
        model.history['loss'].append(loss)
        model.history['train_acc'].append(train_acc)
        model.history['val_acc'].append(val_acc)

        # 回调
        if output_callback and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
            output_callback(epoch, loss, train_acc, val_acc)

    return model


# ============================================================
# 3. 可视化函数
# ============================================================
def plot_decision_boundary(model, X, y, title='Decision Boundary'):
    """
    绘制2D决策边界（仅适用于2D输入数据）
    展示神经网络学到的非线性分类边界
    """
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格上每个点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k',
                         cmap=plt.cm.RdYlBu, s=40)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    return fig


def plot_training_curves(history):
    """绘制训练loss和准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(len(history['loss']))

    # Loss曲线
    ax1.plot(epochs, history['loss'], 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=1.5)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gradient_flow(model, X_sample, y_sample_oh):
    """
    可视化梯度在各层的分布（梯度流）
    用于检查是否存在梯度消失/爆炸问题
    """
    _, cache = model.forward(X_sample)
    grads = model.backward(cache, y_sample_oh)

    layers = ['W1', 'W2']
    grad_means = []
    grad_stds = []

    for layer in layers:
        g = grads[layer]
        grad_means.append(np.mean(np.abs(g)))
        grad_stds.append(np.std(g))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, grad_means, width, label='Mean |grad|', color='steelblue')
    bars2 = ax.bar(x + width/2, grad_stds, width, label='Std of grad', color='coral')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Across Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# 4. 主流程函数
# ============================================================
def run_backprop_demo(dataset_type='moons', hidden_size=20, learning_rate=0.5,
                       epochs=1000, reg_lambda=0.01):
    """
    运行完整的反向传播演示。

    返回:
        results: 包含所有结果和图像的字典
    """
    results = {}

    # 生成数据
    X_train, X_val, y_train_oh, y_val_oh, y_train, y_val, n_classes = generate_data(
        dataset_type=dataset_type
    )
    results['n_train'] = len(X_train)
    results['n_val'] = len(X_val)
    results['n_classes'] = n_classes
    results['input_dim'] = X_train.shape[1]

    # 训练日志
    log_lines = []

    def log_callback(epoch, loss, train_acc, val_acc):
        line = f"Epoch {epoch:5d} | Loss: {loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
        log_lines.append(line)

    # 训练
    model = train_network(
        X_train, y_train_oh, X_val, y_val_oh, y_val,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
        reg_lambda=reg_lambda,
        output_callback=log_callback
    )
    results['log'] = '\n'.join(log_lines)
    results['history'] = model.history
    results['final_train_acc'] = model.history['train_acc'][-1]
    results['final_val_acc'] = model.history['val_acc'][-1]
    results['final_loss'] = model.history['loss'][-1]

    # 训练曲线图
    results['curve_fig'] = plot_training_curves(model.history)

    # 决策边界图
    results['boundary_fig'] = plot_decision_boundary(
        model, X_train, y_train, title='Decision Boundary (Train Set)'
    )

    # 梯度流图
    X_sample = X_train[:min(200, len(X_train))]
    y_sample_oh = y_train_oh[:min(200, len(X_train))]
    results['grad_flow_fig'] = plot_gradient_flow(model, X_sample, y_sample_oh)

    # 参数信息
    results['params'] = {
        'W1_shape': model.W1.shape,
        'W2_shape': model.W2.shape,
        'total_params': model.W1.size + model.W2.size + model.b1.size + model.b2.size,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
    }

    return results
