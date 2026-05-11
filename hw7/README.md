# 🧠 自监督学习（Self-Supervised Learning）教学项目

一个基于 **PyTorch** 和 **Streamlit** 的交互式自监督学习教学演示应用。

## 🎯 项目概述

本项目实现了两种经典的自监督学习方法，帮助理解 SSL 的核心思想：无需人工标注，通过设计 pretext task 从数据本身生成监督信号。

### 模块 A：Rotation Prediction（旋转预测）
- 对 MNIST 图像随机旋转 0°/90°/180°/270°
- 训练 CNN 预测旋转角度（4 分类）
- 参考论文：Gidaris et al., ICLR 2018

### 模块 B：MAE（Masked Autoencoder）
- 随机遮挡图像 patch（支持 25%/50%/75%）
- 使用卷积 AutoEncoder 重建原图
- 参考论文：He et al., CVPR 2022

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 预训练模型（可选）
首次启动应用时，可以运行预训练脚本提前训练模型：
```bash
python pretrain.py
```

### 3. 启动 Web 应用
```bash
streamlit run app.py
```

然后在浏览器中打开 http://localhost:8501

## 📁 项目结构

```
hw7/
├── app.py                 # Streamlit 主程序
├── pretrain.py            # 预训练脚本
├── requirements.txt       # Python 依赖
├── README.md             # 项目说明
├── models/
│   ├── __init__.py
│   ├── rotation_cnn.py   # Rotation Prediction CNN 模型
│   └── mae.py            # MAE 自编码器模型
├── utils/
│   ├── __init__.py
│   ├── data_utils.py     # 数据加载与预处理
│   └── visualization.py  # 可视化工具函数
├── pretrained/           # 预训练模型保存目录
└── data/                 # MNIST 数据集下载目录（自动创建）
```

## 🖥️ 应用页面

| 页面 | 功能 |
|------|------|
| 🔄 Rotation Prediction | 训练旋转预测模型、可视化预测结果、上传图片测试 |
| 🧩 MAE Reconstruction | 训练 MAE 模型（可调遮挡比例）、可视化重建效果 |
| 📊 Comparison | 对比不同遮挡比例效果、训练前后对比、汇总表格 |
| 📖 About Project | 项目介绍、技术说明、参考文献 |

## 🛠️ 技术栈

- **PyTorch**：深度学习模型训练
- **Streamlit**：交互式 Web 界面
- **Matplotlib + Plotly**：数据可视化
- **MNIST**：训练数据集（通过 torchvision 自动下载）

## 📊 模型说明

### Rotation CNN
- 输入：28×28 灰度图
- 架构：Conv(1→32) → Conv(32→64) → FC(64×7×7→128) → FC(128→4)
- 训练时间：CPU 约 1-2 分钟（5 epochs）

### MAE Autoencoder
- 编码器：Conv(1→16) → Conv(16→32) → Conv(32→64)
- 解码器：ConvTranspose(64→32) → ConvTranspose(32→16) → ConvTranspose(16→1)
- 训练时间：CPU 约 2-3 分钟（5 epochs）

## 📚 参考文献

1. Gidaris, S., Singh, P., & Komodakis, N. (2018). Unsupervised Representation Learning by Predicting Image Rotations. *ICLR*.
2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.
3. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.
