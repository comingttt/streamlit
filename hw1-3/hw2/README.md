# 计算机视觉课程设计 - 图像处理演示系统

## 功能概述

本系统基于Python和OpenCV实现了一个综合的图像处理演示GUI应用程序，包含以下功能模块：

### 1. 空间滤波器演示 ⭐
- **Box滤波**：均值滤波，去除噪声
- **Gaussian滤波**：高斯平滑，保持边缘
- **Median滤波**：中值滤波，消除椒盐噪声
- **Sobel滤波**：边缘检测，计算梯度幅度

### 2. 图像梯度演示 ⭐
- 选择局部区域（可配置坐标）
- 计算梯度方向和幅度
- HSV颜色编码显示梯度信息（色相表示方向，亮度表示幅度）

### 3. 频率域滤波演示 ⭐
- 基于傅里叶变换的频谱分析
- 绘制频谱图（对数幅度谱）
- 比较图像在旋转、平移、缩放变换下的频谱变化
- 验证频域变换的性质

## 技术实现

### 核心库
- **OpenCV**: 图像处理和计算机视觉算法
- **NumPy**: 数值计算和数组操作
- **Matplotlib**: 频谱图绘制
- **PIL/Pillow**: 图像格式转换
- **Tkinter**: GUI界面

### 关键算法

#### 空间滤波器
```python
# Box滤波
cv2.blur(image, (kernel_size, kernel_size))

# Gaussian滤波
cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Median滤波
cv2.medianBlur(image, kernel_size)

# Sobel滤波
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
sobel = cv2.magnitude(sobelx, sobely)
```

#### 梯度计算
```python
# 计算梯度分量
sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度方向
gradient_direction = cv2.phase(sobelx, sobely, angleInDegrees=True)

# HSV编码显示
hsv[..., 0] = gradient_direction / 2  # 方向映射到0-180度
hsv[..., 1] = 255  # 饱和度最大
hsv[..., 2] = gradient_magnitude  # 幅度作为亮度
```

#### 频率域分析
```python
# 傅里叶变换
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

# 计算幅度谱
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 逆变换
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
```

## 使用方法

1. 运行程序：
```bash
python a2.py
```

2. 加载图像：点击"加载图像"按钮选择图片文件

3. 选择功能模块：
   - 颜色空间演示：调节颜色参数，观察通道分离
   - 图像插值演示：选择变换类型和插值算法，比较效果
   - 空间滤波器演示：选择滤波器类型和核大小，观察去噪效果
   - 图像梯度演示：设置区域坐标，计算并显示梯度
   - 频率域滤波演示：选择变换类型，观察频谱变化

## 实验结果分析

### 空间滤波器比较
- **Box滤波**：简单快速，但可能造成边缘模糊
- **Gaussian滤波**：保持边缘特征，去除高频噪声
- **Median滤波**：特别适合消除椒盐噪声，保持边缘锐度
- **Sobel滤波**：突出边缘信息，用于特征提取

### 梯度算法演示
- 梯度方向用HSV色相编码（0-360度映射到0-180度）
- 梯度幅度用亮度表示
- 红色表示水平边缘，蓝色表示垂直边缘

### 频率域特性验证
- **旋转不变性**：图像旋转不改变频谱形状
- **平移特性**：图像平移只影响相位谱，不改变幅度谱
- **缩放特性**：图像缩放导致频谱相应缩放

## 系统要求

- Python 3.10+
- OpenCV 4.13+
- NumPy 2.2+
- Matplotlib 3.10+
- PIL/Pillow 12.1+
- Tkinter (Python标准库)

## 注意事项

- GUI程序需要在有显示的环境中运行
- 建议使用高清图像以获得最佳效果
- 频率域分析对大图像可能较慢
- 梯度计算区域不应超出图像边界