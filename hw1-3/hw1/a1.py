"""
作业1：颜色空间转换与图像插值
要求：
1. 实现RGB、HSV等颜色空间相关的实现（输入图像，输出不同颜色空间中每个通道的图像）
2. 实现图像插值算法：最近邻、双线性插值（输入图像，选择图像操作：放大、缩小、旋转拉伸等，输出插值后的图像）
"""

import cv2
import numpy as np


# ==================== 颜色空间转换 ====================

def get_color_space_channels(image, color_space="RGB"):
    """
    将图像转换到指定颜色空间并返回各个通道
    
    参数:
        image: BGR格式的输入图像
        color_space: 目标颜色空间，可选 "RGB", "HSV", "Lab", "YCrCb"
    
    返回:
        (channels, channel_names): 通道图像列表和通道名称列表
    """
    if color_space == "RGB":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        channels = cv2.split(rgb)
        channel_names = ["Red (R)", "Green (G)", "Blue (B)"]
    elif color_space == "HSV":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = cv2.split(hsv)
        channel_names = ["Hue (H)", "Saturation (S)", "Value (V)"]
    elif color_space == "Lab":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        channels = cv2.split(lab)
        channel_names = ["Lightness (L)", "Green-Red (a)", "Blue-Yellow (b)"]
    elif color_space == "YCrCb":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        channel_names = ["Luminance (Y)", "Chrominance-Red (Cr)", "Chrominance-Blue (Cb)"]
    else:
        channels = cv2.split(image)
        channel_names = ["Channel 1", "Channel 2", "Channel 3"]
    
    return channels, channel_names


def convert_color_space(image, target_space):
    """
    将BGR图像转换为目标颜色空间
    
    参数:
        image: BGR格式输入图像
        target_space: 目标颜色空间 "RGB", "HSV", "Lab", "YCrCb"
    
    返回:
        转换后的图像
    """
    if target_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif target_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif target_space == "Lab":
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif target_space == "YCrCb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image


# ==================== 图像插值算法 ====================

def nearest_neighbor_interpolation(image, new_width, new_height):
    """
    最近邻插值算法
    
    参数:
        image: 输入图像
        new_width: 目标宽度
        new_height: 目标高度
    
    返回:
        插值后的图像
    """
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        channels = image.shape[2]
        result = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    else:
        channels = 1
        result = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # 计算缩放比例
    row_ratio = h / new_height
    col_ratio = w / new_width
    
    for dst_y in range(new_height):
        # 找到最近邻的源坐标
        src_y = int(round(dst_y * row_ratio))
        src_y = min(src_y, h - 1)
        
        for dst_x in range(new_width):
            src_x = int(round(dst_x * col_ratio))
            src_x = min(src_x, w - 1)
            
            if channels == 1:
                result[dst_y, dst_x] = image[src_y, src_x]
            else:
                result[dst_y, dst_x] = image[src_y, src_x]
    
    return result


def bilinear_interpolation(image, new_width, new_height):
    """
    双线性插值算法
    
    参数:
        image: 输入图像
        new_width: 目标宽度
        new_height: 目标高度
    
    返回:
        插值后的图像
    """
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        channels = image.shape[2]
        result = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    else:
        channels = 1
        result = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # 计算缩放比例
    row_ratio = h / new_height
    col_ratio = w / new_width
    
    for dst_y in range(new_height):
        # 计算源图像中的对应位置（浮点坐标）
        src_y = dst_y * row_ratio
        
        # 找到四个邻近像素的坐标
        y1 = int(np.floor(src_y))
        y2 = min(y1 + 1, h - 1)
        dy = src_y - y1
        
        for dst_x in range(new_width):
            src_x = dst_x * col_ratio
            x1 = int(np.floor(src_x))
            x2 = min(x1 + 1, w - 1)
            dx = src_x - x1
            
            if channels == 1:
                # 单通道双线性插值
                top = image[y1, x1] * (1 - dx) + image[y1, x2] * dx
                bottom = image[y2, x1] * (1 - dx) + image[y2, x2] * dx
                result[dst_y, dst_x] = top * (1 - dy) + bottom * dy
            else:
                for c in range(channels):
                    # 对每个通道进行双线性插值
                    top = image[y1, x1, c] * (1 - dx) + image[y1, x2, c] * dx
                    bottom = image[y2, x1, c] * (1 - dx) + image[y2, x2, c] * dx
                    result[dst_y, dst_x, c] = top * (1 - dy) + bottom * dy
    
    return result.astype(np.uint8)


def apply_interpolation(image, operation, params, method="bilinear"):
    """
    应用插值操作
    
    参数:
        image: 输入图像
        operation: 操作类型 "放大", "缩小", "旋转", "拉伸"
        params: 参数字典
            - 放大/缩小: {"scale": float}
            - 旋转: {"angle": float}
            - 拉伸: {"width": int, "height": int}
        method: 插值方法 "nearest" 或 "bilinear"
    
    返回:
        处理后的图像
    """
    h, w = image.shape[:2]
    
    if operation == "放大":
        scale = params.get("scale", 2.0)
        new_w, new_h = int(w * scale), int(h * scale)
    elif operation == "缩小":
        scale = params.get("scale", 0.5)
        new_w, new_h = int(w * scale), int(h * scale)
    elif operation == "旋转":
        angle = params.get("angle", 45)
        # 先计算旋转后的图像尺寸
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        # 调整旋转矩阵的平移部分
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        if method == "nearest":
            return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
        else:
            return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    elif operation == "拉伸":
        new_w = params.get("width", int(w * 1.5))
        new_h = params.get("height", int(h * 1.5))
    else:
        return image.copy()
    
    # 对放大、缩小、拉伸使用指定的插值方法
    if method == "nearest":
        return nearest_neighbor_interpolation(image, new_w, new_h)
    else:
        return bilinear_interpolation(image, new_w, new_h)


# ==================== OpenCV对比实现 ====================

def apply_interpolation_opencv(image, operation, params, method="bilinear"):
    """
    使用OpenCV内置函数进行插值（用于对比）
    """
    h, w = image.shape[:2]
    
    if method == "nearest":
        cv_method = cv2.INTER_NEAREST
    else:
        cv_method = cv2.INTER_LINEAR
    
    if operation == "放大":
        scale = params.get("scale", 2.0)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv_method)
    elif operation == "缩小":
        scale = params.get("scale", 0.5)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv_method)
    elif operation == "旋转":
        angle = params.get("angle", 45)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv_method)
    elif operation == "拉伸":
        new_w = params.get("width", int(w * 1.5))
        new_h = params.get("height", int(h * 1.5))
        return cv2.resize(image, (new_w, new_h), interpolation=cv_method)
    
    return image.copy()
