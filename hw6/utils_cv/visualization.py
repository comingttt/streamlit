"""
工具模块 - 可视化、指标计算、数据处理通用函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import cv2
import time
from typing import List, Dict, Tuple, Optional

# COCO 类别名称
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# VOC 类别+背景
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 用于可视化的颜色映射
COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
]


def generate_colormap(n: int) -> np.ndarray:
    """生成 n 种不同颜色的 colormap"""
    color_list = []
    for i in range(n):
        r = int((i * 67 + 123) % 256)
        g = int((i * 137 + 89) % 256)
        b = int((i * 211 + 47) % 256)
        color_list.append([r, g, b])
    return np.array(color_list, dtype=np.uint8)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL图像转numpy数组"""
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """numpy数组转PIL图像"""
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
    return Image.fromarray(array)


def draw_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    将分割mask叠加到原图上
    Args:
        image: (H, W, 3) RGB原图
        mask: (H, W) 分割mask，值为类别索引
        alpha: 透明度
        color_map: (num_classes, 3) 颜色映射
    Returns:
        overlay: (H, W, 3) 叠加图
    """
    if color_map is None:
        num_classes = max(mask.max() + 1, 22)
        color_map = generate_colormap(num_classes)

    # 创建彩色mask
    colored_mask = np.zeros_like(image)
    for cls_idx in np.unique(mask):
        if cls_idx == 0:
            continue  # 跳过背景
        colored_mask[mask == cls_idx] = color_map[cls_idx % len(color_map)]

    # 叠加
    overlay = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return overlay


def draw_detection_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str] = None,
    score_threshold: float = 0.5,
    line_width: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    绘制检测框
    Args:
        image: (H, W, 3) RGB原图
        boxes: (N, 4) [x1, y1, x2, y2] 格式
        labels: (N,) 类别索引
        scores: (N,) 置信度分数
        class_names: 类别名称列表
        score_threshold: 分数阈值
    Returns:
        带框的图像
    """
    if class_names is None:
        class_names = COCO_CLASSES

    img = image.copy()
    colors = generate_colormap(len(class_names))

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = colors[label % len(colors)].tolist()

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        # 标签文字
        if label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"cls_{label}"
        text = f"{class_name}: {score:.2f}"

        # 文字背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1)

    return img


def draw_instance_masks(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str] = None,
    score_threshold: float = 0.5,
    mask_alpha: float = 0.5,
) -> np.ndarray:
    """
    绘制实例分割结果
    Args:
        image: (H, W, 3) RGB原图
        masks: (N, H, W) 二值masks
        boxes: (N, 4) 边界框
        labels: (N,) 类别索引
        scores: (N,) 置信度分数
    Returns:
        带实例分割的结果图
    """
    if class_names is None:
        class_names = COCO_CLASSES

    img = image.copy()
    colors = generate_colormap(max(len(class_names), 82))

    # 先画所有mask
    for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        if score < score_threshold:
            continue

        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0.5] = color
        img = cv2.addWeighted(img, 1.0, colored_mask, mask_alpha, 0)

    # 再画框和标签
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < score_threshold:
            continue

        color = colors[i % len(colors)].tolist()
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"cls_{label}"
        text = f"{class_name}: {score:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return img


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> float:
    """计算像素精度"""
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum()
    total = mask.sum()
    return correct / total if total > 0 else 0.0


def compute_mean_iou(
    pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = 255
) -> Tuple[float, np.ndarray]:
    """计算Mean IoU"""
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    ious = np.array(ious)
    valid_ious = ious[~np.isnan(ious)]
    return valid_ious.mean() if len(valid_ious) > 0 else 0.0, ious


def count_parameters(model) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """将matplotlib figure转为numpy数组"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def create_comparison_figure(
    original: np.ndarray,
    results: List[Dict],
    titles: List[str],
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    创建对比图
    Args:
        original: 原图
        results: 结果图列表
        titles: 每张图的标题
    Returns:
        matplotlib Figure
    """
    n = len(results) + 1
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    for i, (result, title) in enumerate(zip(results, titles)):
        axes[i + 1].imshow(result)
        axes[i + 1].set_title(title, fontsize=12)
        axes[i + 1].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


class Timer:
    """简单的计时器上下文管理器"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

    @property
    def elapsed_ms(self) -> float:
        return getattr(self, 'elapsed', 0.0) * 1000
