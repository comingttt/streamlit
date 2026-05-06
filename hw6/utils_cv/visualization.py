"""
工具模块 - 可视化、指标计算、数据处理通用函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import io
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


def _get_font(size: int = 14):
    """获取PIL字体（尽力获取可用字体）"""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _blend(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """纯numpy图像混合，替代 cv2.addWeighted"""
    return np.clip(
        img1.astype(np.float32) * (1 - alpha) + img2.astype(np.float32) * alpha,
        0, 255
    ).astype(np.uint8)


def _draw_boxes_on_image(
    img_np: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    score_threshold: float,
    colors: np.ndarray,
    line_width: int = 2,
) -> np.ndarray:
    """用PIL在图像上绘制边界框和标签，返回numpy数组"""
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    font = _get_font(13)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = tuple(int(c) for c in colors[label % len(colors)])

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        class_name = class_names[label] if label < len(class_names) else f"cls_{label}"
        text = f"{class_name}: {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 文字背景
        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 4, y1)], fill=color)
        draw.text((x1 + 2, y1 - th - 2), text, fill=(255, 255, 255), font=font)

    return np.array(img_pil)


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
    overlay = _blend(image, colored_mask, alpha)
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

    colors = generate_colormap(len(class_names))
    return _draw_boxes_on_image(image, boxes, labels, scores, class_names, score_threshold, colors, line_width)


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
        img = _blend(img, colored_mask, mask_alpha)

    # 再画框和标签
    img = _draw_boxes_on_image(img, boxes, labels, scores, class_names, score_threshold, colors)

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
