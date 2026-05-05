"""
工具函数模块 - 包含各模块共用的辅助函数
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配Streamlit
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import base64
from PIL import Image

# ============================================================
# 中文字体设置 - 尝试多个常见中文字体
# ============================================================
def setup_chinese_font():
    """尝试设置中文字体，如果找不到则使用默认字体"""
    chinese_fonts = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Source Han Sans SC',
        'STHeiti', 'AR PL UMing CN'
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


# 初始化字体
_CN_FONT = setup_chinese_font()


def fig_to_base64(fig, dpi=100):
    """将matplotlib图像转为base64编码的HTML img标签"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;">'


def fig_to_pil(fig):
    """将matplotlib图像转为PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def numpy_to_pil(arr):
    """将numpy数组转为PIL Image（支持灰度与RGB）"""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode='RGB')
    return Image.fromarray(arr)
