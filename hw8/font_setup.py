"""
matplotlib 中文字体配置
必须在任何 matplotlib 绘图之前调用
"""

import matplotlib.pyplot as plt
import matplotlib
import platform


def setup_chinese_font():
    """配置 matplotlib 支持中文显示"""
    system = platform.system()

    if system == "Windows":
        # Windows 常用中文字体
        font_names = ["Microsoft YaHei", "SimHei", "KaiTi", "FangSong"]
    elif system == "Darwin":  # macOS
        font_names = ["PingFang SC", "Heiti SC", "STHeiti"]
    else:  # Linux
        font_names = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]

    for font in font_names:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
            return font
        except Exception:
            continue

    # 如果都找不到，尝试从系统获取
    try:
        available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        cjk_fonts = [f for f in available if any(
            keyword in f.lower() for keyword in ["hei", "song", "ming", "kai", "cjk", "yahei"]
        )]
        if cjk_fonts:
            plt.rcParams["font.sans-serif"] = [cjk_fonts[0], "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return cjk_fonts[0]
    except Exception:
        pass

    # 最后 fallback
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


# 模块导入时自动配置
_configured_font = setup_chinese_font()
print(f"[font] matplotlib 中文字体配置: {_configured_font}")
