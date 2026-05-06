"""
模块4: 方法性能对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List
import time


def get_comparison_data() -> pd.DataFrame:
    """
    获取三种方法的对比数据

    数据来源：
    - FCN: 基于Pascal VOC数据集的典型指标
    - Faster R-CNN: 基于COCO数据集的典型指标
    - Mask R-CNN: 基于COCO数据集的典型指标
    """
    data = {
        '方法': ['FCN (语义分割)', 'Faster R-CNN (目标检测)', 'Mask R-CNN (实例分割)'],
        '任务类型': ['语义分割', '目标检测', '实例分割'],
        'mIoU / mAP (%)': [67.2, 37.0, 37.8],
        '推理时间 (ms/图)': [45, 85, 120],
        '参数量 (百万)': [23.5, 41.8, 44.0],
        '骨干网络': ['VGG16', 'ResNet50+FPN', 'ResNet50+FPN'],
        '输出粒度': ['像素级（类别）', '框级（类别）', '像素级（实例）'],
        '可区分实例': ['否', '是（框级别）', '是（mask级别）'],
        '典型应用': ['场景理解', '物体定位', '精细分割'],
    }

    return pd.DataFrame(data)


def create_comparison_charts() -> Dict[str, go.Figure]:
    """创建对比图表"""
    df = get_comparison_data()
    charts = {}

    # 1. 精度对比柱状图
    fig1 = go.Figure(data=[
        go.Bar(
            x=df['方法'].tolist(),
            y=df['mIoU / mAP (%)'].tolist(),
            text=df['mIoU / mAP (%)'].tolist(),
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            name='精度指标'
        )
    ])
    fig1.update_layout(
        title='精度指标对比 (mIoU / mAP)',
        yaxis_title='mIoU / mAP (%)',
        xaxis_title='方法',
        template='plotly_white',
        height=400,
    )
    charts['accuracy'] = fig1

    # 2. 推理时间对比
    fig2 = go.Figure(data=[
        go.Bar(
            x=df['方法'].tolist(),
            y=df['推理时间 (ms/图)'].tolist(),
            text=df['推理时间 (ms/图)'].tolist(),
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            name='推理时间'
        )
    ])
    fig2.update_layout(
        title='推理时间对比 (ms/图)',
        yaxis_title='推理时间 (ms)',
        xaxis_title='方法',
        template='plotly_white',
        height=400,
    )
    charts['inference_time'] = fig2

    # 3. 参数量对比
    fig3 = go.Figure(data=[
        go.Bar(
            x=df['方法'].tolist(),
            y=df['参数量 (百万)'].tolist(),
            text=df['参数量 (百万)'].tolist(),
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            name='参数量'
        )
    ])
    fig3.update_layout(
        title='模型参数量对比 (百万)',
        yaxis_title='参数量 (百万)',
        xaxis_title='方法',
        template='plotly_white',
        height=400,
    )
    charts['params'] = fig3

    # 4. 综合雷达图
    categories = ['精度', '速度\n(归一化)', '轻量性\n(归一化)', '功能\n丰富度']
    # 归一化：精度越高越好，推理时间越低越好，参数量越低越好
    acc = df['mIoU / mAP (%)'].values
    time_normalized = 1.0 - (df['推理时间 (ms/图)'].values / df['推理时间 (ms/图)'].max())
    params_normalized = 1.0 - (df['参数量 (百万)'].values / df['参数量 (百万)'].max())
    capability = [1, 2, 3]  # 功能丰富度评分

    fig4 = go.Figure()
    for i, method in enumerate(df['方法']):
        fig4.add_trace(go.Scatterpolar(
            r=[
                acc[i] / acc.max(),
                time_normalized[i],
                params_normalized[i],
                capability[i] / 3.0,
            ],
            theta=categories,
            fill='toself',
            name=method,
        ))

    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='综合能力雷达图 (归一化)',
        template='plotly_white',
        height=450,
    )
    charts['radar'] = fig4

    return charts


def get_comparison_summary() -> str:
    """获取对比分析总结"""
    return """
### 📊 方法对比分析总结

---

#### 1. **FCN (语义分割)**
- **核心思想**: 全卷积网络，将分类网络改造为像素级预测
- **优势**: 
  - 结构简单，推理速度快
  - 适合场景级别的语义理解
  - 参数量相对较少
- **局限**: 
  - 无法区分同一类别的不同实例
  - 边界精度相对较低（FCN-32s）
  - 对小物体分割效果不佳

#### 2. **Faster R-CNN (目标检测)**
- **核心思想**: RPN + 检测网络，端到端区域提议
- **优势**: 
  - 检测精度高，定位准确
  - 可区分不同实例（框级别）
  - RPN实现高效区域提议
- **局限**: 
  - 输出仅为边界框，不是像素级分割
  - 推理速度较FCN慢
  - 对小物体检测能力受限

#### 3. **Mask R-CNN (实例分割)**
- **核心思想**: 在Faster R-CNN基础上增加Mask分支
- **优势**: 
  - 同时提供检测框和像素级分割
  - 可精细区分每个实例的轮廓
  - RoIAlign解决量化误差
- **局限**: 
  - 推理速度最慢
  - 参数量最大
  - 训练更为复杂

---

#### 🎯 **选择建议**
| 应用场景 | 推荐方法 |
|---------|---------|
| 场景语义理解 | FCN |
| 物体检测与定位 | Faster R-CNN |
| 精细实例分割 | Mask R-CNN |
"""


def run_benchmark(
    detectors: Dict[str, object],
    test_image: np.ndarray,
    num_runs: int = 3
) -> pd.DataFrame:
    """
    运行性能基准测试

    Args:
        detectors: {'name': detector_object} 字典
        test_image: 测试图像
        num_runs: 每个方法运行次数

    Returns:
        基准测试结果DataFrame
    """
    results = []

    for name, detector in detectors.items():
        times = []
        for _ in range(num_runs):
            start = time.time()

            if hasattr(detector, 'predict'):
                detector.predict(test_image)
            elif hasattr(detector, 'predict_single'):
                detector.predict_single(test_image)

            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # 获取参数量
        num_params = 0
        if hasattr(detector, 'get_model_info'):
            info = detector.get_model_info()
            num_params = info.get('num_params', 0)
        elif hasattr(detector, 'model'):
            from utils_cv.visualization import count_parameters
            num_params = count_parameters(detector.model)

        results.append({
            '方法': name,
            '平均推理时间 (ms)': round(avg_time, 1),
            '标准差 (ms)': round(std_time, 1),
            '参数量': f'{num_params / 1e6:.1f}M' if num_params > 1e6 else str(num_params),
        })

    return pd.DataFrame(results)
