"""
现代视觉任务（分割与检测）- Streamlit交互应用

启动方式: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import sys
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="现代视觉任务 - 分割与检测",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils_cv.visualization import (
    COCO_CLASSES, VOC_CLASSES, Timer, count_parameters,
    draw_segmentation_mask, generate_colormap
)
from modules.fcn_module import FCNSegmenter, get_pretrained_segmentation_model
from modules.rcnn_module import FasterRCNNDetector, SimpleRCNNDetector
from modules.mask_rcnn_module import MaskRCNNSegmenter
from modules.comparison import (
    get_comparison_data, create_comparison_charts,
    get_comparison_summary, run_benchmark
)


# ==================== 缓存模型加载 ====================

@st.cache_resource
def load_rcnn_models():
    """加载R-CNN系列模型（缓存）"""
    with st.spinner("正在加载 Faster R-CNN 模型..."):
        faster_rcnn = FasterRCNNDetector(pretrained=True)
    return faster_rcnn


@st.cache_resource
def load_mask_rcnn_model():
    """加载Mask R-CNN模型（缓存）"""
    with st.spinner("正在加载 Mask R-CNN 模型..."):
        mask_rcnn = MaskRCNNSegmenter(pretrained=True)
    return mask_rcnn


# ==================== UI 组件 ====================

def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("🎯 现代视觉任务")
    st.sidebar.markdown("---")

    # 模块选择
    module = st.sidebar.radio(
        "选择模块",
        [
            "🏠 首页",
            "🔬 FCN 语义分割",
            "📦 R-CNN 目标检测",
            "🎨 Mask R-CNN 实例分割",
            "📊 方法性能对比",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 关于")
    st.sidebar.info(
        """
        本项目实现了三种现代视觉任务方法：
        - **FCN**: 全卷积语义分割
        - **R-CNN系列**: 目标检测演进
        - **Mask R-CNN**: 实例分割
        
        基于 PyTorch + torchvision + Streamlit
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 技术栈")
    st.sidebar.code("PyTorch | torchvision | Streamlit | Plotly", language=None)

    return module


def image_upload_section(label: str = "上传图片", key: str = "uploader"):
    """图片上传区域"""
    uploaded_file = st.file_uploader(
        label,
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key=key,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        return image
    return None


def sample_image_selector():
    """示例图片选择器"""
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_images')

    # 检查是否有示例图片
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if samples:
            selected = st.selectbox("或选择示例图片", ['--'] + samples)
            if selected != '--':
                return Image.open(os.path.join(sample_dir, selected)).convert('RGB')
    return None


def confidence_slider(key: str = "conf"):
    """置信度阈值滑块"""
    return st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        key=key,
        help="只显示置信度高于此阈值的结果"
    )


# ==================== 首页 ====================

def render_home():
    """渲染首页"""
    st.title("🎯 现代视觉任务：分割与检测")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔬 FCN 语义分割
        - 像素级分割
        - VGG16 Backbone
        - FCN-32s / FCN-8s
        - 输出：语义mask
        """)

    with col2:
        st.markdown("""
        ### 📦 R-CNN 目标检测
        - R-CNN → Fast R-CNN → Faster R-CNN
        - ResNet50+FPN Backbone
        - RPN区域提议
        - 输出：边界框 + 类别
        """)

    with col3:
        st.markdown("""
        ### 🎨 Mask R-CNN 实例分割
        - Faster R-CNN + Mask分支
        - RoIAlign精确定位
        - ResNet50+FPN Backbone
        - 输出：边界框 + 实例mask
        """)

    st.markdown("---")

    # 方法演进图
    st.subheader("📈 R-CNN 系列演进过程")
    st.markdown("""
    ```
    R-CNN (2014)                    Fast R-CNN (2015)               Faster R-CNN (2015)              Mask R-CNN (2017)
    +-----------------+           +-----------------+           +-----------------+           +-----------------+
    | Selective Search |           | Selective Search |           |      RPN        |           |      RPN        |
    |       ↓          |           |       ↓          |           | (Region Proposal|           |       ↓          |
    | Warped Regions   |           |  Shared CNN     |           |   Network)      |           |   RoIAlign      |
    |       ↓          |           |       ↓          |           |       ↓          |           |       ↓          |
    | CNN per Region   |           |  RoI Pooling    |           |  RoI Pooling    |           |  Classification |
    |       ↓          |           |       ↓          |           |       ↓          |           |  + BBox Reg     |
    | SVM Classifier   |           |   FC Layers     |           |   FC Layers     |           |  + Mask Branch  |
    +-----------------+           +-----------------+           +-----------------+           +-----------------+
    
    缺点：多阶段、慢            改进：共享特征、更快            改进：端到端、RPN              改进：像素级实例mask
    ```
    """)

    st.markdown("---")
    st.subheader("🚀 快速开始")
    st.markdown("""
    1. 在左侧边栏选择模块
    2. 上传图片或使用示例图片
    3. 调节参数（如置信度阈值）
    4. 点击"运行"按钮查看结果
    """)

    # 方法对比预览
    st.subheader("📊 方法对比一览")
    df = get_comparison_data()
    st.dataframe(df.set_index('方法'), width='stretch')


# ==================== FCN 语义分割 ====================

def render_fcn_module():
    """渲染FCN语义分割模块"""
    st.title("🔬 FCN 语义分割")
    st.markdown("基于全卷积网络（Fully Convolutional Network）的语义分割")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ 参数设置")

        model_type = st.selectbox(
            "模型类型",
            ['fcn32s', 'fcn8s'],
            index=0,
            help="FCN-32s: 32倍上采样; FCN-8s: 融合多层特征，8倍上采样"
        )

        use_pretrained = st.checkbox("使用预训练模型", value=True,
                                     help="使用ImageNet预训练的VGG16 backbone")

        use_torchvision_fcn = st.checkbox("使用torchvision预训练FCN", value=True,
                                          help="使用torchvision内置的预训练FCN（推荐，精度更高）")

        st.markdown("### 📤 输入")

        image = image_upload_section("上传图片进行语义分割", key="fcn_upload")
        st.markdown("*或使用内置摄像头拍照（如可用）*")
        camera_img = st.camera_input("拍照", key="fcn_camera")
        if camera_img is not None and image is None:
            image = Image.open(camera_img).convert('RGB')

        run_btn = st.button("🚀 运行语义分割", type="primary", width='stretch')

    with col2:
        st.markdown("### 📊 结果")

        if run_btn and image is not None:
            with st.spinner("正在进行语义分割..."):
                try:
                    with Timer() as timer:
                        if use_torchvision_fcn:
                            # 使用torchvision预训练FCN
                            import torch

                            model, weights = get_pretrained_segmentation_model()
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            model.to(device)
                            model.eval()

                            # 预处理
                            preprocess = weights.transforms()
                            img_tensor = preprocess(Image.fromarray(np.array(image))).unsqueeze(0).to(device)

                            with torch.no_grad():
                                output = model(img_tensor)['out'][0]
                                pred = output.argmax(0).cpu().numpy()

                            # 上采样回原图尺寸
                            orig_size = image.size  # (w, h)
                            pred_img = Image.fromarray(pred.astype(np.uint8))
                            pred_img = pred_img.resize(orig_size, Image.NEAREST)
                            pred_resized = np.array(pred_img)

                            # 获取类别名称
                            class_names = weights.meta['categories']
                            num_classes = len(class_names)
                            colormap = generate_colormap(num_classes)

                            # 生成叠加图
                            orig_np = np.array(image)
                            overlay = draw_segmentation_mask(orig_np, pred_resized, alpha=0.5, color_map=colormap)

                            model_info = {
                                'model_type': 'FCN (torchvision pretrained)',
                                'num_params': count_parameters(model),
                                'backbone': 'ResNet50',
                            }
                        else:
                            # 使用自定义FCN
                            segmenter = FCNSegmenter(
                                num_classes=21,
                                model_type=model_type,
                                pretrained=use_pretrained,
                            )
                            orig_np = np.array(image)
                            pred_resized, overlay = segmenter.predict_single(orig_np)
                            class_names = VOC_CLASSES[:21]
                            colormap = generate_colormap(21)
                            model_info = segmenter.get_model_info()

                    elapsed = timer.elapsed

                    # 显示结果
                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.markdown("**原图**")
                        st.image(image, width='stretch')

                    with result_col2:
                        st.markdown("**分割Mask**")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(pred_resized, cmap='tab20')
                        ax.set_title("Segmentation Mask")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)

                    with result_col3:
                        st.markdown("**叠加效果**")
                        st.image(overlay, width='stretch')

                    # 图例
                    st.markdown("**类别图例**")
                    unique_classes = np.unique(pred_resized)
                    legend_items = []
                    for cls in unique_classes:
                        if cls < len(class_names):
                            color = colormap[cls % len(colormap)]
                            color_hex = '#{:02x}{:02x}{:02x}'.format(
                                int(color[0]), int(color[1]), int(color[2]))
                            legend_items.append(
                                f"<span style='color:{color_hex}'>●</span> {class_names[cls]}")
                    st.markdown(" | ".join(legend_items[:20]), unsafe_allow_html=True)

                    # 指标
                    st.markdown("---")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("推理时间", f"{elapsed * 1000:.1f} ms")
                    with metrics_col2:
                        st.metric("模型参数", f"{model_info['num_params'] / 1e6:.1f}M")
                    with metrics_col3:
                        st.metric("类别数", str(len(class_names)))

                except Exception as e:
                    st.error(f"分割出错: {str(e)}")
                    st.info("请确保图片格式正确，或尝试其他图片。")

        elif run_btn and image is None:
            st.warning("请先上传一张图片！")

        elif image is not None and not run_btn:
            st.image(image, caption="待处理的图片", width='stretch')
            st.info("请在左侧点击「运行语义分割」按钮开始分析。")


# ==================== R-CNN 目标检测 ====================

def render_rcnn_module():
    """渲染R-CNN目标检测模块"""
    st.title("📦 R-CNN 系列目标检测")
    st.markdown("展示R-CNN → Fast R-CNN → Faster R-CNN的演进")

    # 演进说明
    with st.expander("📖 R-CNN 系列演进说明", expanded=False):
        tabs = st.tabs(["R-CNN", "Fast R-CNN", "Faster R-CNN"])

        with tabs[0]:
            st.markdown("""
            ### R-CNN (2014) — Regions with CNN Features
            **核心流程:**
            1. **Selective Search** 生成约2000个候选区域
            2. 每个区域 **Warp** 到固定大小(227×227)
            3. 每个区域独立通过 **CNN** 提取特征
            4. 用 **SVM** 对特征分类
            5. 用回归器精调边界框
            
            **缺点:** 训练多阶段、慢（每张图约47秒）、存储开销大
            """)

        with tabs[1]:
            st.markdown("""
            ### Fast R-CNN (2015) — 共享卷积特征
            **核心改进:**
            1. 整张图只跑 **一次CNN**，共享特征图
            2. 引入 **RoI Pooling** 提取固定大小特征
            3. 分类和回归使用 **全连接层**，联合训练
            
            **优势:** 训练速度提升9倍，推理速度提升213倍
            """)

        with tabs[2]:
            st.markdown("""
            ### Faster R-CNN (2015) — 端到端区域提议
            **核心创新:**
            1. 引入 **RPN (Region Proposal Network)** 替代Selective Search
            2. RPN与检测网络 **共享卷积特征**
            3. 实现真正的 **端到端训练**
            
            **优势:** 几乎实时的检测速度，精度高
            """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ 参数设置")

        detection_method = st.radio(
            "检测方法",
            ['Faster R-CNN (推荐)', 'R-CNN (简化演示)'],
            index=0,
            help="Faster R-CNN使用torchvision预训练模型；R-CNN为简化演示版"
        )

        score_threshold = confidence_slider(key="rcnn_conf")

        st.markdown("### 📤 输入")
        image = image_upload_section("上传图片进行目标检测", key="rcnn_upload")
        camera_img = st.camera_input("拍照", key="rcnn_camera")
        if camera_img is not None and image is None:
            image = Image.open(camera_img).convert('RGB')

        run_btn = st.button("🚀 运行目标检测", type="primary", width='stretch')

    with col2:
        st.markdown("### 📊 检测结果")

        if run_btn and image is not None:
            with st.spinner("正在进行目标检测..."):
                try:
                    if detection_method == 'Faster R-CNN (推荐)':
                        detector = load_rcnn_models()

                        with Timer() as timer:
                            result_image, detections = detector.predict_pil(image, score_threshold)

                        model_info = detector.get_model_info()
                    else:
                        detector = SimpleRCNNDetector()

                        with Timer() as timer:
                            result_image, detections = detector.predict(np.array(image), score_threshold)

                        model_info = detector.get_model_info()

                    elapsed = timer.elapsed

                    # 显示结果
                    st.image(result_image,
                             caption=f"检测结果 ({detections['num_detections']} 个目标)",
                             width='stretch')

                    # 检测详情
                    if detections['num_detections'] > 0:
                        st.markdown("### 📋 检测详情")

                        # 构建表格数据
                        table_data = []
                        for i in range(min(detections['num_detections'], 20)):
                            lbl = int(detections['labels'][i])
                            class_name = COCO_CLASSES[lbl] if lbl < len(COCO_CLASSES) else f"class_{lbl}"
                            box = detections['boxes'][i]
                            table_data.append({
                                '#': i + 1,
                                '类别': class_name,
                                '置信度': f"{detections['scores'][i]:.3f}",
                                '边界框': f"({int(box[0])}, {int(box[1])}) - ({int(box[2])}, {int(box[3])})",
                            })

                        st.dataframe(table_data, width='stretch')

                    # 指标
                    st.markdown("---")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("推理时间", f"{elapsed * 1000:.1f} ms")
                    with metrics_col2:
                        st.metric("检测目标数", str(detections['num_detections']))
                    with metrics_col3:
                        param_str = f"{model_info['num_params'] / 1e6:.1f}M" if model_info['num_params'] > 0 else 'N/A'
                        st.metric("模型参数", param_str)

                except Exception as e:
                    st.error(f"检测出错: {str(e)}")

        elif run_btn and image is None:
            st.warning("请先上传一张图片！")

        elif image is not None and not run_btn:
            st.image(image, caption="待检测的图片", width='stretch')
            st.info("请在左侧点击「运行目标检测」按钮开始分析。")


# ==================== Mask R-CNN 实例分割 ====================

def render_mask_rcnn_module():
    """渲染Mask R-CNN实例分割模块"""
    st.title("🎨 Mask R-CNN 实例分割")
    st.markdown("像素级实例分割 —— 同时输出边界框和实例mask")

    with st.expander("📖 Mask R-CNN 原理简介", expanded=False):
        st.markdown("""
        ### Mask R-CNN (2017)
        
        **架构设计:**
        ```
        Input Image → Backbone (ResNet50+FPN) → RPN → RoIAlign →┬→ Classification
                                                                  ├→ BBox Regression
                                                                  └→ Mask Branch (FCN)
        ```
        
        **核心创新:**
        1. **Mask分支**: 与分类和回归并行的小型FCN
        2. **RoIAlign**: 双线性插值替代量化取整，解决RoIPool的misalignment问题
        3. **解耦预测**: Mask预测与类别预测独立，每个类别输出独立的mask
        
        **输出形式:** 每个检测到的实例都有其专属的mask、边界框和类别标签
        """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ 参数设置")

        score_threshold = confidence_slider(key="mask_conf")

        st.markdown("### 📤 输入")
        image = image_upload_section("上传图片进行实例分割", key="mask_upload")
        camera_img = st.camera_input("拍照", key="mask_camera")
        if camera_img is not None and image is None:
            image = Image.open(camera_img).convert('RGB')

        run_btn = st.button("🚀 运行实例分割", type="primary", width='stretch')

    with col2:
        st.markdown("### 🎯 分割结果")

        if run_btn and image is not None:
            with st.spinner("正在进行实例分割..."):
                try:
                    segmenter = load_mask_rcnn_model()

                    with Timer() as timer:
                        result_image, detections = segmenter.predict_pil(image, score_threshold)

                    elapsed = timer.elapsed
                    model_info = segmenter.get_model_info()

                    # 显示结果
                    st.image(result_image,
                             caption=f"实例分割结果 ({detections['num_instances']} 个实例)",
                             width='stretch')

                    # 实例详情
                    if detections['num_instances'] > 0:
                        st.markdown("### 📋 实例详情")

                        table_data = []
                        for i in range(min(detections['num_instances'], 20)):
                            cls_idx = int(detections['labels'][i])
                            class_name = COCO_CLASSES[cls_idx] if cls_idx < len(COCO_CLASSES) else f"class_{cls_idx}"
                            box = detections['boxes'][i]
                            area = (box[2] - box[0]) * (box[3] - box[1])

                            table_data.append({
                                '#': i + 1,
                                '类别': class_name,
                                '置信度': f"{detections['scores'][i]:.3f}",
                                '面积(pixels)': f"{int(area)}",
                            })

                        st.dataframe(table_data, width='stretch')

                    # 指标
                    st.markdown("---")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("推理时间", f"{elapsed * 1000:.1f} ms")
                    with metrics_col2:
                        st.metric("实例数", str(detections['num_instances']))
                    with metrics_col3:
                        st.metric("模型参数", f"{model_info['num_params'] / 1e6:.1f}M")

                except Exception as e:
                    st.error(f"实例分割出错: {str(e)}")

        elif run_btn and image is None:
            st.warning("请先上传一张图片！")

        elif image is not None and not run_btn:
            st.image(image, caption="待分割的图片", width='stretch')
            st.info("请在左侧点击「运行实例分割」按钮开始分析。")


# ==================== 方法性能对比 ====================

def render_comparison_module():
    """渲染性能对比模块"""
    st.title("📊 方法性能对比")
    st.markdown("FCN vs Faster R-CNN vs Mask R-CNN 全方位对比分析")

    # 数据表格
    st.subheader("📋 对比数据表")
    df = get_comparison_data()
    st.dataframe(df.set_index('方法'), width='stretch')

    # 图表
    st.subheader("📈 可视化对比")
    charts = create_comparison_charts()

    tab1, tab2, tab3, tab4 = st.tabs(["精度对比", "推理时间", "参数量", "综合雷达图"])

    with tab1:
        st.plotly_chart(charts['accuracy'], width='stretch')

    with tab2:
        st.plotly_chart(charts['inference_time'], width='stretch')

    with tab3:
        st.plotly_chart(charts['params'], width='stretch')

    with tab4:
        st.plotly_chart(charts['radar'], width='stretch')

    # 总结
    st.markdown("---")
    st.markdown(get_comparison_summary())

    # 实时基准测试（可选）
    st.markdown("---")
    st.subheader("⏱ 实时推理基准测试")
    st.markdown("使用您上传的图片进行实际推理时间对比")

    benchmark_image = image_upload_section("上传测试图片", key="benchmark_upload")

    if benchmark_image is not None and st.button("🏃 运行基准测试", type="primary"):
        with st.spinner("正在进行基准测试（这将需要一些时间）..."):
            img_np = np.array(benchmark_image)

            results = []

            # 测试Faster R-CNN
            st.info("测试 Faster R-CNN...")
            detector = load_rcnn_models()
            times_rcnn = []
            for _ in range(3):
                start = time.time()
                detector.predict(img_np)
                times_rcnn.append((time.time() - start) * 1000)
            results.append({
                '方法': 'Faster R-CNN',
                '平均推理时间 (ms)': f"{np.mean(times_rcnn):.1f}",
                '最快/最慢 (ms)': f"{np.min(times_rcnn):.1f} / {np.max(times_rcnn):.1f}",
            })

            # 测试Mask R-CNN
            st.info("测试 Mask R-CNN...")
            segmenter = load_mask_rcnn_model()
            times_mask = []
            for _ in range(3):
                start = time.time()
                segmenter.predict(img_np)
                times_mask.append((time.time() - start) * 1000)
            results.append({
                '方法': 'Mask R-CNN',
                '平均推理时间 (ms)': f"{np.mean(times_mask):.1f}",
                '最快/最慢 (ms)': f"{np.min(times_mask):.1f} / {np.max(times_mask):.1f}",
            })

            # 测试torchvision FCN
            st.info("测试 FCN 语义分割...")
            import torch
            model_ft, weights_ft = get_pretrained_segmentation_model()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_ft.to(device)
            model_ft.eval()
            preprocess = weights_ft.transforms()
            img_tensor = preprocess(Image.fromarray(img_np)).unsqueeze(0).to(device)

            times_fcn = []
            with torch.no_grad():
                for _ in range(3):
                    start = time.time()
                    model_ft(img_tensor)['out']
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times_fcn.append((time.time() - start) * 1000)

            results.append({
                '方法': 'FCN (语义分割)',
                '平均推理时间 (ms)': f"{np.mean(times_fcn):.1f}",
                '最快/最慢 (ms)': f"{np.min(times_fcn):.1f} / {np.max(times_fcn):.1f}",
            })

            # 显示结果
            import pandas as pd
            st.dataframe(pd.DataFrame(results).set_index('方法'), width='stretch')

            # 对比柱状图
            fig = go.Figure(data=[
                go.Bar(
                    x=[r['方法'] for r in results],
                    y=[float(r['平均推理时间 (ms)']) for r in results],
                    text=[r['平均推理时间 (ms)'] for r in results],
                    textposition='auto',
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                )
            ])
            fig.update_layout(
                title='实测推理时间对比',
                yaxis_title='推理时间 (ms)',
                template='plotly_white',
                height=400,
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(f"测试设备: {device.type.upper()}")


# ==================== 主入口 ====================

def main():
    """主应用入口"""
    module = render_sidebar()

    if module == "🏠 首页":
        render_home()
    elif module == "🔬 FCN 语义分割":
        render_fcn_module()
    elif module == "📦 R-CNN 目标检测":
        render_rcnn_module()
    elif module == "🎨 Mask R-CNN 实例分割":
        render_mask_rcnn_module()
    elif module == "📊 方法性能对比":
        render_comparison_module()


if __name__ == "__main__":
    main()
