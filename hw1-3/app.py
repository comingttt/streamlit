"""
计算机视觉课程设计 - 综合演示系统 (Streamlit版)
整合作业1(颜色空间+插值)、作业2(空间滤波+梯度+频率域)、作业3(边缘检测+特征点+匹配+全景拼接)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io

# 设置页面
st.set_page_config(page_title="计算机视觉课程设计", layout="wide", page_icon="🖼️")

# 导入各作业模块
from hw1 import a1 as hw1
from hw2 import a2 as hw2
from hw3 import a3 as hw3


def img_to_streamlit(image_bgr, channels="BGR", width=None):
    """将OpenCV BGR图像转换为Streamlit可显示的格式"""
    if image_bgr is None:
        return None
    if channels == "BGR":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif channels == "GRAY":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif channels == "HSV":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_HSV2RGB)
    elif channels == "Lab":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_Lab2RGB)
    else:
        rgb = image_bgr
    return rgb


def load_image_from_uploader(uploaded_file):
    """从Streamlit上传的文件加载图像"""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None


# ==================== 侧边栏导航 ====================
st.sidebar.title("📌 计算机视觉课程设计")
homework = st.sidebar.radio(
    "选择作业",
    ["作业1: 颜色空间与插值", "作业2: 空间滤波与频率域", "作业3: 边缘检测与特征匹配"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**虚拟环境:** imghw")
st.sidebar.markdown("**作者:** 计算机视觉课程设计")


# ==================== 作业1: 颜色空间与图像插值 ====================
if homework == "作业1: 颜色空间与插值":
    st.title("🎨 作业1: 颜色空间转换与图像插值")
    
    tab1, tab2 = st.tabs(["🌈 颜色空间转换", "🔍 图像插值算法"])
    
    # ---------- Tab1: 颜色空间 ----------
    with tab1:
        st.markdown("### 颜色空间通道可视化")
        st.markdown("上传图像，观察不同颜色空间中各通道的分布情况。")
        
        uploaded = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png", "bmp"], key="hw1_color")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            color_space = st.selectbox(
                "选择颜色空间",
                ["RGB", "HSV", "Lab", "YCrCb"],
                key="cs_select"
            )
        
        if uploaded:
            image = load_image_from_uploader(uploaded)
            if image is not None:
                with col2:
                    st.markdown("**原始图像 (BGR)**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                
                # 显示各通道
                channels, channel_names = hw1.get_color_space_channels(image, color_space)
                
                st.markdown(f"**{color_space} 颜色空间各通道**")
                ch_cols = st.columns(3)
                for i, (ch, name) in enumerate(zip(channels, channel_names)):
                    with ch_cols[i]:
                        ch_display = cv2.merge([ch, ch, ch])
                        st.image(img_to_streamlit(ch_display), caption=name, use_container_width=True)
                
                # 显示转换后的彩色图像
                converted = hw1.convert_color_space(image, color_space)
                st.markdown(f"**{color_space} 彩色图像**")
                st.image(converted, use_container_width=True)
        else:
            st.info("👆 请上传图像以查看颜色空间转换效果")
    
    # ---------- Tab2: 图像插值 ----------
    with tab2:
        st.markdown("### 图像插值算法演示")
        st.markdown("对比最近邻插值与双线性插值在不同操作下的效果。")
        
        uploaded2 = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png", "bmp"], key="hw1_interp")
        
        if uploaded2:
            image = load_image_from_uploader(uploaded2)
            if image is not None:
                h, w = image.shape[:2]
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    operation = st.selectbox(
                        "选择操作",
                        ["放大", "缩小", "旋转", "拉伸"],
                        key="interp_op"
                    )
                
                with col2:
                    method = st.selectbox(
                        "选择插值方法",
                        ["最近邻 (Nearest)", "双线性 (Bilinear)"],
                        key="interp_method"
                    )
                    method_key = "nearest" if "最近邻" in method else "bilinear"
                
                with col3:
                    params = {}
                    if operation in ["放大", "缩小"]:
                        scale = st.slider("缩放比例", 0.1, 5.0, 2.0 if operation == "放大" else 0.5, 0.1)
                        params["scale"] = scale
                    elif operation == "旋转":
                        angle = st.slider("旋转角度", 0, 360, 45)
                        params["angle"] = angle
                    elif operation == "拉伸":
                        new_w = st.slider("目标宽度", 50, 2000, int(w * 1.5))
                        new_h = st.slider("目标高度", 50, 2000, int(h * 1.5))
                        params["width"] = new_w
                        params["height"] = new_h
                
                # 显示原始图像
                st.markdown("**原始图像**")
                st.image(img_to_streamlit(image), caption=f"尺寸: {w}x{h}", use_container_width=True)
                
                # 应用插值
                result_manual = hw1.apply_interpolation(image, operation, params, method_key)
                result_opencv = hw1.apply_interpolation_opencv(image, operation, params, method_key)
                
                if result_manual is not None:
                    st.markdown(f"**{operation} ({method})**")
                    cols = st.columns(2)
                    with cols[0]:
                        st.image(img_to_streamlit(result_manual), 
                               caption=f"手动实现 - 尺寸: {result_manual.shape[1]}x{result_manual.shape[0]}",
                               use_container_width=True)
                    with cols[1]:
                        st.image(img_to_streamlit(result_opencv),
                               caption=f"OpenCV实现 - 尺寸: {result_opencv.shape[1]}x{result_opencv.shape[0]}",
                               use_container_width=True)
        else:
            st.info("👆 请上传图像以查看插值效果")


# ==================== 作业2: 空间滤波与频率域 ====================
elif homework == "作业2: 空间滤波与频率域":
    st.title("🔬 作业2: 空间图像滤波与频率域分析")
    
    func = st.sidebar.radio(
        "选择功能",
        ["空间滤波器演示", "图像梯度演示", "频率域滤波演示"],
        key="hw2_func"
    )
    
    uploaded = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png", "bmp"], key="hw2_img")
    
    if uploaded:
        image = load_image_from_uploader(uploaded)
        if image is not None:
            
            # ----- 空间滤波器 -----
            if func == "空间滤波器演示":
                st.markdown("### 空间滤波器对比")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    filter_type = st.selectbox(
                        "滤波器类型",
                        ["Box滤波", "Gaussian滤波", "Median滤波", "Sobel滤波"],
                        key="filter_type"
                    )
                    kernel_size = st.slider("核大小", 3, 31, 5, step=2, key="filter_kernel")
                
                with col2:
                    st.markdown("**原始图像**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                
                result = hw2.apply_spatial_filter(image, filter_type, kernel_size)
                
                comp_cols = st.columns(2)
                with comp_cols[0]:
                    st.markdown("**原始图像**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                with comp_cols[1]:
                    st.markdown(f"**{filter_type} (核大小={kernel_size})**")
                    st.image(img_to_streamlit(result), use_container_width=True)
            
            # ----- 图像梯度 -----
            elif func == "图像梯度演示":
                st.markdown("### 图像梯度算法演示")
                st.markdown("选择图像局部区域，计算该区域的梯度方向和幅度（HSV颜色编码：色相=方向，亮度=幅度）。")
                
                h_img, w_img = image.shape[:2]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**区域坐标设置**")
                    x1 = st.number_input("x1", 0, w_img-1, max(0, w_img//4), key="grad_x1")
                    y1 = st.number_input("y1", 0, h_img-1, max(0, h_img//4), key="grad_y1")
                    x2 = st.number_input("x2", 0, w_img-1, min(w_img-1, w_img*3//4), key="grad_x2")
                    y2 = st.number_input("y2", 0, h_img-1, min(h_img-1, h_img*3//4), key="grad_y2")
                    
                    if st.button("计算梯度", key="calc_grad"):
                        marked, gradient = hw2.compute_gradient(image, x1, y1, x2, y2)
                        st.session_state["marked_img"] = marked
                        st.session_state["gradient_img"] = gradient
                
                with col2:
                    st.markdown("**原始图像**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                
                if "marked_img" in st.session_state:
                    grad_cols = st.columns(2)
                    with grad_cols[0]:
                        st.markdown("**标记区域**")
                        st.image(img_to_streamlit(st.session_state["marked_img"]), use_container_width=True)
                    with grad_cols[1]:
                        st.markdown("**梯度图像 (HSV编码)**")
                        st.image(img_to_streamlit(st.session_state["gradient_img"]), use_container_width=True)
            
            # ----- 频率域滤波 -----
            elif func == "频率域滤波演示":
                st.markdown("### 频率域图像滤波分析")
                st.markdown("基于傅里叶变换分析图像频谱，比较旋转/平移/缩放对频谱的影响。")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    transform = st.selectbox(
                        "变换类型",
                        ["原始", "旋转", "平移", "缩放"],
                        key="freq_transform"
                    )
                    
                    params = {}
                    if transform == "旋转":
                        params["angle"] = st.slider("旋转角度", 0, 360, 30, key="freq_angle")
                    elif transform == "平移":
                        params["shift"] = st.slider("平移距离", 0, 200, 50, key="freq_shift")
                    elif transform == "缩放":
                        params["scale"] = st.slider("缩放比例", 0.1, 3.0, 0.5, 0.1, key="freq_scale")
                
                with col2:
                    st.markdown("**原始图像**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                
                # 计算原始频谱
                orig_spectrum = hw2.compute_spectrum(image)
                
                # 应用变换
                transformed = hw2.apply_frequency_transform(image, transform, **params)
                trans_spectrum = hw2.compute_spectrum(transformed)
                
                disp_cols = st.columns(2)
                with disp_cols[0]:
                    st.markdown("**原始频谱图**")
                    st.image(orig_spectrum, clamp=True, caption="傅里叶幅度谱", use_container_width=True)
                    
                    st.markdown("**原始图像**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                
                with disp_cols[1]:
                    st.markdown(f"**{transform}后频谱图**")
                    st.image(trans_spectrum, clamp=True, caption=f"{transform}后幅度谱", use_container_width=True)
                    
                    st.markdown(f"**{transform}后图像**")
                    st.image(img_to_streamlit(transformed), use_container_width=True)
    
    else:
        st.info("👆 请上传图像以开始演示")


# ==================== 作业3: 边缘检测与特征匹配 ====================
else:
    st.title("🎯 作业3: 边缘检测与特征匹配")
    
    func = st.sidebar.radio(
        "选择功能",
        ["边缘检测 (Canny)", "特征点检测", "图像匹配", "全景拼接"],
        key="hw3_func"
    )
    
    # ----- 边缘检测 -----
    if func == "边缘检测 (Canny)":
        st.markdown("### Canny边缘检测")
        st.markdown("对比非最大值抑制前后的边缘检测效果。")
        
        uploaded = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png", "bmp"], key="hw3_edge")
        
        if uploaded:
            image = load_image_from_uploader(uploaded)
            if image is not None:
                col1, col2 = st.columns([1, 3])
                with col1:
                    min_thresh = st.slider("最小阈值", 0, 255, 50, key="canny_min")
                    max_thresh = st.slider("最大阈值", 0, 255, 150, key="canny_max")
                
                with col2:
                    st.markdown("**原始图像**")
                    st.image(img_to_streamlit(image), use_container_width=True)
                
                gradient, before_nms, edges = hw3.apply_canny_edge_detection(image, min_thresh, max_thresh)
                
                edge_cols = st.columns(3)
                with edge_cols[0]:
                    st.markdown("**梯度幅值**")
                    st.image(gradient, clamp=True, use_container_width=True)
                with edge_cols[1]:
                    st.markdown("**非最大值抑制前**")
                    st.image(before_nms, clamp=True, use_container_width=True)
                with edge_cols[2]:
                    st.markdown("**Canny边缘 (含NMS)**")
                    st.image(edges, clamp=True, use_container_width=True)
    
    # ----- 特征点检测 -----
    elif func == "特征点检测":
        st.markdown("### 特征点检测")
        st.markdown("Harris角点与SIFT特征点检测对比（圆形区域表示特征点）。")
        
        uploaded = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png", "bmp"], key="hw3_feat")
        
        if uploaded:
            image = load_image_from_uploader(uploaded)
            if image is not None:
                st.markdown("**原始图像**")
                st.image(img_to_streamlit(image), use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Harris角点检测")
                    block_size = st.slider("Harris块大小", 1, 10, 2, key="harris_block")
                    k_val = st.slider("Harris k值", 0.01, 0.1, 0.04, 0.01, key="harris_k")
                    
                    if st.button("运行Harris检测", key="run_harris"):
                        harris_result = hw3.detect_harris_corners(image, block_size, k_val)
                        st.session_state["harris_result"] = harris_result
                    
                    if "harris_result" in st.session_state:
                        st.image(img_to_streamlit(st.session_state["harris_result"]), 
                                caption="Harris角点 (红色=角点, 绿色圆圈=标记)", use_container_width=True)
                
                with col2:
                    st.markdown("#### SIFT特征点检测")
                    nfeatures = st.slider("SIFT特征点数量", 10, 500, 100, key="sift_n")
                    
                    if st.button("运行SIFT检测", key="run_sift"):
                        sift_result, kp = hw3.detect_sift_features(image, nfeatures)
                        st.session_state["sift_result"] = sift_result
                        st.session_state["sift_kp_count"] = len(kp)
                    
                    if "sift_result" in st.session_state:
                        st.image(img_to_streamlit(st.session_state["sift_result"]),
                                caption=f"SIFT特征点 (绿色圆圈=特征点, 共{st.session_state.get('sift_kp_count', 0)}个)",
                                use_container_width=True)
    
    # ----- 图像匹配 -----
    elif func == "图像匹配":
        st.markdown("### 图像匹配流程可视化")
        st.markdown("展示特征点检测、描述、初始匹配、RANSAC筛选、变换对齐的完整流程。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded1 = st.file_uploader("选择第一幅图像", type=["jpg", "jpeg", "png", "bmp"], key="match_img1")
        with col2:
            uploaded2 = st.file_uploader("选择第二幅图像", type=["jpg", "jpeg", "png", "bmp"], key="match_img2")
        
        if uploaded1 and uploaded2:
            img1 = load_image_from_uploader(uploaded1)
            img2 = load_image_from_uploader(uploaded2)
            
            if img1 is not None and img2 is not None:
                disp_cols = st.columns(2)
                with disp_cols[0]:
                    st.markdown("**第一幅图像**")
                    st.image(img_to_streamlit(img1), use_container_width=True)
                with disp_cols[1]:
                    st.markdown("**第二幅图像**")
                    st.image(img_to_streamlit(img2), use_container_width=True)
                
                method = st.selectbox("匹配方法", ["SIFT", "ORB"], key="match_method")
                nfeatures = st.slider("SIFT特征点数量", 10, 500, 100, key="match_nfeatures")
                
                if st.button("执行匹配", key="run_match"):
                    with st.spinner("正在匹配..."):
                        if method == "SIFT":
                            init_result, ransac_result, n_init, n_inlier = hw3.match_images_sift(
                                img1, img2, nfeatures
                            )
                            st.session_state["init_match"] = init_result
                            st.session_state["ransac_match"] = ransac_result
                            st.session_state["n_init"] = n_init
                            st.session_state["n_inlier"] = n_inlier
                        else:
                            result, n_matches = hw3.match_images_orb(img1, img2)
                            st.session_state["orb_match"] = result
                            st.session_state["orb_n"] = n_matches
                
                if method == "SIFT":
                    if "init_match" in st.session_state:
                        match_cols = st.columns(2)
                        with match_cols[0]:
                            st.markdown(f"**初始匹配 (共{st.session_state['n_init']}对)**")
                            st.image(img_to_streamlit(st.session_state["init_match"]), use_container_width=True)
                        with match_cols[1]:
                            st.markdown(f"**RANSAC后 (内点{st.session_state['n_inlier']}对)**")
                            st.image(img_to_streamlit(st.session_state["ransac_match"]), use_container_width=True)
                else:
                    if "orb_match" in st.session_state:
                        st.markdown(f"**ORB匹配 (共{st.session_state['orb_n']}对)**")
                        st.image(img_to_streamlit(st.session_state["orb_match"]), use_container_width=True)
    
    # ----- 全景拼接 -----
    elif func == "全景拼接":
        st.markdown("### 多图像全景拼接")
        st.markdown("输入同一场景的多幅有重叠区域的图像，输出全景图。")
        
        uploaded_files = st.file_uploader(
            "选择多幅图像 (按住Ctrl选择多个)", 
            type=["jpg", "jpeg", "png", "bmp"], 
            accept_multiple_files=True,
            key="pano_files"
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            images = []
            for f in uploaded_files:
                img = load_image_from_uploader(f)
                if img is not None:
                    images.append(img)
            
            st.markdown(f"**已加载 {len(images)} 幅图像**")
            
            # 显示所有输入图像
            pano_cols = st.columns(min(len(images), 5))
            for i, img in enumerate(images):
                with pano_cols[i % 5]:
                    st.image(img_to_streamlit(img), caption=f"图像{i+1}", use_container_width=True)
            
            if st.button("执行全景拼接", key="run_pano"):
                with st.spinner("正在拼接中..."):
                    pano, success = hw3.stitch_panorama(images)
                
                if success and pano is not None:
                    st.markdown("#### 全景拼接结果")
                    st.image(img_to_streamlit(pano), use_container_width=True)
                    st.success("✅ 全景拼接成功!")
                else:
                    st.error("❌ 拼接失败。请确保图像有足够重叠区域，且特征点充足。")
        else:
            st.info("👆 请上传至少2幅有重叠区域的图像")


# ==================== 页脚 ====================
st.sidebar.markdown("---")
st.sidebar.markdown("**说明:** 本系统整合了3个作业的全部功能")
st.sidebar.markdown("- 作业1: 颜色空间转换 + 图像插值")
st.sidebar.markdown("- 作业2: 空间滤波 + 梯度 + 频率域")
st.sidebar.markdown("- 作业3: 边缘检测 + 特征点 + 匹配 + 全景拼接")
