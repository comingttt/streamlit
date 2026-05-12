import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import threading

# tkinter 仅在独立运行时导入（Streamlit Cloud 无 tkinter）
try:
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import ImageTk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

# ==================== 核心处理函数（供Streamlit调用）====================

def apply_canny_edge_detection(image, min_thresh=50, max_thresh=150):
    """Canny边缘检测，返回梯度图、非最大值抑制前、Canny结果"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 梯度幅值和方向
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    gradient = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 非最大值抑制前（双阈值）
    _, thresh_low = cv2.threshold(magnitude, min_thresh, 255, cv2.THRESH_BINARY)
    _, thresh_high = cv2.threshold(magnitude, max_thresh, 255, cv2.THRESH_BINARY)
    before_nms = cv2.bitwise_or(thresh_low.astype(np.uint8), thresh_high.astype(np.uint8))
    
    # 完整Canny（含非最大值抑制）
    edges = cv2.Canny(blurred, min_thresh, max_thresh)
    
    return gradient, before_nms, edges


def detect_harris_corners(image, block_size=2, k=0.04):
    """Harris角点检测"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, blockSize=block_size, ksize=3, k=k)
    dst = cv2.dilate(dst, None)
    
    result = image.copy()
    corners = np.where(dst > 0.01 * dst.max())
    for y, x in zip(corners[0], corners[1]):
        cv2.circle(result, (x, y), 5, (0, 255, 0), 2)
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    return result


def detect_sift_features(image, nfeatures=100):
    """SIFT特征点检测"""
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=5)
    keypoints, _ = sift.detectAndCompute(image, None)
    
    result = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size / 2)
        cv2.circle(result, (x, y), size, (0, 255, 0), 2)
    
    return result, keypoints


def match_images_sift(img1, img2, nfeatures=100):
    """SIFT图像匹配（含RANSAC），返回匹配结果图"""
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=5)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    result_img = _draw_matches(img1, kp1, img2, kp2, good_matches)
    
    # RANSAC
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            mask_ravel = mask.ravel()
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_ravel[i]]
            
            # 绘制边界框
            h, w = img1.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2_box = img2.copy()
            img2_box = cv2.polylines(img2_box, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            ransac_img = _draw_matches(img1, kp1, img2_box, kp2, inlier_matches)
            return result_img, ransac_img, len(good_matches), len(inlier_matches)
    
    return result_img, result_img, len(good_matches), 0


def match_images_orb(img1, img2):
    """ORB图像匹配"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    result = _draw_matches(img1, kp1, img2, kp2, matches)
    return result, len(matches)


def _draw_matches(img1, kp1, img2, kp2, matches):
    """绘制匹配连线图"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[0:h1, 0:w1] = img1
    result[0:h2, w1:w1+w2] = img2
    
    for i, m in enumerate(matches):
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))
        color = (int(255 * (i % 10) / 10.0),
                 int(255 * ((i * 7) % 10) / 10.0),
                 int(255 * ((i * 3) % 10) / 10.0))
        cv2.line(result, pt1, pt2, color, 2)
        cv2.circle(result, pt1, 5, (0, 255, 255), -1)
        cv2.circle(result, pt2, 5, (0, 255, 255), -1)
    
    return result


def stitch_panorama(images, blend_method="线性Blending"):
    """全景拼接"""
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK and pano is not None and pano.size > 0:
        return pano, True
    return None, False


class ComputerVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("计算机视觉课程设计 - 图像处理演示")
        self.root.geometry("1400x900")

        # 初始化变量
        self.image = None
        self.original_image = None
        self.current_image = None
        self.processed_images = {}
        self.images_list = []  # 用于全景拼接的多幅图像
        self.second_image = None  # 用于图像匹配的第二幅图像

        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 创建右侧显示区域
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # 创建加载图像按钮
        ttk.Button(self.control_frame, text="加载图像", command=self.load_image).pack(pady=5, fill=tk.X)

        # 创建加载多幅图像按钮（用于全景拼接）
        ttk.Button(self.control_frame, text="加载多幅图像", command=self.load_multiple_images).pack(pady=5, fill=tk.X)

        # 创建Notebook来组织功能设置
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # 绑定tab切换事件
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # 创建边缘检测tab
        self.edge_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.edge_tab, text="边缘检测")

        # Canny参数
        ttk.Label(self.edge_tab, text="最小阈值").pack(pady=5)
        self.canny_min_var = tk.IntVar(value=50)
        ttk.Scale(self.edge_tab, from_=0, to=255, variable=self.canny_min_var, orient=tk.HORIZONTAL, command=self.apply_edge_detection).pack(pady=5, fill=tk.X)

        ttk.Label(self.edge_tab, text="最大阈值").pack(pady=5)
        self.canny_max_var = tk.IntVar(value=150)
        ttk.Scale(self.edge_tab, from_=0, to=255, variable=self.canny_max_var, orient=tk.HORIZONTAL, command=self.apply_edge_detection).pack(pady=5, fill=tk.X)

        # 创建特征点检测tab
        self.feature_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.feature_tab, text="特征点检测")

        # 特征点类型选择
        ttk.Label(self.feature_tab, text="特征点类型").pack(pady=5)
        self.feature_var = tk.StringVar()
        self.feature_combobox = ttk.Combobox(self.feature_tab, textvariable=self.feature_var, values=["Harris角点", "SIFT特征点"])
        self.feature_combobox.current(0)
        self.feature_combobox.pack(pady=5, fill=tk.X)
        self.feature_combobox.bind("<<ComboboxSelected>>", self.apply_feature_detection)

        # Harris参数
        ttk.Label(self.feature_tab, text="Harris块大小").pack(pady=5)
        self.harris_block_var = tk.IntVar(value=2)
        ttk.Scale(self.feature_tab, from_=1, to=10, variable=self.harris_block_var, orient=tk.HORIZONTAL, command=self.apply_feature_detection).pack(pady=5, fill=tk.X)

        ttk.Label(self.feature_tab, text="Harris k值").pack(pady=5)
        self.harris_k_var = tk.DoubleVar(value=0.04)
        ttk.Scale(self.feature_tab, from_=0.01, to=0.1, variable=self.harris_k_var, orient=tk.HORIZONTAL, command=self.apply_feature_detection).pack(pady=5, fill=tk.X)

        # SIFT参数
        ttk.Label(self.feature_tab, text="SIFT特征点数量").pack(pady=5)
        self.sift_nfeatures_var = tk.IntVar(value=100)
        ttk.Scale(self.feature_tab, from_=10, to=500, variable=self.sift_nfeatures_var, orient=tk.HORIZONTAL, command=self.apply_feature_detection).pack(pady=5, fill=tk.X)

        # 创建图像匹配tab
        self.match_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.match_tab, text="图像匹配")

        # 加载第二幅图像
        ttk.Button(self.match_tab, text="加载第二幅图像", command=self.load_second_image).pack(pady=5, fill=tk.X)

        # 匹配参数
        ttk.Label(self.match_tab, text="匹配方法").pack(pady=5)
        self.match_method_var = tk.StringVar()
        self.match_method_combobox = ttk.Combobox(self.match_tab, textvariable=self.match_method_var, values=["SIFT", "ORB"])
        self.match_method_combobox.current(0)
        self.match_method_combobox.pack(pady=5, fill=tk.X)

        ttk.Button(self.match_tab, text="执行匹配", command=self.apply_image_matching).pack(pady=5, fill=tk.X)

        # 创建全景拼接tab
        self.panorama_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.panorama_tab, text="全景拼接")

        # Blending方法选择
        ttk.Label(self.panorama_tab, text="Blending方法").pack(pady=5)
        self.blend_var = tk.StringVar()
        self.blend_combobox = ttk.Combobox(self.panorama_tab, textvariable=self.blend_var, values=["线性Blending", "多频段Blending"])
        self.blend_combobox.current(0)
        self.blend_combobox.pack(pady=5, fill=tk.X)

        ttk.Button(self.panorama_tab, text="执行拼接", command=self.apply_panorama_stitching).pack(pady=5, fill=tk.X)

        # 创建显示区域框架
        self.edge_display_frame = ttk.Frame(self.display_frame)
        self.feature_display_frame = ttk.Frame(self.display_frame)
        self.match_display_frame = ttk.Frame(self.display_frame)
        self.panorama_display_frame = ttk.Frame(self.display_frame)

        # 边缘检测显示区域
        self.edge_show_frame = ttk.LabelFrame(self.edge_display_frame, text="边缘检测结果", padding="10")
        self.edge_show_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 梯度图像
        self.edge_gradient_label = ttk.Label(self.edge_show_frame, text="梯度图像")
        self.edge_gradient_label.pack(side=tk.TOP, padx=10, pady=10)
        self.edge_gradient_image = ttk.Label(self.edge_show_frame)
        self.edge_gradient_image.pack(side=tk.TOP, padx=10, pady=10)

        # 非最大值抑制前
        self.edge_before_nms_label = ttk.Label(self.edge_show_frame, text="非最大值抑制前")
        self.edge_before_nms_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.edge_before_nms_image = ttk.Label(self.edge_show_frame)
        self.edge_before_nms_image.pack(side=tk.LEFT, padx=10, pady=10)

        # Canny结果
        self.edge_canny_label = ttk.Label(self.edge_show_frame, text="Canny边缘检测")
        self.edge_canny_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.edge_canny_image = ttk.Label(self.edge_show_frame)
        self.edge_canny_image.pack(side=tk.LEFT, padx=10, pady=10)

        # 特征点检测显示区域
        self.feature_show_frame = ttk.LabelFrame(self.feature_display_frame, text="特征点检测结果", padding="10")
        self.feature_show_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Harris角点
        self.feature_harris_label = ttk.Label(self.feature_show_frame, text="Harris角点检测")
        self.feature_harris_label.pack(side=tk.TOP, padx=10, pady=10)
        self.feature_harris_image = ttk.Label(self.feature_show_frame)
        self.feature_harris_image.pack(side=tk.TOP, padx=10, pady=10)

        # SIFT特征点
        self.feature_sift_label = ttk.Label(self.feature_show_frame, text="SIFT特征点检测")
        self.feature_sift_label.pack(side=tk.TOP, padx=10, pady=10)
        self.feature_sift_image = ttk.Label(self.feature_show_frame)
        self.feature_sift_image.pack(side=tk.TOP, padx=10, pady=10)

        # 图像匹配显示区域
        self.match_show_frame = ttk.LabelFrame(self.match_display_frame, text="图像匹配结果", padding="10")
        self.match_show_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 第一幅图像
        self.match_img1_label = ttk.Label(self.match_show_frame, text="第一幅图像")
        self.match_img1_label.pack(side=tk.TOP, padx=10, pady=10)
        self.match_img1_image = ttk.Label(self.match_show_frame)
        self.match_img1_image.pack(side=tk.TOP, padx=10, pady=10)

        # 第二幅图像
        self.match_img2_label = ttk.Label(self.match_show_frame, text="第二幅图像")
        self.match_img2_label.pack(side=tk.TOP, padx=10, pady=10)
        self.match_img2_image = ttk.Label(self.match_show_frame)
        self.match_img2_image.pack(side=tk.TOP, padx=10, pady=10)

        # 匹配结果
        self.match_result_label = ttk.Label(self.match_show_frame, text="匹配结果")
        self.match_result_label.pack(side=tk.TOP, padx=10, pady=10)
        self.match_result_image = ttk.Label(self.match_show_frame)
        self.match_result_image.pack(side=tk.TOP, padx=10, pady=10)

        # 全景拼接显示区域
        self.panorama_show_frame = ttk.LabelFrame(self.panorama_display_frame, text="全景拼接", padding="10")
        self.panorama_show_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建输入图像显示区域
        self.panorama_input_frame = ttk.LabelFrame(self.panorama_show_frame, text="输入图像", padding="5")
        self.panorama_input_frame.pack(fill=tk.X, pady=5)

        # 创建一个滚动框架来显示所有输入图像
        self.input_canvas = tk.Canvas(self.panorama_input_frame, height=150)
        self.input_scrollbar = ttk.Scrollbar(self.panorama_input_frame, orient="horizontal", command=self.input_canvas.xview)
        self.input_canvas.configure(xscrollcommand=self.input_scrollbar.set)

        self.input_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.input_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

        # 在canvas中创建frame来放置图像
        self.input_images_frame = ttk.Frame(self.input_canvas)
        self.input_canvas.create_window((0, 0), window=self.input_images_frame, anchor="nw")

        # 绑定canvas大小调整事件
        self.input_images_frame.bind("<Configure>", lambda e: self.input_canvas.configure(scrollregion=self.input_canvas.bbox("all")))

        # 拼接结果显示区域
        self.panorama_result_frame = ttk.LabelFrame(self.panorama_show_frame, text="拼接结果", padding="5")
        self.panorama_result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.panorama_status_label = ttk.Label(self.panorama_result_frame, text="")
        self.panorama_status_label.pack(pady=5)

        self.panorama_result_image = ttk.Label(self.panorama_result_frame)
        self.panorama_result_image.pack(fill=tk.BOTH, expand=True)

        # 初始化控制面板，默认显示边缘检测
        self.current_tab = "边缘检测"
        self.on_tab_changed(None)

    def on_tab_changed(self, event):
        """处理tab切换事件"""
        self.current_tab = self.notebook.tab(self.notebook.select(), "text")
        print(f"Switched to tab: {self.current_tab}")
        
        if self.current_tab == "边缘检测":
            self.show_edge_display()
        elif self.current_tab == "特征点检测":
            self.show_feature_display()
        elif self.current_tab == "图像匹配":
            self.show_match_display()
        elif self.current_tab == "全景拼接":
            self.show_panorama_display()

    def show_edge_display(self):
        """显示边缘检测显示区域"""
        self.edge_display_frame.pack(fill=tk.BOTH, expand=True)
        self.feature_display_frame.pack_forget()
        self.match_display_frame.pack_forget()
        self.panorama_display_frame.pack_forget()
        if self.original_image is not None:
            self.apply_edge_detection()

    def show_feature_display(self):
        """显示特征点检测显示区域"""
        self.edge_display_frame.pack_forget()
        self.feature_display_frame.pack(fill=tk.BOTH, expand=True)
        self.match_display_frame.pack_forget()
        self.panorama_display_frame.pack_forget()
        if self.original_image is not None:
            self.apply_feature_detection()

    def show_match_display(self):
        """显示图像匹配显示区域"""
        self.edge_display_frame.pack_forget()
        self.feature_display_frame.pack_forget()
        self.match_display_frame.pack(fill=tk.BOTH, expand=True)
        self.panorama_display_frame.pack_forget()
        # 显示已加载的图像
        self.display_matching_images()

    def show_panorama_display(self):
        """显示全景拼接显示区域"""
        self.edge_display_frame.pack_forget()
        self.feature_display_frame.pack_forget()
        self.match_display_frame.pack_forget()
        self.panorama_display_frame.pack(fill=tk.BOTH, expand=True)
        # 显示已加载的图像
        self.display_panorama_images()

    def load_image(self):
        """加载图像"""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            if file_path:
                print(f"Loading image: {file_path}")
                self.image = cv2.imread(file_path)
                if self.image is None:
                    print("Failed to load image")
                    return
                print(f"Image loaded successfully: {self.image.shape}")
                self.original_image = self.image.copy()
                self.current_image = self.image.copy()
                
                # 根据当前功能显示相应的内容
                if self.current_tab == "边缘检测":
                    self.apply_edge_detection()
                elif self.current_tab == "特征点检测":
                    self.apply_feature_detection()
                elif self.current_tab == "图像匹配":
                    self.display_matching_images()
                elif self.current_tab == "全景拼接":
                    self.display_panorama_images()
        except Exception as e:
            print(f"Error loading image: {str(e)}")

    def load_multiple_images(self):
        """加载多幅图像用于全景拼接"""
        try:
            file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            if file_paths:
                self.images_list = []
                for file_path in file_paths:
                    img = cv2.imread(file_path)
                    if img is not None:
                        self.images_list.append(img)
                print(f"Loaded {len(self.images_list)} images for panorama stitching")
                if self.current_tab == "全景拼接":
                    self.display_panorama_images()
        except Exception as e:
            print(f"Error loading multiple images: {str(e)}")

    def load_second_image(self):
        """加载第二幅图像用于匹配"""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            if file_path:
                self.second_image = cv2.imread(file_path)
                print(f"Loaded second image: {self.second_image.shape}")
                if self.current_tab == "图像匹配":
                    self.display_matching_images()
        except Exception as e:
            print(f"Error loading second image: {str(e)}")

    def apply_edge_detection(self, event=None):
        """应用Canny边缘检测"""
        if self.original_image is not None:
            try:
                # 转换为灰度图
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

                # 高斯模糊去噪
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # 计算梯度幅值和方向
                sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = cv2.magnitude(sobelx, sobely)
                angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

                # 归一化梯度图像用于显示
                gradient = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                # 非最大值抑制前：应用双阈值但不进行非最大值抑制
                min_thresh = self.canny_min_var.get()
                max_thresh = self.canny_max_var.get()
                _, thresh_low = cv2.threshold(magnitude, min_thresh, 255, cv2.THRESH_BINARY)
                _, thresh_high = cv2.threshold(magnitude, max_thresh, 255, cv2.THRESH_BINARY)
                before_nms = cv2.bitwise_or(thresh_low.astype(np.uint8), thresh_high.astype(np.uint8))

                # 完整的Canny边缘检测（包括非最大值抑制）
                edges = cv2.Canny(blurred, min_thresh, max_thresh)

                # 显示结果
                self.display_edge_images(gradient, before_nms, edges)

            except Exception as e:
                print(f"Error in edge detection: {str(e)}")

    def apply_feature_detection(self, event=None):
        """应用特征点检测"""
        if self.original_image is not None:
            try:
                feature_type = self.feature_var.get()

                if feature_type == "Harris角点":
                    self.detect_harris_corners()
                elif feature_type == "SIFT特征点":
                    self.detect_sift_features()

            except Exception as e:
                print(f"Error in feature detection: {str(e)}")

    def detect_harris_corners(self):
        """检测Harris角点"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        block_size = self.harris_block_var.get()
        k = self.harris_k_var.get()

        # 检测Harris角点
        dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=3, k=k)

        # 结果标准化
        dst = cv2.dilate(dst, None)

        # 标记角点为圆形区域
        result = self.original_image.copy()
        result[dst > 0.01 * dst.max()] = [0, 0, 255]  # 红色标记角点

        # 获取角点位置
        corners = np.where(dst > 0.01 * dst.max())
        for y, x in zip(corners[0], corners[1]):
            # 绘制圆形区域以表示角点
            cv2.circle(result, (x, y), 5, (0, 255, 0), 2)  # 绿色圆圈

        # 显示结果
        self.display_feature_images(result, None)

    def detect_sift_features(self):
        """检测SIFT特征点"""
        try:
            # 创建SIFT检测器，降低阈值以检测更多特征点
            sift = cv2.SIFT_create(nfeatures=self.sift_nfeatures_var.get(),
                                 contrastThreshold=0.02,  # 降低对比度阈值
                                 edgeThreshold=5)         # 降低边缘阈值

            # 检测关键点和描述符
            keypoints, descriptors = sift.detectAndCompute(self.original_image, None)

            # 打印检测到的关键点数量用于调试
            print(f"Detected {len(keypoints)} SIFT keypoints")

            # 在图像上绘制关键点，使用圆形区域表示
            result = self.original_image.copy()
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size / 2)  # 圆圈大小基于关键点大小
                cv2.circle(result, (x, y), size, (0, 255, 0), 2)  # 绿色圆圈

            # 如果关键点很少，尝试在灰度图上检测
            if len(keypoints) < 10:
                print("Few keypoints detected, trying on grayscale image...")
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                keypoints_gray, descriptors_gray = sift.detectAndCompute(gray, None)
                print(f"Detected {len(keypoints_gray)} keypoints on grayscale")
                if len(keypoints_gray) > len(keypoints):
                    result = self.original_image.copy()
                    for kp in keypoints_gray:
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        size = int(kp.size / 2)
                        cv2.circle(result, (x, y), size, (255, 0, 0), 2)  # 蓝色圆圈
                    keypoints = keypoints_gray

            # 显示结果
            self.display_feature_images(None, result)

        except Exception as e:
            print(f"Error in SIFT detection: {str(e)}")

    def apply_image_matching(self):
        """应用图像匹配"""
        if self.original_image is not None and self.second_image is not None:
            try:
                method = self.match_method_var.get()

                if method == "SIFT":
                    self.match_images_sift()
                elif method == "ORB":
                    self.match_images_orb()

            except Exception as e:
                print(f"Error in image matching: {str(e)}")

    def draw_matches_with_lines(self, img1, kp1, img2, kp2, matches, mask=None):
        """自定义绘制匹配线，更清晰地显示连接"""
        # 获取图像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 创建结果图像
        result_height = max(h1, h2)
        result_width = w1 + w2
        result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
        
        # 放置两张图像
        result[0:h1, 0:w1] = img1
        result[0:h2, w1:w1+w2] = img2
        
        # 绘制匹配线
        for i, match in enumerate(matches):
            if mask is None or mask[i]:
                # 获取关键点坐标
                pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
                pt2 = (int(kp2[match.trainIdx].pt[0]) + w1, int(kp2[match.trainIdx].pt[1]))
                
                # 绘制连接线（彩虹色渐变）
                color = (
                    int(255 * (i % 10) / 10.0),  # B
                    int(255 * ((i * 7) % 10) / 10.0),  # G
                    int(255 * ((i * 3) % 10) / 10.0)   # R
                )
                cv2.line(result, pt1, pt2, color, 2)
                
                # 在关键点位置绘制圆圈
                cv2.circle(result, pt1, 5, (0, 255, 255), -1)
                cv2.circle(result, pt2, 5, (0, 255, 255), -1)
        
        return result

    def match_images_sift(self):
        """使用SIFT进行图像匹配"""
        try:
            # 创建SIFT检测器
            sift = cv2.SIFT_create(nfeatures=self.sift_nfeatures_var.get(), contrastThreshold=0.02, edgeThreshold=5)

            # 检测关键点和描述符
            kp1, des1 = sift.detectAndCompute(self.original_image, None)
            kp2, des2 = sift.detectAndCompute(self.second_image, None)

            # 使用FLANN匹配器
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            print(f"Found {len(good_matches)} good matches")

            # 如果匹配点足够多，使用RANSAC找到homography并筛选内点
            if len(good_matches) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # 应用变换并可视化
                    h, w = self.original_image.shape[:2]
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # 在第二幅图像上绘制边界框
                    img2_with_box = self.second_image.copy()
                    img2_with_box = cv2.polylines(img2_with_box, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # 使用自定义函数绘制匹配线（只绘制内点）
                    mask_ravel = mask.ravel()
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_ravel[i]]
                    
                    print(f"Inlier matches after RANSAC: {len(inlier_matches)}")
                    
                    # 绘制带连接线的匹配结果
                    result_combined = self.draw_matches_with_lines(
                        self.original_image, kp1, 
                        img2_with_box, kp2, 
                        inlier_matches
                    )
                    
                    # 显示合并后的结果
                    self.display_matching_result(result_combined)
                    return

            # 如果没有足够的匹配点或homography计算失败，直接显示所有匹配
            result_combined = self.draw_matches_with_lines(
                self.original_image, kp1, 
                self.second_image, kp2, 
                good_matches
            )
            self.display_matching_result(result_combined)

        except Exception as e:
            print(f"Error in SIFT matching: {str(e)}")
            # 出错时使用简单匹配
            try:
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(self.original_image, None)
                kp2, des2 = sift.detectAndCompute(self.second_image, None)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                result_combined = self.draw_matches_with_lines(
                    self.original_image, kp1, 
                    self.second_image, kp2, 
                    good_matches
                )
                self.display_matching_result(result_combined)
            except Exception as e2:
                print(f"Fallback matching also failed: {str(e2)}")

    def match_images_orb(self):
        """使用ORB进行图像匹配"""
        try:
            # 创建ORB检测器
            orb = cv2.ORB_create()

            # 检测关键点和描述符
            kp1, des1 = orb.detectAndCompute(self.original_image, None)
            kp2, des2 = orb.detectAndCompute(self.second_image, None)

            # 使用BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # 排序匹配结果
            matches = sorted(matches, key=lambda x: x.distance)

            # 选择前50个匹配
            good_matches = matches[:50]

            print(f"Found {len(good_matches)} good matches")

            # 如果匹配点足够多，使用RANSAC找到homography并筛选内点
            if len(good_matches) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # 应用变换并可视化
                    h, w = self.original_image.shape[:2]
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # 在第二幅图像上绘制边界框
                    img2_with_box = self.second_image.copy()
                    img2_with_box = cv2.polylines(img2_with_box, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # 使用自定义函数绘制匹配线（只绘制内点）
                    mask_ravel = mask.ravel()
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_ravel[i]]
                    
                    print(f"Inlier matches after RANSAC: {len(inlier_matches)}")
                    
                    # 绘制带连接线的匹配结果
                    result_combined = self.draw_matches_with_lines(
                        self.original_image, kp1, 
                        img2_with_box, kp2, 
                        inlier_matches
                    )
                    
                    # 显示合并后的结果
                    self.display_matching_result(result_combined)
                    return

            # 如果没有足够的匹配点或homography计算失败，直接显示所有匹配
            result_combined = self.draw_matches_with_lines(
                self.original_image, kp1, 
                self.second_image, kp2, 
                good_matches
            )
            self.display_matching_result(result_combined)

        except Exception as e:
            print(f"Error in ORB matching: {str(e)}")
            # 出错时使用简单匹配
            try:
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(self.original_image, None)
                kp2, des2 = orb.detectAndCompute(self.second_image, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)[:50]
                
                result_combined = self.draw_matches_with_lines(
                    self.original_image, kp1, 
                    self.second_image, kp2, 
                    matches
                )
                self.display_matching_result(result_combined)
            except Exception as e2:
                print(f"Fallback matching also failed: {str(e2)}")

    def display_edge_images(self, gradient, before_nms, canny):
        """显示边缘检测结果"""
        try:
            # 显示梯度图像
            gradient_rgb = cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB)
            gradient_pil = Image.fromarray(gradient_rgb)
            gradient_resized = gradient_pil.resize((300, 200), Image.LANCZOS)
            gradient_tk = ImageTk.PhotoImage(gradient_resized)
            self.edge_gradient_image.config(image=gradient_tk)
            self.edge_gradient_image.image = gradient_tk

            # 显示非最大值抑制前
            before_nms_rgb = cv2.cvtColor(before_nms, cv2.COLOR_GRAY2RGB)
            before_nms_pil = Image.fromarray(before_nms_rgb)
            before_nms_resized = before_nms_pil.resize((300, 200), Image.LANCZOS)
            before_nms_tk = ImageTk.PhotoImage(before_nms_resized)
            self.edge_before_nms_image.config(image=before_nms_tk)
            self.edge_before_nms_image.image = before_nms_tk

            # 显示Canny结果
            canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            canny_pil = Image.fromarray(canny_rgb)
            canny_resized = canny_pil.resize((300, 200), Image.LANCZOS)
            canny_tk = ImageTk.PhotoImage(canny_resized)
            self.edge_canny_image.config(image=canny_tk)
            self.edge_canny_image.image = canny_tk

        except Exception as e:
            print(f"Error displaying edge images: {str(e)}")

    def display_feature_images(self, harris_result, sift_result):
        """显示特征点检测结果"""
        try:
            if harris_result is not None:
                # 显示Harris结果
                harris_rgb = cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB)
                harris_pil = Image.fromarray(harris_rgb)
                harris_resized = harris_pil.resize((400, 300), Image.LANCZOS)
                harris_tk = ImageTk.PhotoImage(harris_resized)
                self.feature_harris_image.config(image=harris_tk)
                self.feature_harris_image.image = harris_tk

            if sift_result is not None:
                # 显示SIFT结果，增大显示尺寸以使特征点更明显
                sift_rgb = cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB)
                sift_pil = Image.fromarray(sift_rgb)
                sift_resized = sift_pil.resize((600, 450), Image.LANCZOS)  # 增大尺寸
                sift_tk = ImageTk.PhotoImage(sift_resized)
                self.feature_sift_image.config(image=sift_tk)
                self.feature_sift_image.image = sift_tk

        except Exception as e:
            print(f"Error displaying feature images: {str(e)}")

    def display_matching_images(self):
        """显示匹配图像"""
        try:
            if self.original_image is not None:
                # 显示第一幅图像
                img1_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                img1_pil = Image.fromarray(img1_rgb)
                img1_resized = img1_pil.resize((300, 200), Image.LANCZOS)
                img1_tk = ImageTk.PhotoImage(img1_resized)
                self.match_img1_image.config(image=img1_tk)
                self.match_img1_image.image = img1_tk

            if self.second_image is not None:
                # 显示第二幅图像
                img2_rgb = cv2.cvtColor(self.second_image, cv2.COLOR_BGR2RGB)
                img2_pil = Image.fromarray(img2_rgb)
                img2_resized = img2_pil.resize((300, 200), Image.LANCZOS)
                img2_tk = ImageTk.PhotoImage(img2_resized)
                self.match_img2_image.config(image=img2_tk)
                self.match_img2_image.image = img2_tk

        except Exception as e:
            print(f"Error displaying matching images: {str(e)}")

    def display_matching_result(self, result):
        """显示匹配结果"""
        try:
            # 调整结果图像大小以适应显示区域
            h, w = result.shape[:2]
            max_height = 500
            max_width = 1000
            
            if h > max_height or w > max_width:
                ratio = min(max_width / w, max_height / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                result = cv2.resize(result, (new_w, new_h))
            
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            result_tk = ImageTk.PhotoImage(result_pil)
            self.match_result_image.config(image=result_tk)
            self.match_result_image.image = result_tk

        except Exception as e:
            print(f"Error displaying matching result: {str(e)}")

    def display_panorama_images(self):
        """显示全景拼接的输入图像"""
        try:
            # 清空之前的图像
            for widget in self.input_images_frame.winfo_children():
                widget.destroy()

            for i, img in enumerate(self.images_list):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_resized = img_pil.resize((150, 100), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_resized)

                # 创建标签显示图像
                img_label = ttk.Label(self.input_images_frame, image=img_tk)
                img_label.image = img_tk
                img_label.pack(side=tk.LEFT, padx=5, pady=5)

        except Exception as e:
            print(f"Error displaying panorama images: {str(e)}")

    def display_panorama_result(self, pano):
        """显示全景拼接结果"""
        try:
            print("Displaying panorama result")
            pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
            pano_pil = Image.fromarray(pano_rgb)
            # 调整大小以适应显示
            height, width = pano.shape[:2]
            print(f"Original panorama size: {width}x{height}")
            max_width = 800
            max_height = 600
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            print(f"Resized to: {new_width}x{new_height}")
            pano_resized = pano_pil.resize((new_width, new_height), Image.LANCZOS)
            pano_tk = ImageTk.PhotoImage(pano_resized)
            self.panorama_result_image.config(image=pano_tk)
            self.panorama_result_image.image = pano_tk
            print("Panorama displayed successfully")
            self.panorama_status_label.config(text="拼接成功")

        except Exception as e:
            print(f"Error displaying panorama result: {str(e)}")

    def apply_panorama_stitching(self):
        """应用全景拼接"""
        if len(self.images_list) > 1:
            self.panorama_status_label.config(text="正在处理中...")
            # 使用线程避免冻结GUI
            threading.Thread(target=self._stitch_images, daemon=True).start()
        else:
            self.panorama_status_label.config(text="需要至少2幅图像")

    def _stitch_images(self):
        """在后台线程中执行拼接"""
        try:
            blend_method = self.blend_var.get()
            print(f"Starting stitching with {len(self.images_list)} images, blend method: {blend_method}")

            # 使用OpenCV的Stitcher
            stitcher = cv2.Stitcher_create()
            status, pano = stitcher.stitch(self.images_list)
            print(f"Stitcher status: {status}")

            # 在主线程中更新GUI
            self.root.after(0, lambda: self._update_stitch_result(status, pano))

        except Exception as e:
            print(f"Error in panorama stitching: {str(e)}")
            self.root.after(0, lambda: self.panorama_status_label.config(text=f"拼接错误: {str(e)}"))

    def _update_stitch_result(self, status, pano):
        """在主线程中更新拼接结果"""
        if status == cv2.Stitcher_OK:
            print("Stitching successful")
            if pano is not None and pano.size > 0:
                print(f"Panorama shape: {pano.shape}")
                self.panorama_status_label.config(text="拼接成功")
                # 显示结果
                self.display_panorama_result(pano)
            else:
                print("Panorama is None or empty")
                self.panorama_status_label.config(text="拼接结果为空")
        else:
            print(f"Stitching failed with status: {status}")
            self.panorama_status_label.config(text=f"拼接失败，状态码: {status}")

if __name__ == "__main__":
    if _HAS_TK:
        root = tk.Tk()
        app = ComputerVisionApp(root)
        root.mainloop()
    else:
        print("tkinter 不可用，请通过 Streamlit 启动：streamlit run app.py")