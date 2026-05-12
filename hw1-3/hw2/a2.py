import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# ==================== 核心处理函数（供Streamlit调用）====================

def apply_spatial_filter(image, filter_type, kernel_size):
    """应用空间滤波器"""
    if filter_type == "Box滤波":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == "Gaussian滤波":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "Median滤波":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "Sobel滤波":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    return image


def compute_gradient(image, x1, y1, x2, y2):
    """计算图像局部区域的梯度"""
    h, w = image.shape[:2]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    
    region = image[y1:y2, x1:x2]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_direction = cv2.phase(sobelx, sobely, angleInDegrees=True)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    hsv = np.zeros_like(region)
    hsv[..., 0] = gradient_direction / 2
    hsv[..., 1] = 255
    hsv[..., 2] = gradient_magnitude
    gradient_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 在原图上画框
    marked = image.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return marked, gradient_color


def compute_spectrum(image):
    """计算图像的傅里叶频谱图"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return magnitude_spectrum


def apply_frequency_transform(image, transform_type, **kwargs):
    """对图像应用空间域变换以观察频谱变化"""
    if transform_type == "原始":
        return image.copy()
    elif transform_type == "旋转":
        angle = kwargs.get("angle", 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    elif transform_type == "平移":
        shift = kwargs.get("shift", 50)
        h, w = image.shape[:2]
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        return cv2.warpAffine(image, M, (w, h))
    elif transform_type == "缩放":
        scale = kwargs.get("scale", 0.5)
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image.copy()


class ColorSpaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("计算机视觉课程设计 - 颜色空间与插值演示")
        self.root.geometry("1200x900")
        
        # 初始化变量
        self.image = None
        self.original_image = None
        self.current_image = None
        self.transformed_image = None
        self.small_image = None  # 缩放到0.3倍的图像
        self.freq_transformed_data = None
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 创建右侧显示区域
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # 创建功能选择标签
        ttk.Label(self.control_frame, text="功能选择").pack(pady=5)
        
        # 创建功能选择下拉菜单
        self.function_var = tk.StringVar()
        self.function_combobox = ttk.Combobox(self.control_frame, textvariable=self.function_var, values=["空间滤波器演示", "图像梯度演示", "频率域滤波演示"])
        self.function_combobox.current(0)
        self.function_combobox.pack(pady=5, fill=tk.X)
        self.function_combobox.bind("<<ComboboxSelected>>", self.update_control_panel)
        
        # 创建加载图像按钮
        ttk.Button(self.control_frame, text="加载图像", command=self.load_image).pack(pady=5, fill=tk.X)
        
        # 创建空间滤波器控制面板
        self.filter_frame = ttk.LabelFrame(self.control_frame, text="空间滤波器设置", padding="10")
        
        # 滤波器选择
        ttk.Label(self.filter_frame, text="滤波器类型").pack(pady=5)
        self.filter_var = tk.StringVar()
        self.filter_combobox = ttk.Combobox(self.filter_frame, textvariable=self.filter_var, values=["Box滤波", "Gaussian滤波", "Median滤波", "Sobel滤波"])
        self.filter_combobox.current(0)
        self.filter_combobox.pack(pady=5, fill=tk.X)
        self.filter_combobox.bind("<<ComboboxSelected>>", self.apply_filter)
        
        # 核大小
        ttk.Label(self.filter_frame, text="核大小").pack(pady=5)
        self.kernel_var = tk.IntVar(value=6)
        ttk.Scale(self.filter_frame, from_=6, to=20, variable=self.kernel_var, orient=tk.HORIZONTAL, command=self.apply_filter).pack(pady=5, fill=tk.X)
        
        # 创建图像梯度控制面板
        self.gradient_frame = ttk.LabelFrame(self.control_frame, text="图像梯度设置", padding="10")
        
        # 计算梯度按钮
        ttk.Button(self.gradient_frame, text="计算梯度", command=self.compute_gradient).pack(pady=5, fill=tk.X)
        
        # 区域选择
        ttk.Label(self.gradient_frame, text="选择区域 (x1,y1,x2,y2)").pack(pady=5)
        self.region_var = tk.StringVar(value="100,100,200,200")
        ttk.Entry(self.gradient_frame, textvariable=self.region_var).pack(pady=5, fill=tk.X)
        
        # 创建频率域滤波控制面板
        self.frequency_frame = ttk.LabelFrame(self.control_frame, text="频率域滤波设置", padding="10")
        
        # 变换类型选择
        ttk.Label(self.frequency_frame, text="变换类型").pack(pady=5)
        self.freq_transform_var = tk.StringVar()
        self.freq_transform_combobox = ttk.Combobox(self.frequency_frame, textvariable=self.freq_transform_var, values=["原始", "旋转", "平移", "缩放"])
        self.freq_transform_combobox.current(0)
        self.freq_transform_combobox.pack(pady=5, fill=tk.X)
        self.freq_transform_combobox.bind("<<ComboboxSelected>>", self.apply_frequency_transform)
        
        # 旋转角度
        ttk.Label(self.frequency_frame, text="旋转角度").pack(pady=5)
        self.freq_angle_var = tk.DoubleVar(value=30.0)
        ttk.Scale(self.frequency_frame, from_=0, to=360, variable=self.freq_angle_var, orient=tk.HORIZONTAL, command=self.apply_frequency_transform).pack(pady=5, fill=tk.X)
        
        # 平移距离
        ttk.Label(self.frequency_frame, text="平移距离").pack(pady=5)
        self.freq_shift_var = tk.IntVar(value=50)
        ttk.Scale(self.frequency_frame, from_=0, to=200, variable=self.freq_shift_var, orient=tk.HORIZONTAL, command=self.apply_frequency_transform).pack(pady=5, fill=tk.X)
        
        # 缩放比例
        ttk.Label(self.frequency_frame, text="缩放比例").pack(pady=5)
        self.freq_scale_var = tk.DoubleVar(value=0.5)
        ttk.Scale(self.frequency_frame, from_=0.1, to=3.0, variable=self.freq_scale_var, orient=tk.HORIZONTAL, command=self.apply_frequency_transform).pack(pady=5, fill=tk.X)
        
        # 创建颜色空间显示区域
        self.color_space_display_frame = ttk.Frame(self.display_frame)
        
        # 通道显示区域（上方）
        self.channels_frame = ttk.LabelFrame(self.color_space_display_frame, text="颜色通道", padding="10")
        self.channels_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建通道标签
        self.channel1_label = ttk.Label(self.channels_frame, text="通道1")
        self.channel1_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.channel1_image = ttk.Label(self.channels_frame)
        self.channel1_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.channel2_label = ttk.Label(self.channels_frame, text="通道2")
        self.channel2_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.channel2_image = ttk.Label(self.channels_frame)
        self.channel2_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.channel3_label = ttk.Label(self.channels_frame, text="通道3")
        self.channel3_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.channel3_image = ttk.Label(self.channels_frame)
        self.channel3_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 原图和处理后图像显示区域（下方）
        self.color_images_frame = ttk.Frame(self.color_space_display_frame)
        self.color_images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建原始图像标签
        self.original_label = ttk.Label(self.color_images_frame, text="原始图像")
        self.original_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.original_image_label = ttk.Label(self.color_images_frame)
        self.original_image_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 创建处理后图像标签
        self.processed_label = ttk.Label(self.color_images_frame, text="处理后图像")
        self.processed_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.processed_image_label = ttk.Label(self.color_images_frame)
        self.processed_image_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 创建空间滤波器显示区域
        self.filter_display_frame = ttk.Frame(self.display_frame)
        
        # 滤波器对比显示区域
        self.filter_compare_frame = ttk.LabelFrame(self.filter_display_frame, text="滤波器对比", padding="10")
        self.filter_compare_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 原始图像
        self.filter_original_label = ttk.Label(self.filter_compare_frame, text="原始图像")
        self.filter_original_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.filter_original_image = ttk.Label(self.filter_compare_frame)
        self.filter_original_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 滤波后图像
        self.filter_processed_label = ttk.Label(self.filter_compare_frame, text="滤波后图像")
        self.filter_processed_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.filter_processed_image = ttk.Label(self.filter_compare_frame)
        self.filter_processed_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 创建图像梯度显示区域
        self.gradient_display_frame = ttk.Frame(self.display_frame)
        
        # 梯度显示区域
        self.gradient_show_frame = ttk.LabelFrame(self.gradient_display_frame, text="梯度演示", padding="10")
        self.gradient_show_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 原始图像
        self.gradient_original_label = ttk.Label(self.gradient_show_frame, text="原始图像")
        self.gradient_original_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.gradient_original_image = ttk.Label(self.gradient_show_frame)
        self.gradient_original_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 梯度图像
        self.gradient_result_label = ttk.Label(self.gradient_show_frame, text="梯度图像")
        self.gradient_result_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.gradient_result_image = ttk.Label(self.gradient_show_frame)
        self.gradient_result_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 创建频率域滤波显示区域
        self.frequency_display_frame = ttk.Frame(self.display_frame)
        
        # 频率域显示区域
        self.frequency_show_frame = ttk.LabelFrame(self.frequency_display_frame, text="频率域分析", padding="10")
        self.frequency_show_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 原始图像和谱图
        self.freq_original_label = ttk.Label(self.frequency_show_frame, text="原始图像")
        self.freq_original_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.freq_original_image = ttk.Label(self.frequency_show_frame)
        self.freq_original_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.freq_spectrum_label = ttk.Label(self.frequency_show_frame, text="频谱图")
        self.freq_spectrum_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.freq_spectrum_image = ttk.Label(self.frequency_show_frame)
        self.freq_spectrum_image.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 变换后图像和谱图
        self.freq_transformed_label = ttk.Label(self.frequency_show_frame, text="变换后图像")
        self.freq_transformed_label.pack(side=tk.TOP, padx=10, pady=10)
        self.freq_transformed_image = ttk.Label(self.frequency_show_frame)
        self.freq_transformed_image.pack(side=tk.TOP, padx=10, pady=10)
        
        self.freq_transformed_spectrum_label = ttk.Label(self.frequency_show_frame, text="变换后频谱图")
        self.freq_transformed_spectrum_label.pack(side=tk.TOP, padx=10, pady=10)
        self.freq_transformed_spectrum_image = ttk.Label(self.frequency_show_frame)
        self.freq_transformed_spectrum_image.pack(side=tk.TOP, padx=10, pady=10)
        
        # 初始化控制面板
        self.update_control_panel()
    
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
                
                # 计算缩小到0.3倍的图像
                h, w = self.original_image.shape[:2]
                small_w = int(w * 0.3)
                small_h = int(h * 0.3)
                self.small_image = cv2.resize(self.original_image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                print(f"Small image created: {self.small_image.shape}")
                
                # 根据当前功能显示相应的内容
                function = self.function_var.get()
                if function == "颜色空间演示":
                    self.display_images()
                elif function == "图像插值演示":
                    self.display_interpolation_input()
                elif function == "空间滤波器演示":
                    self.display_filter_images()
                elif function == "图像梯度演示":
                    self.display_gradient_images()
                elif function == "频率域滤波演示":
                    self.display_frequency_images()
        except Exception as e:
            print(f"Error loading image: {str(e)}")
    
    def display_interpolation_input(self):
        """显示插值输入图像（原始图像和缩小图像）"""
        if self.original_image is not None:
            try:
                # 显示原始图像
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                original_pil = Image.fromarray(original_rgb)
                original_resized = original_pil.resize((250, 180), Image.LANCZOS)
                original_tk = ImageTk.PhotoImage(original_resized)
                self.interpolation_original_image.config(image=original_tk)
                self.interpolation_original_image.image = original_tk
                
                # 显示缩小到0.3倍的图像
                if self.small_image is not None:
                    small_rgb = cv2.cvtColor(self.small_image, cv2.COLOR_BGR2RGB)
                    small_pil = Image.fromarray(small_rgb)
                    small_resized = small_pil.resize((250, 180), Image.LANCZOS)
                    small_tk = ImageTk.PhotoImage(small_resized)
                    self.small_image_display.config(image=small_tk)
                    self.small_image_display.image = small_tk
            except Exception as e:
                print(f"Error displaying interpolation input: {str(e)}")
    
    def display_images(self):
        """显示图像"""
        try:
            if self.original_image is not None:
                print(f"Displaying images, original shape: {self.original_image.shape}")
                
                # 显示原始图像
                try:
                    original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                    print("Original image converted to RGB")
                    
                    original_pil = Image.fromarray(original_rgb)
                    print("Original image converted to PIL")
                    
                    original_resized = original_pil.resize((400, 300), Image.LANCZOS)
                    print("Original image resized")
                    
                    original_tk = ImageTk.PhotoImage(original_resized)
                    print("Original image converted to Tkinter")
                    
                    self.original_image_label.config(image=original_tk)
                    self.original_image_label.image = original_tk
                    print("Original image displayed")
                except Exception as e:
                    print(f"Error displaying original image: {str(e)}")
                
                # 显示处理后图像
                if self.current_image is not None:
                    try:
                        current_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                        print("Current image converted to RGB")
                        
                        current_pil = Image.fromarray(current_rgb)
                        print("Current image converted to PIL")
                        
                        current_resized = current_pil.resize((400, 300), Image.LANCZOS)
                        print("Current image resized")
                        
                        current_tk = ImageTk.PhotoImage(current_resized)
                        print("Current image converted to Tkinter")
                        
                        self.processed_image_label.config(image=current_tk)
                        self.processed_image_label.image = current_tk
                        print("Processed image displayed")
                    except Exception as e:
                        print(f"Error displaying processed image: {str(e)}")
                        
                # 显示颜色通道
                try:
                    self.display_channels()
                except Exception as e:
                    print(f"Error displaying channels: {str(e)}")
                
                # 强制更新界面
                self.root.update_idletasks()
                print("Interface updated")
        except Exception as e:
            print(f"Error in display_images: {str(e)}")
    
    def display_channels(self):
        """显示颜色通道"""
        if self.current_image is not None:
            color_space = self.color_space_var.get()
            
            # 根据颜色空间提取通道
            if color_space == "BGR":
                channels = cv2.split(self.current_image)
                channel_names = ["Blue", "Green", "Red"]
            elif color_space == "RGB":
                channels = cv2.split(cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                channel_names = ["Red", "Green", "Blue"]
            elif color_space == "HSV":
                hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
                channels = cv2.split(hsv)
                channel_names = ["Hue", "Saturation", "Value"]
            elif color_space == "Lab":
                lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2Lab)
                channels = cv2.split(lab)
                channel_names = ["L", "a", "b"]
            elif color_space == "YCrCb":
                ycrcb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YCrCb)
                channels = cv2.split(ycrcb)
                channel_names = ["Y", "Cr", "Cb"]
            else:
                channels = cv2.split(self.current_image)
                channel_names = ["Channel 1", "Channel 2", "Channel 3"]
            
            # 更新通道标签
            self.channel1_label.config(text=channel_names[0])
            self.channel2_label.config(text=channel_names[1])
            self.channel3_label.config(text=channel_names[2])
            
            # 显示每个通道
            channel_labels = [self.channel1_image, self.channel2_image, self.channel3_image]
            for i, (channel, label) in enumerate(zip(channels, channel_labels)):
                try:
                    # 将单通道图像转换为三通道，以便显示
                    channel_3ch = cv2.merge([channel, channel, channel])
                    channel_rgb = cv2.cvtColor(channel_3ch, cv2.COLOR_BGR2RGB)
                    channel_pil = Image.fromarray(channel_rgb)
                    channel_resized = channel_pil.resize((200, 150), Image.LANCZOS)
                    channel_tk = ImageTk.PhotoImage(channel_resized)
                    label.config(image=channel_tk)
                    label.image = channel_tk
                except Exception as e:
                    print(f"Error processing channel {i}: {e}")
            
    def update_control_panel(self, event=None):
        """更新控制面板"""
        function = self.function_var.get()
        if function == "空间滤波器演示":
            # 显示滤波器控制面板
            self.filter_frame.pack(fill=tk.X, pady=5)
            self.gradient_frame.pack_forget()
            self.frequency_frame.pack_forget()
            
            # 显示滤波器显示区域
            self.filter_display_frame.pack(fill=tk.BOTH, expand=True)
            self.gradient_display_frame.pack_forget()
            self.frequency_display_frame.pack_forget()
            
            # 如果有图像，显示滤波器对比
            if self.original_image is not None:
                self.display_filter_images()
        elif function == "图像梯度演示":
            # 显示梯度控制面板
            self.filter_frame.pack_forget()
            self.gradient_frame.pack(fill=tk.X, pady=5)
            self.frequency_frame.pack_forget()
            
            # 显示梯度显示区域
            self.filter_display_frame.pack_forget()
            self.gradient_display_frame.pack(fill=tk.BOTH, expand=True)
            self.frequency_display_frame.pack_forget()
            
            # 如果有图像，显示梯度
            if self.original_image is not None:
                self.display_gradient_images()
        elif function == "频率域滤波演示":
            # 显示频率域控制面板
            self.filter_frame.pack_forget()
            self.gradient_frame.pack_forget()
            self.frequency_frame.pack(fill=tk.X, pady=5)
            
            # 显示频率域显示区域
            self.filter_display_frame.pack_forget()
            self.gradient_display_frame.pack_forget()
            self.frequency_display_frame.pack(fill=tk.BOTH, expand=True)
            
            # 如果有图像，显示频率域分析
            if self.original_image is not None:
                self.display_frequency_images()
    
    def apply_filter(self, event=None):
        """应用空间滤波器"""
        if self.original_image is not None:
            filter_type = self.filter_var.get()
            kernel_size = self.kernel_var.get()
            # 确保核大小为奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            if filter_type == "Box滤波":
                self.current_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
            elif filter_type == "Gaussian滤波":
                self.current_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
            elif filter_type == "Median滤波":
                self.current_image = cv2.medianBlur(self.original_image, kernel_size)
            elif filter_type == "Sobel滤波":
                # 转换为灰度图
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                # 计算Sobel梯度
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
                sobel = cv2.magnitude(sobelx, sobely)
                # 归一化到0-255
                sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # 转换为三通道
                self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            
            self.display_filter_images()
    
    def compute_gradient(self):
        """计算图像梯度"""
        if self.original_image is not None:
            # 获取区域坐标
            try:
                coords = self.region_var.get().split(',')
                x1, y1, x2, y2 = map(int, coords)
            except:
                x1, y1, x2, y2 = 100, 100, 200, 200
            
            # 确保坐标在图像范围内
            h, w = self.original_image.shape[:2]
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            
            # 提取区域
            region = self.original_image[y1:y2, x1:x2]
            
            # 转换为灰度图
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # 计算梯度
            sobelx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度方向
            gradient_direction = cv2.phase(sobelx, sobely, angleInDegrees=True)
            
            # 计算梯度幅度
            gradient_magnitude = cv2.magnitude(sobelx, sobely)
            
            # 归一化幅度到0-255
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # 创建彩色梯度图像（方向用色相表示，幅度用亮度表示）
            hsv = np.zeros_like(region)
            hsv[..., 0] = gradient_direction / 2  # 方向映射到0-180度
            hsv[..., 1] = 255  # 饱和度最大
            hsv[..., 2] = gradient_magnitude  # 幅度作为亮度
            
            gradient_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 在原始图像上绘制区域框
            self.current_image = self.original_image.copy()
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 保存梯度图像
            self.gradient_image = gradient_color
            
            self.display_gradient_images()
    
    def apply_frequency_transform(self, event=None):
        """应用频率域变换"""
        if self.original_image is not None:
            transform_type = self.freq_transform_var.get()
            
            if transform_type == "原始":
                self.freq_transformed_data = self.original_image.copy()
            elif transform_type == "旋转":
                angle = self.freq_angle_var.get()
                h, w = self.original_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                self.freq_transformed_data = cv2.warpAffine(self.original_image, M, (w, h))
            elif transform_type == "平移":
                shift = self.freq_shift_var.get()
                h, w = self.original_image.shape[:2]
                M = np.float32([[1, 0, shift], [0, 1, shift]])
                self.freq_transformed_data = cv2.warpAffine(self.original_image, M, (w, h))
            elif transform_type == "缩放":
                scale = self.freq_scale_var.get()
                h, w = self.original_image.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                self.freq_transformed_data = cv2.resize(self.original_image, (new_w, new_h))
            
            self.display_frequency_images()
    
    def display_filter_images(self):
        """显示滤波器对比"""
        if self.original_image is not None:
            try:
                # 显示原始图像
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                original_pil = Image.fromarray(original_rgb)
                original_resized = original_pil.resize((300, 200), Image.LANCZOS)
                original_tk = ImageTk.PhotoImage(original_resized)
                self.filter_original_image.config(image=original_tk)
                self.filter_original_image.image = original_tk
                
                # 显示滤波后图像
                if hasattr(self, 'current_image') and self.current_image is not None:
                    processed_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                    processed_pil = Image.fromarray(processed_rgb)
                    processed_resized = processed_pil.resize((300, 200), Image.LANCZOS)
                    processed_tk = ImageTk.PhotoImage(processed_resized)
                    self.filter_processed_image.config(image=processed_tk)
                    self.filter_processed_image.image = processed_tk
            except Exception as e:
                print(f"Error displaying filter images: {str(e)}")
    
    def display_gradient_images(self):
        """显示梯度图像"""
        if self.original_image is not None:
            try:
                # 显示原始图像（带区域框）
                if hasattr(self, 'current_image') and self.current_image is not None:
                    original_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                else:
                    original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                original_pil = Image.fromarray(original_rgb)
                original_resized = original_pil.resize((300, 200), Image.LANCZOS)
                original_tk = ImageTk.PhotoImage(original_resized)
                self.gradient_original_image.config(image=original_tk)
                self.gradient_original_image.image = original_tk
                
                # 显示梯度图像
                if hasattr(self, 'gradient_image') and self.gradient_image is not None:
                    gradient_rgb = cv2.cvtColor(self.gradient_image, cv2.COLOR_BGR2RGB)
                    gradient_pil = Image.fromarray(gradient_rgb)
                    gradient_resized = gradient_pil.resize((300, 200), Image.LANCZOS)
                    gradient_tk = ImageTk.PhotoImage(gradient_resized)
                    self.gradient_result_image.config(image=gradient_tk)
                    self.gradient_result_image.image = gradient_tk
            except Exception as e:
                print(f"Error displaying gradient images: {str(e)}")
    
    def display_frequency_images(self):
        """显示频率域分析"""
        if self.original_image is not None:
            try:
                # 显示原始图像
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                original_pil = Image.fromarray(original_rgb)
                original_resized = original_pil.resize((200, 150), Image.LANCZOS)
                original_tk = ImageTk.PhotoImage(original_resized)
                self.freq_original_image.config(image=original_tk)
                self.freq_original_image.image = original_tk
                
                # 计算并显示原始频谱
                self.display_spectrum(self.original_image, self.freq_spectrum_image)
                
                # 显示变换后图像
                if self.freq_transformed_data is not None:
                    transformed_rgb = cv2.cvtColor(self.freq_transformed_data, cv2.COLOR_BGR2RGB)
                    transformed_pil = Image.fromarray(transformed_rgb)
                    transformed_resized = transformed_pil.resize((200, 150), Image.LANCZOS)
                    transformed_tk = ImageTk.PhotoImage(transformed_resized)
                    self.freq_transformed_image.config(image=transformed_tk)
                    self.freq_transformed_image.image = transformed_tk
                    
                    # 计算并显示变换后频谱
                    self.display_spectrum(self.freq_transformed_data, self.freq_transformed_spectrum_image)
            except Exception as e:
                print(f"Error displaying frequency images: {str(e)}")
    
    def display_spectrum(self, image, label):
        """显示频谱图"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 进行傅里叶变换
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # 归一化到0-255
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # 转换为PIL图像
            spectrum_pil = Image.fromarray(magnitude_spectrum)
            spectrum_resized = spectrum_pil.resize((200, 150), Image.LANCZOS)
            spectrum_tk = ImageTk.PhotoImage(spectrum_resized)
            label.config(image=spectrum_tk)
            label.image = spectrum_tk
        except Exception as e:
            print(f"Error displaying spectrum: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorSpaceApp(root)
    root.mainloop()