import cv2
import numpy as np

# 创建测试图像
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# 测试滤波器
# Box滤波
box_filtered = cv2.blur(image, (5, 5))
print("Box filter applied")

# Gaussian滤波
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
print("Gaussian filter applied")

# Median滤波
median_filtered = cv2.medianBlur(image, 5)
print("Median filter applied")

# Sobel滤波
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
print("Sobel filter applied")

# 测试梯度
region = gray[10:50, 10:50]
sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
gradient_direction = cv2.phase(sobelx, sobely, angleInDegrees=True)
print("Gradient computed")

# 测试频率域
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
print("FFT computed")

print("All tests passed")