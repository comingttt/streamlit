"""
模块1: FCN语义分割模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FCN32s(nn.Module):
    """
    FCN-32s 语义分割网络
    基于VGG16 backbone，将全连接层替换为卷积层，一次性上采样32倍
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super(FCN32s, self).__init__()

        # 使用预训练的VGG16作为backbone
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)

        # 提取VGG16的特征提取部分（去掉全连接层和最后的maxpool）
        features = list(vgg.features.children())

        self.features_block1 = nn.Sequential(*features[:5])   # conv1_1, relu, conv1_2, relu, pool1
        self.features_block2 = nn.Sequential(*features[5:10])  # conv2_x, pool2
        self.features_block3 = nn.Sequential(*features[10:17]) # conv3_x, pool3
        self.features_block4 = nn.Sequential(*features[17:24]) # conv4_x, pool4
        self.features_block5 = nn.Sequential(*features[24:31]) # conv5_x, pool5

        # 将VGG的全连接层替换为卷积层
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        # 分数层：输出每个类别的分数
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # 上采样层：32倍上采样
        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=64, stride=32, padding=16, bias=False
        )

        self._init_weights()

    def _init_weights(self):
        """初始化新增层的权重"""
        for m in [self.fc6, self.fc7, self.score_fr]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 上采样层使用双线性插值初始化
        nn.init.zeros_(self.upscore.weight)
        # 双线性插值核
        c = self.upscore.weight.shape[0]
        k = self.upscore.kernel_size[0]
        bilinear_kernel = self._bilinear_kernel(c, c, k)
        self.upscore.weight.data.copy_(bilinear_kernel)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        """生成双线性插值卷积核"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = (torch.arange(kernel_size).float() - center) / factor
        og = og.view(1, -1).repeat(kernel_size, 1)
        kernel = (1 - og.abs()) * (1 - og.T.abs())
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(in_channels, out_channels, 1, 1)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            (B, num_classes, H, W) 分割分数图
        """
        input_h, input_w = x.shape[2], x.shape[3]

        # Encoder
        x = self.features_block1(x)
        x = self.features_block2(x)
        pool3 = self.features_block3(x)  # 1/8
        pool4 = self.features_block4(pool3)  # 1/16
        pool5 = self.features_block5(pool4)  # 1/32

        # 分类器（卷积化的全连接层）
        x = self.fc6(pool5)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        # 分数图
        x = self.score_fr(x)

        # 32倍上采样
        x = self.upscore(x)

        # 裁剪到输入尺寸（处理padding带来的尺寸差异）
        x = x[:, :, 16:16 + input_h, 16:16 + input_w]

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测分割mask"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            pred = torch.argmax(logits, dim=1)  # (B, H, W)
        return pred


class FCN8s(nn.Module):
    """
    FCN-8s 语义分割网络
    融合pool3和pool4的特征，上采样8倍
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super(FCN8s, self).__init__()

        # Backbone
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        features = list(vgg.features.children())

        self.features_block1 = nn.Sequential(*features[:5])
        self.features_block2 = nn.Sequential(*features[5:10])
        self.features_block3 = nn.Sequential(*features[10:17])
        self.features_block4 = nn.Sequential(*features[17:24])
        self.features_block5 = nn.Sequential(*features[24:31])

        # 卷积化分类器
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # 融合pool4的分支
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        # 融合pool3的分支
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        # 上采样层
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc6, self.fc7, self.score_fr, self.score_pool4, self.score_pool3]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 双线性初始化上采样层
        for upscore in [self.upscore2, self.upscore_pool4, self.upscore8]:
            c = upscore.weight.shape[0]
            k = upscore.kernel_size[0]
            if k == 4:
                bilinear_kernel = self._bilinear_kernel(c, c, 4)
                upscore.weight.data.copy_(bilinear_kernel)
            elif k == 16:
                bilinear_kernel = self._bilinear_kernel(c, c, 16)
                upscore.weight.data.copy_(bilinear_kernel)

    def _bilinear_kernel(self, in_c, out_c, k):
        factor = (k + 1) // 2
        center = factor - 1 if k % 2 == 1 else factor - 0.5
        og = (torch.arange(k).float() - center) / factor
        og = og.view(1, -1).repeat(k, 1)
        kernel = (1 - og.abs()) * (1 - og.T.abs())
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, k, k).repeat(in_c, out_c, 1, 1)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        # Encoder
        x = self.features_block1(x)
        x = self.features_block2(x)
        pool3 = self.features_block3(x)      # (B, 256, H/8, W/8)
        pool4 = self.features_block4(pool3)   # (B, 512, H/16, W/16)
        pool5 = self.features_block5(pool4)   # (B, 512, H/32, W/32)

        # FCN-32s 分支
        x = self.fc6(pool5)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        score32 = self.score_fr(x)            # (B, C, H/32, W/32)

        # 2x up -> 融合pool4
        upscore2 = self.upscore2(score32)     # (B, C, H/16, W/16)
        score_pool4 = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score_pool4

        # 2x up -> 融合pool3
        upscore_pool4 = self.upscore_pool4(fuse_pool4)  # (B, C, H/8, W/8)
        score_pool3 = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score_pool3

        # 8x up
        x = self.upscore8(fuse_pool3)
        x = x[:, :, :input_h, :input_w]
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)
