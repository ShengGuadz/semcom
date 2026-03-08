# -*- coding: utf-8 -*-
"""
DeepJSCC-CSI 发送端模块
实现图像编码、量化和传输功能（仅做“发送端”特征提取与量化）
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

from model_csi import DeepJSCCWithCSIFeedback
from utils_csi import set_seed  # 如需 PSNR/SSIM 可从 utils_csi 导入


# ==========================
# 数据集定义
# ==========================

class ImageDataset(Dataset):
    """图像数据集加载器"""

    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# ==========================
# 配置类
# ==========================

class ConfigSenderCSI:
    """JSCC-CSI 发送端配置类"""
    # 基础配置
    seed = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络配置
    # c = 8  # bottleneck 通道数，需与接收端一致
    c = 4  # bottleneck 通道数，需与接收端一致
    channel_type = 'AWGN'  # 训练时用的信道类型
    snr = 10  # 训练时用的 SNR (dB)
    feedback_bits = 32  # CSI 反馈比特数，需与训练保持一致

    # 数据配置
    batch_size = 1
    test_data_dir = "./data/kodak/"  # 测试数据路径

    # 模型配置（请改成你实际训练好的 JSCC-CSI 模型路径）
    # model_path = "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_fb32_22h33m02s_on_Nov_28_2025/epoch_405.pkl"
    model_path = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_fb32_10h59m40s_on_Nov_29_2025/epoch_447.pkl"

    # 保存路径
    sent_dir = './static/saved_csi/sent_image/'

    host = '127.0.0.1'  # 发送端连接的接收端地址（本机）
    port = 61000  # 必须与接收端监听端口一致


# ==========================
# 发送端主体
# ==========================

class SenderCSI:
    """JSCC-CSI 发送端类：负责加载模型、读取数据并输出量化特征"""

    def __init__(self, config: ConfigSenderCSI):
        self.config = config
        self.model = self._load_model()
        self.test_loader = self._get_dataloader()
        self._prepare_save_dirs()

        # 设置随机种子
        set_seed(config.seed)

    # --------- 模型与数据 ---------
    def _load_model(self):
        """加载预训练 JSCC-CSI 模型"""
        model = DeepJSCCWithCSIFeedback(
            c=self.config.c,
            channel_type=self.config.channel_type,
            snr=self.config.snr,
            feedback_bits=self.config.feedback_bits,
            use_csi_aware_decoder=True
        )

        if os.path.exists(self.config.model_path):
            print(f"[Sender-CSI] Loading model from {self.config.model_path}")
            state = torch.load(self.config.model_path, map_location=self.config.device)
            model.load_state_dict(state)
        else:
            print(f"[Sender-CSI] WARNING: Model file {self.config.model_path} not found. Using random weights.")

        model = model.to(self.config.device)
        model.eval()
        return model

    def _get_dataloader(self):
        """获取测试数据 DataLoader"""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = ImageDataset(self.config.test_data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        return dataloader

    def _prepare_save_dirs(self):
        """准备发送端图像保存目录"""
        os.makedirs(self.config.sent_dir, exist_ok=True)
        # 清空目录
        for file in os.listdir(self.config.sent_dir):
            file_path = os.path.join(self.config.sent_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # --------- 量化函数（保持与原 DeepJSCC 一致）---------
    def _quantize(self, tensor, num_bits=8):
        """
        量化特征到指定位数
        Args:
            tensor: 输入张量 (B, C, H, W)
            num_bits: 量化位数，默认 8 位
        Returns:
            quantized: 量化后的 uint8/uint16 数组
            min_val: 原始最小值
            max_val: 原始最大值
        """
        tensor_np = tensor.cpu().detach().numpy()

        # 归一化到 [0, 1]
        min_val = tensor_np.min()
        max_val = tensor_np.max()

        if max_val - min_val < 1e-8:
            normalized = np.zeros_like(tensor_np)
        else:
            normalized = (tensor_np - min_val) / (max_val - min_val)

        # 量化
        if num_bits == 8:
            quantized = np.round(normalized * 255).astype(np.uint8)
        elif num_bits == 16:
            quantized = np.round(normalized * 65535).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")

        return quantized, float(min_val), float(max_val)

    # --------- 对单张图像进行编码与量化（供 Flask / socket 调用）---------
    def encode_one_image(self, input_image: torch.Tensor, num_bits: int = 8):
        """
        对一张输入图像进行 JSCC-CSI 编码并量化特征
        Args:
            input_image: 图像张量 (1, 3, H, W)，已在正确设备上
            num_bits: 量化位数
        Returns:
            feature_quantized: 量化后的特征 (numpy 数组)
            min_val, max_val: 量化时使用的最小/最大值
            feature_shape: 原始特征形状
            image_bgr: 原始图像 (H, W, 3) BGR，用于 JPEG 对比或保存
        """
        B, C, H, W = input_image.shape

        with torch.no_grad():
            # 1) 编码：只用 encoder，不走内部信道与 CSI
            feature = self.model.encoder(input_image)
            feature_shape = feature.shape

            # 2) 量化
            feature_quantized, min_val, max_val = self._quantize(feature, num_bits=num_bits)

        # 3) 准备原始图像 (BGR) 用于保存 / 发送 / JPEG 对比
        image_np = input_image[0].cpu().numpy()  # (C, H, W)
        image_np = (image_np * 255).clip(0, 255).astype('uint8')
        image_np = image_np.transpose(1, 2, 0)  # (H, W, C)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return feature_quantized, min_val, max_val, feature_shape, image_bgr
