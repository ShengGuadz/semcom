"""
sender_hailo.py
适配 Hailo 硬件加速的 DeepJSCC 发送端逻辑
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from utils import set_seed

# 引入 Hailo 推理接口 (确保 hailo_infer.py 在同级目录)
from hailo_infer import HailoInference, static_power_normalization


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image


class ConfigSender:
    # 基础配置
    seed = 1024
    device = torch.device("cpu")  # Hailo 不需要 GPU

    # 你的 HEF 模型路径
    hef_path = "deepjscc_encoder_512x768.hef"

    # 网络与数据配置
    test_data_dir = "./data/kodak/"
    sent_dir = './static/saved/sent_image/'
    # host = '127.0.0.1'  # 目标接收端IP
    host = '10.129.78.30'  # 目标接收端IP
    port = 60000

    # 模型参数 (用于归一化计算)
    c = 4
    snr = 10


class Sender:
    def __init__(self, config: ConfigSender):
        self.config = config
        self._prepare_save_dirs()
        set_seed(config.seed)

        # 1. 初始化 Hailo 引擎
        print(f"Initializing Hailo Engine: {self.config.hef_path}")
        self.hailo = HailoInference(self.config.hef_path)
        self.hailo.load_model()

        # 获取模型期望的输入尺寸 (H, W, C)
        self.input_shape = self.hailo.input_shape
        print(f"Model expects input: {self.input_shape}")

        # 2. 数据加载器
        self.test_loader = self._get_dataloader()

    def _get_dataloader(self):
        # 仅转为 Tensor，不缩放，保持原始分辨率以便我们手动处理旋转
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ImageDataset(self.config.test_data_dir, transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def _prepare_save_dirs(self):
        os.makedirs(self.config.sent_dir, exist_ok=True)

    def encode(self, image_tensor):
        """
        核心编码流程: 旋转 -> 推理 -> 归一化 -> 量化
        """
        # 1. 格式转换: PyTorch (NCHW) -> Numpy (H, W, C)
        # image_tensor: [1, 3, H, W], Range [0, 1]
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()

        h_img, w_img = img_np.shape[0], img_np.shape[1]
        h_model, w_model = self.input_shape[0], self.input_shape[1]

        rotated = 0
        input_data = img_np

        # 2. 自动旋转/缩放逻辑
        if h_img == h_model and w_img == w_model:
            pass  # 完美匹配
        elif h_img == w_model and w_img == h_model:
            # 宽高颠倒（例如图片是 768x512，模型要 512x768）
            print(f"Auto-rotating image from {h_img}x{w_img} to {h_model}x{w_model}")
            # 转置: (H, W, C) -> (W, H, C)
            input_data = np.transpose(img_np, (1, 0, 2))
            rotated = 1
        else:
            print(f"Warning: Resize {h_img}x{w_img} -> {h_model}x{w_model}")
            input_data = cv2.resize(img_np, (w_model, h_model))

        # 3. Hailo 推理
        # 输入必须是 float32
        input_data = input_data.astype(np.float32)
        feature = self.hailo.infer(input_data)  # 输出通常是 (1, H_feat, W_feat, C_feat)

        # 4. 功率归一化 (Power Normalization)
        # 因为导出的 ONNX 去掉了 norm 层，这里必须补上
        # k = H * W * C (特征的总维度，或者根据 DeepJSCC 论文是信道维度)
        # 这里使用你的 hailo_infer.py 中的静态归一化
        # 注意：hailo_infer.static_power_normalization 期望输入 (B, 2*c, H, W) 或类似
        # Hailo 输出通常是 NHWC。我们需要先转回 NCHW 方便计算，或者调整 norm 函数

        # 转换 Hailo 输出 (NHWC) -> (NCHW) 以匹配原来的数学逻辑
        feature_nchw = np.transpose(feature, (0, 3, 1, 2))

        # 计算 k (Encoder输出的总维度)
        k_val = np.prod(feature_nchw.shape[1:])

        # 执行归一化
        feature_norm = static_power_normalization(feature_nchw, k=k_val)

        # 5. 量化
        feature_quantized, min_val, max_val = self._quantize_numpy(feature_norm)

        return feature_quantized, min_val, max_val, rotated, input_data

    def _quantize_numpy(self, data, num_bits=8):
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val < 1e-8:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - min_val) / (max_val - min_val)

        if num_bits == 8:
            quantized = np.round(normalized * 255).astype(np.uint8)
        return quantized, float(min_val), float(max_val)

    def close(self):
        if self.hailo:
            self.hailo.close()