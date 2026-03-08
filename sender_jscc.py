"""
DeepJSCC 发送端模块
实现图像编码、量化和传输功能
"""
import os
import cv2
import torch
import numpy as np
import argparse
from model import DeepJSCC
from utils import set_seed, get_psnr, get_ssim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob


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


class ConfigSender:
    """发送端配置类"""
    # 基础配置
    seed = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 网络配置
    c = 4  # bottleneck通道数，控制压缩率
    channel_type = 'AWGN'  # 信道类型: 'AWGN' 或 'Rayleigh'
    snr =10  # 信噪比 (dB)
    
    # 数据配置
    # image_size = 128  # 图像大小
    batch_size = 1
    test_data_dir = "./data/kodak/"  # 测试数据路径
    # test_data_dir = "./data/junshi_test/"  # 测试数据路径
    # test_data_dir = "./data/military_test/"  # 测试数据路径
    # test_data_dir = "./data/0924/"  # 测试数据路径

    # 模型配置
    # model_path = "./out/checkpoints/CIFAR10_4_1.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"  # 预训练模型路径
    model_path = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"
    # model_path = "./out_military/checkpoints/MILITARY_4_4.0_0.08_AWGN_256/best_model.pth"
    # model_path = "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"
    # 保存路径
    sent_dir = './static/saved/sent_image/'
    
    # 网络配置
    host = '127.0.0.1'
    port = 60000


class Sender:
    """发送端类"""
    def __init__(self, config: ConfigSender):
        self.config = config
        self.model = self._load_model()
        self.test_loader = self._get_dataloader()
        self._prepare_save_dirs()
        
        # 设置随机种子
        set_seed(config.seed)
        
    def _load_model(self):
        """加载预训练模型"""
        model = DeepJSCC(c=self.config.c, channel_type=None, snr=None)
        
        if os.path.exists(self.config.model_path):
            print(f"Loading model from {self.config.model_path}")
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device))
        else:
            print(f"Warning: Model file {self.config.model_path} not found. Using random weights.")
        
        model = model.to(self.config.device)
        model.eval()
        return model
    
    def _get_dataloader(self):
        """获取数据加载器"""
        transform = transforms.Compose([
            # transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
        ])
        
        dataset = ImageDataset(self.config.test_data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        return dataloader
    
    def _prepare_save_dirs(self):
        """准备保存目录"""
        os.makedirs(self.config.sent_dir, exist_ok=True)
        # 清空目录
        for file in os.listdir(self.config.sent_dir):
            file_path = os.path.join(self.config.sent_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    def _quantize(self, tensor, num_bits=8):
        """
        量化特征到指定位数
        Args:
            tensor: 输入张量
            num_bits: 量化位数，默认8位
        Returns:
            quantized: 量化后的uint8数组
            min_val: 最小值
            max_val: 最大值
        """
        tensor_np = tensor.cpu().detach().numpy()
        
        # 归一化到 [0, 1]
        min_val = tensor_np.min()
        max_val = tensor_np.max()
        
        if max_val - min_val < 1e-8:
            # 避免除零
            normalized = np.zeros_like(tensor_np)
        else:
            normalized = (tensor_np - min_val) / (max_val - min_val)
        
        # 量化到 [0, 255]
        if num_bits == 8:
            quantized = np.round(normalized * 255).astype(np.uint8)
        elif num_bits == 16:
            quantized = np.round(normalized * 65535).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")
        
        return quantized, min_val, max_val
    
    def _add_gaussian_noise(self, image, snr_db):
        """
        向图像添加高斯噪声（用于对比）
        Args:
            image: 输入图像 (BGR格式)
            snr_db: 信噪比 (dB)
        Returns:
            noisy_image: 加噪后的图像
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 计算信号功率
        signal_pwr = np.mean(image_rgb ** 2)
        
        # 计算噪声功率
        noise_pwr = signal_pwr / (10 ** (snr_db / 10))
        
        # 生成噪声
        noise = np.random.normal(0, np.sqrt(noise_pwr), image_rgb.shape).astype(np.float32)
        
        # 添加噪声并裁剪到 [0, 255]
        noisy_image = np.clip(image_rgb + noise, 0, 255).astype(np.uint8)
        
        # 转回BGR格式
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        
        return noisy_image


if __name__ == '__main__':
    # 测试发送端
    config = ConfigSender()
    sender = Sender(config)
    
    print(f"Model loaded. Device: {config.device}")
    print(f"Test data directory: {config.test_data_dir}")
    print(f"Number of test images: {len(sender.test_loader.dataset)}")
    
    # 测试编码
    with torch.no_grad():
        for idx, image in enumerate(sender.test_loader):
            if idx >= 1:  # 只测试第一张
                break
            
            image = image.to(config.device)
            print(f"\nInput image shape: {image.shape}")
            
            # 编码
            feature = sender.model.encoder(image)
            print(f"Encoded feature shape: {feature.shape}")
            
            # 量化
            feature_quantized, min_val, max_val = sender._quantize(feature)
            print(f"Quantized feature shape: {feature_quantized.shape}")
            print(f"Feature value range: [{min_val:.4f}, {max_val:.4f}]")
            
            # 计算大小
            feature_size = feature_quantized.nbytes / 1024  # KB
            print(f"Quantized feature size: {feature_size:.2f} KB")
            
            # 计算压缩率
            original_size = image.shape[2] * image.shape[3] * 3  # pixels
            feature_elements = feature_quantized.size
            compression_ratio = original_size / feature_elements
            print(f"Compression ratio: {compression_ratio:.2f}×")
