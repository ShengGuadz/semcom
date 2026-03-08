"""
sender_ptq.py
DeepJSCC 发送端模块 (PTQ Int8 量化版)
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import set_seed
import glob
from PIL import Image
from pytorch_msssim import ssim
from quant_model import QuantizableDeepJSCC
from quant_model import create_int8_model_structure

class ImageDataset(torch.utils.data.Dataset):
    """图像数据集加载器 (与原版保持一致)"""

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


class ConfigSenderPTQ:
    """发送端配置类 (PTQ版)"""
    # 基础配置
    seed = 1024
    # === 关键修改：强制使用 CPU ===
    device = torch.device("cpu")

    # 网络配置
    c = 4
    channel_type = 'AWGN'
    snr = 10

    # 数据配置
    batch_size = 1
    test_data_dir = "./data/kodak/"

    # === 关键修改：指向 Int8 模型路径 ===
    model_path = "deepjscc_int8.pth"

    # 保存路径
    sent_dir = './static/saved/sent_image/'

    # 网络配置
    host = '127.0.0.1'
    port = 60000


class SenderPTQ:
    """发送端类 (PTQ版)"""

    def __init__(self, config: ConfigSenderPTQ):
        self.config = config
        self.model = self._load_model()
        self.test_loader = self._get_dataloader()
        self._prepare_save_dirs()
        set_seed(config.seed)

    def _load_model(self):
        print(f"Loading PTQ Int8 weights from {self.config.model_path}")

        try:
            model = create_int8_model_structure(device='cpu')
        except Exception as e:
            raise RuntimeError(f"构建结构失败: {e}")

        if os.path.exists(self.config.model_path):
            state_dict = torch.load(self.config.model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"File {self.config.model_path} not found.")

        return model

    def _get_dataloader(self):
        """获取数据加载器"""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = ImageDataset(self.config.test_data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        return dataloader

    def _prepare_save_dirs(self):
        """准备保存目录"""
        os.makedirs(self.config.sent_dir, exist_ok=True)
        for file in os.listdir(self.config.sent_dir):
            file_path = os.path.join(self.config.sent_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _quantize(self, tensor, num_bits=8):
        """传输层面的量化 (保持不变)"""
        tensor_np = tensor.cpu().detach().numpy()
        min_val = tensor_np.min()
        max_val = tensor_np.max()

        if max_val - min_val < 1e-8:
            normalized = np.zeros_like(tensor_np)
        else:
            normalized = (tensor_np - min_val) / (max_val - min_val)

        if num_bits == 8:
            quantized = np.round(normalized * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")

        return quantized, min_val, max_val
def calculate_psnr(img1, img2):
    """计算PSNR (NumPy版)"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10: return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)

def calculate_ssim(img1, img2):
    """计算SSIM (NumPy输入 -> Tensor计算)"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return ssim(img1_tensor, img2_tensor, data_range=1.0).item()

if __name__ == '__main__':
    # 测试代码
    config = ConfigSenderPTQ()
    sender = SenderPTQ(config)
    print(f"PTQ Sender ready. Device: {config.device}")

    # 验证模型结构
    print(f"Model structure type: {type(sender.model)}")
    if hasattr(sender.model, 'encoder'):
        print("Encoder found in model.")