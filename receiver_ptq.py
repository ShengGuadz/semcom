"""
receiver_ptq.py
DeepJSCC 接收端模块 (PTQ Int8 量化版)
"""
import os
import cv2
import torch
import numpy as np
from utils import set_seed, get_psnr, get_ssim
from pytorch_msssim import ssim
from quant_model import QuantizableDeepJSCC
from quant_model import create_int8_model_structure

class ConfigReceiverPTQ:
    """接收端配置类 (PTQ版)"""
    # 基础配置
    seed = 1024
    # === 关键修改：强制使用 CPU ===
    device = torch.device("cpu")

    # 网络配置
    c = 4
    channel_type = 'AWGN'
    snr = 10

    # 数据配置
    image_size = 128

    # === 关键修改：指向 Int8 模型路径 ===
    model_path = "deepjscc_int8.pth"

    # 保存路径
    save_dir_semantic = './static/saved/semantic_communication/'
    save_dir_traditional = './static/saved/traditional_communication/'

    # 网络配置
    host = '0.0.0.0'
    port = 60000


class ReceiverPTQ:
    """接收端类 (PTQ版)"""

    def __init__(self, config: ConfigReceiverPTQ):
        self.config = config
        self.model = self._load_model()
        self._prepare_save_dirs()
        set_seed(config.seed)

    def _load_model(self):
        """加载量化模型 (State Dict 方式)"""
        print(f"Loading PTQ Int8 weights from {self.config.model_path}")

        # 1. 先构建一个空的 Int8 模型结构
        try:
            model = create_int8_model_structure(device='cpu')
        except Exception as e:
            raise RuntimeError(f"构建量化模型结构失败: {e}")

        # 2. 加载权重字典
        if os.path.exists(self.config.model_path):
            try:
                # 加载权重字典
                state_dict = torch.load(self.config.model_path, map_location='cpu')
                # 将权重填入模型
                model.load_state_dict(state_dict)
                print("Weights loaded successfully.")
            except Exception as e:
                # 如果这里报错，说明生成的 pth 不是 state_dict，请确认是否运行了新的 quantize_deepjscc.py
                raise RuntimeError(f"加载权重失败: {e}")
        else:
            raise FileNotFoundError(f"Model file {self.config.model_path} not found.")

        return model

    def _prepare_save_dirs(self):
        """准备保存目录"""
        for save_dir in [self.config.save_dir_semantic, self.config.save_dir_traditional]:
            os.makedirs(save_dir, exist_ok=True)
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _dequantize(self, quantized, min_val, max_val, target_shape):
        """反量化特征 (传输层面)"""
        if quantized.dtype == np.uint8:
            normalized = quantized.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported dtype: {quantized.dtype}")

        dequantized = normalized * (max_val - min_val) + min_val
        dequantized = dequantized.reshape(target_shape)
        tensor = torch.from_numpy(dequantized).float().to(self.config.device)
        return tensor

    def _simulate_bit_errors(self, data_bytes: bytes, snr_db: float, bandwidth: float = 100e6) -> bytes:
        """模拟误码 (保持不变)"""
        import math
        snr_linear = 10 ** (snr_db / 10)
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))
        ber = min(ber, 0.1)

        data_array = np.frombuffer(data_bytes, dtype=np.uint8).copy()
        total_bits = len(data_array) * 8
        num_errors = int(total_bits * ber)

        if num_errors > 0:
            error_positions = np.random.choice(total_bits, num_errors, replace=False)
            for pos in error_positions:
                byte_idx = pos // 8
                bit_idx = pos % 8
                data_array[byte_idx] ^= (1 << bit_idx)

        return data_array.tobytes()

    def _add_gaussian_noise(self, image, snr_db):
        """传统通信对比用"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        signal_pwr = np.mean(image_rgb ** 2)
        noise_pwr = signal_pwr / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_pwr), image_rgb.shape).astype(np.float32)
        noisy_image = np.clip(image_rgb + noise, 0, 255).astype(np.uint8)
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        return noisy_image

    def _decode(self, data):
        """解码"""
        feature_quantized, min_val, max_val, feature_shape = data['feature']
        original_image_data = data['image']

        original_image = cv2.imdecode(np.frombuffer(original_image_data, np.uint8), cv2.IMREAD_COLOR)
        feature = self._dequantize(feature_quantized, min_val, max_val, feature_shape)

        with torch.no_grad():
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)

            # === 这里调用量化模型的 decoder ===
            # feature 是 float，QuantizableDecoder 会自动处理输入量化
            recon_tensor = self.model.decoder(feature)

            recon_image = recon_tensor.squeeze(0).cpu().numpy()
            recon_image = np.transpose(recon_image, (1, 2, 0))
            recon_image = (recon_image * 255).clip(0, 255).astype(np.uint8)
            recon_image = cv2.cvtColor(recon_image, cv2.COLOR_RGB2BGR)

        traditional_image_size = len(original_image_data) // 1024
        return recon_image, original_image, feature_quantized, traditional_image_size
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