"""
receiver_deepjscc.py (修改版)
增加对 rotated 标志的处理
"""
import os
import cv2
import torch
import numpy as np
from model import DeepJSCC
from utils import set_seed, get_psnr, get_ssim  # 假设你有这些工具函数


class ConfigReceiver:
    seed = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    c = 4
    channel_type = 'AWGN'
    snr = 10

    # 你的 PyTorch 模型路径 (PC端)
    model_path = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"

    save_dir_semantic = './static/saved/semantic_communication/'
    save_dir_traditional = './static/saved/traditional_communication/'
    host = '0.0.0.0'
    port = 60000


class Receiver:
    def __init__(self, config: ConfigReceiver):
        self.config = config
        self.model = self._load_model()
        self._prepare_save_dirs()
        set_seed(config.seed)

    def _load_model(self):
        model = DeepJSCC(c=self.config.c, channel_type=None, snr=None)
        if os.path.exists(self.config.model_path):
            print(f"Loading Torch model: {self.config.model_path}")
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device))
        model = model.to(self.config.device)
        model.eval()
        return model

    def _prepare_save_dirs(self):
        os.makedirs(self.config.save_dir_semantic, exist_ok=True)
        os.makedirs(self.config.save_dir_traditional, exist_ok=True)

    def _dequantize(self, quantized, min_val, max_val):
        # 简单的反量化: uint8 -> float32 -> 原始范围
        if quantized.dtype == np.uint8:
            norm = quantized.astype(np.float32) / 255.0
        else:
            norm = quantized.astype(np.float32)

        dequantized = norm * (max_val - min_val) + min_val
        return torch.from_numpy(dequantized).float().to(self.config.device)

    def _decode(self, data):
        """
        解码函数
        Returns:
            recon_bgr: 语义重建图 (已恢复旋转)
            jpeg_clean: 发送端传来的JPEG原图 (无信道噪声，已恢复旋转)
            feature_q: 量化特征
            jpeg_size_kb: JPEG大小
        """
        # 1. 解包
        feature_q = data['feature'][0]
        min_v = data['feature'][1]
        max_v = data['feature'][2]
        rotated = data.get('rotated', 0)

        # 2. 反量化 & 解码
        feature_tensor = self._dequantize(feature_q, min_v, max_v)
        if feature_tensor.dim() == 3:
            feature_tensor = feature_tensor.unsqueeze(0)

        with torch.no_grad():
            recon = self.model.decoder(feature_tensor)
            recon_np = recon.squeeze(0).cpu().numpy()
            recon_np = np.transpose(recon_np, (1, 2, 0))
            recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
            recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)

        # 3. 语义图：恢复旋转
        if rotated == 1:
            # 发送端: [H,W] -> [W,H]
            # 接收端: [W,H] -> [H,W]
            recon_bgr = np.transpose(recon_bgr, (1, 0, 2))

        # 4. JPEG图：解码并恢复旋转
        # 注意：这里的 jpeg_clean 只是发送端压缩后的样子，不是真正的 Ground Truth
        jpeg_data = data['image']
        jpeg_clean = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)

        if rotated == 1 and jpeg_clean is not None:
            # 如果发送端为了适应模型旋转了图片，发过来的JPEG也是旋转过的
            # 我们需要把它转回来，才能和本地数据集对比
            if jpeg_clean.shape != recon_bgr.shape:
                jpeg_clean = np.transpose(jpeg_clean, (1, 0, 2))

        return recon_bgr, jpeg_clean, feature_q, len(jpeg_data) // 1024

    # ... 保留 _simulate_bit_errors 等其他辅助函数 ...
    def _simulate_bit_errors(self, data_bytes, snr_db, bandwidth=100e6):
        # (直接复用你提供的 receiver_deepjscc.py 中的代码)
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