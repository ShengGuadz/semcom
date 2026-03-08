# -*- coding: utf-8 -*-
"""
DeepJSCC-CSI 接收端模块
实现特征接收、反量化、合成 CSI 并进行 CSI-aware 解码
"""

import os
import cv2
import torch
import numpy as np

from model_csi import DeepJSCCWithCSIFeedback
from utils_csi import set_seed


class ConfigReceiverCSI:
    """JSCC-CSI 接收端配置类"""
    # 基础配置
    seed = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络配置
    # c = 8  # bottleneck 通道数，需与发送端一致
    c = 4  # bottleneck 通道数，需与发送端一致
    channel_type = 'AWGN'  # 训练时信道类型
    snr = 10  # 用于合成 CSI 的 SNR (dB)
    feedback_bits = 32  # CSI 反馈比特数，需与训练一致
    csi_dim = 3  # AWGN 下 CSI 维度：[signal_power, noise_power, snr_db]

    # 模型配置（请改成你实际训练好的 JSCC-CSI 模型路径）
    # model_path = "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_fb32_22h33m02s_on_Nov_28_2025/epoch_405.pkl"
    model_path = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_fb32_10h59m40s_on_Nov_29_2025/epoch_447.pkl"

    # 保存路径
    save_dir_semantic = './static/saved_csi/semantic_communication/'
    save_dir_traditional = './static/saved_csi/traditional_communication/'

    # 网络配置（供 socket 使用）
    host = '0.0.0.0'
    port = 61000


class ReceiverCSI:
    """JSCC-CSI 接收端类"""

    def __init__(self, config: ConfigReceiverCSI):
        self.config = config
        self.model = self._load_model()
        self._prepare_save_dirs()

        # 设置随机种子
        set_seed(config.seed)

    # --------- 模型与目录 ---------
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
            print(f"[Receiver-CSI] Loading model from {self.config.model_path}")
            state = torch.load(self.config.model_path, map_location=self.config.device)
            model.load_state_dict(state)
        else:
            print(f"[Receiver-CSI] WARNING: Model file {self.config.model_path} not found. Using random weights.")

        model = model.to(self.config.device)
        model.eval()
        return model

    def _prepare_save_dirs(self):
        """准备保存目录"""
        for save_dir in [self.config.save_dir_semantic, self.config.save_dir_traditional]:
            os.makedirs(save_dir, exist_ok=True)
            # 清空目录
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    # --------- 反量化 ---------
    def _dequantize(self, quantized, min_val, max_val, target_shape):
        """
        反量化特征
        Args:
            quantized: 量化后的数组 (uint8 / uint16)
            min_val: 量化时的最小值
            max_val: 量化时的最大值
            target_shape: 原始特征形状
        Returns:
            tensor: 反量化后的张量 (在正确设备上)
        """
        if quantized.dtype == np.uint8:
            normalized = quantized.astype(np.float32) / 255.0
        elif quantized.dtype == np.uint16:
            normalized = quantized.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"Unsupported dtype: {quantized.dtype}")

        dequantized = normalized * (max_val - min_val) + min_val
        dequantized = dequantized.reshape(target_shape)
        tensor = torch.from_numpy(dequantized).float().to(self.config.device)
        return tensor

    # --------- 可选：传统链路模拟（与原 receiver_deepjscc 保持一致）---------
    def _simulate_bit_errors(self, data_bytes: bytes, snr_db: float) -> bytes:
        """
        在 JPEG 字节流上基于 SNR 模拟比特翻转，简化 QPSK+AWGN BER 模型
        """
        import math

        snr_linear = 10 ** (snr_db / 10.0)
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))
        ber = min(ber, 0.1)  # 限制上界，避免极端情况

        print(f"[Receiver-CSI] 模拟 BER: {ber:.6f} (SNR: {snr_db} dB)")

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
        """
        像素域添加高斯噪声（供传统链路回退使用）
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        signal_pwr = np.mean(image_rgb ** 2)
        noise_pwr = signal_pwr / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_pwr), image_rgb.shape).astype(np.float32)
        noisy_image = np.clip(image_rgb + noise, 0, 255).astype(np.uint8)
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        return noisy_image

    # --------- 语义解码核心：利用 CSI-aware 解码器 ---------
    def _build_awgn_csi(self, z: torch.Tensor) -> torch.Tensor:
        """
        根据特征张量 z 和配置中的 SNR 合成 AWGN 风格 CSI 向量：
        csi = [signal_power, noise_power, snr_db]
        """
        if z.dim() < 2:
            raise ValueError("Feature tensor z must have at least 2 dimensions")

        snr_linear = 10 ** (self.config.snr / 10.0)
        dims = list(range(1, z.dim()))
        signal_power = z.pow(2).mean(dim=dims)  # (B,)
        noise_power = signal_power / snr_linear  # (B,)

        B = z.size(0)
        csi = torch.cat([
            signal_power.view(B, 1),
            noise_power.view(B, 1),
            torch.ones(B, 1, device=z.device) * self.config.snr
        ], dim=1)  # (B, 3)
        return csi

    def decode(self, data):
        """
        解码接收到的数据（给 socket / Flask 调用）
        Args:
            data: 字典，包含：
                - 'feature': [feature_quantized, min_val, max_val, feature_shape]
                - 'image': JPEG 字节流（用于传统链路）
        Returns:
            semantic_image: 语义重建图像 (BGR)
            original_image: 原始图像 (BGR)
            feature_quantized: 原始量化特征（numpy）
            traditional_image_size: 传统 JPEG 大小 (KB)
        """
        # ---------- 解析数据 ----------
        feature_quantized, min_val, max_val, feature_shape = data['feature']
        original_image_data = data['image']

        # 解码原始 JPEG 图像
        original_image = cv2.imdecode(
            np.frombuffer(original_image_data, np.uint8),
            cv2.IMREAD_COLOR
        )

        # ---------- 反量化特征 ----------
        feature = self._dequantize(feature_quantized, min_val, max_val, feature_shape)

        with torch.no_grad():
            # 添加 batch 维
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)

            # ---------- 构造 CSI 并通过反馈模块 ----------
            csi_original = self._build_awgn_csi(feature)
            _, csi_reconstructed = self.model.csi_feedback(csi_original)

            # ---------- CSI-aware 解码 ----------
            recon_tensor = self.model.decoder(feature, csi_reconstructed)

            # 转 numpy 图像
            recon_image = recon_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
            recon_image = np.transpose(recon_image, (1, 2, 0))  # (H, W, C)
            recon_image = (recon_image * 255).clip(0, 255).astype(np.uint8)
            recon_image = cv2.cvtColor(recon_image, cv2.COLOR_RGB2BGR)

        traditional_image_size = len(original_image_data) // 1024  # KB
        return recon_image, original_image, feature_quantized, traditional_image_size
