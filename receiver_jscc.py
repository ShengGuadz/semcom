"""
DeepJSCC 接收端模块
实现特征接收、反量化和图像解码功能
"""
import os
import cv2
import torch
import numpy as np
from model import DeepJSCC
from utils import set_seed, get_psnr, get_ssim
import glob


class ConfigReceiver:
    """接收端配置类"""
    # 基础配置
    seed = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络配置
    c = 4  # bottleneck通道数，必须与发送端一致
    channel_type = 'AWGN'  # 信道类型
    snr = 10  # 信噪比 (dB)

    # 数据配置
    image_size = 128  # 图像大小

    # 模型配置
    # model_path = "./out/checkpoints/CIFAR10_4_1.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"  # 预训练模型路径
    model_path = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"  # 预训练模型路径
    # model_path = "./out_military/checkpoints/MILITARY_4_4.0_0.08_AWGN_256/best_model.pth"
    # model_path = "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"  # 预训练模型路径

    # 保存路径
    save_dir_semantic = './static/saved/semantic_communication/'
    save_dir_traditional = './static/saved/traditional_communication/'

    # 网络配置
    host = '0.0.0.0'
    port = 60000


class Receiver:
    """接收端类"""

    def __init__(self, config: ConfigReceiver):
        self.config = config
        self.model = self._load_model()
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

    def _prepare_save_dirs(self):
        """准备保存目录"""
        for save_dir in [self.config.save_dir_semantic, self.config.save_dir_traditional]:
            os.makedirs(save_dir, exist_ok=True)
            # 清空目录
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _dequantize(self, quantized, min_val, max_val, target_shape):
        """
        反量化特征
        Args:
            quantized: 量化后的数组
            min_val: 最小值
            max_val: 最大值
            target_shape: 目标形状
        Returns:
            tensor: 反量化后的张量
        """
        # 归一化到 [0, 1]
        if quantized.dtype == np.uint8:
            normalized = quantized.astype(np.float32) / 255.0
        elif quantized.dtype == np.uint16:
            normalized = quantized.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"Unsupported dtype: {quantized.dtype}")

        # 恢复到原始范围
        dequantized = normalized * (max_val - min_val) + min_val

        # 重塑形状
        dequantized = dequantized.reshape(target_shape)

        # 转换为torch张量
        tensor = torch.from_numpy(dequantized).float().to(self.config.device)

        return tensor

    def _simulate_bit_errors(self, data_bytes: bytes, snr_db: float, bandwidth: float = 100e6) -> bytes:
        """
        基于信道SNR模拟字节流上的比特错误。
        简化模型：将SNR转换为QPSK的BER，然后随机翻转比特。
        """
        import math

        # QPSK在AWGN下的简化BER计算
        snr_linear = 10 ** (snr_db / 10)
        ber = 0.5 * math.erfc(math.sqrt(snr_linear))

        # 限制BER在合理范围
        ber = min(ber, 0.1)

        print(f"模拟BER: {ber:.6f} (SNR: {snr_db}dB)")

        # 将字节转换为numpy数组进行比特操作
        # ✅ FIX: 添加 .copy() 使数组可写
        data_array = np.frombuffer(data_bytes, dtype=np.uint8).copy()

        # 总比特数
        total_bits = len(data_array) * 8

        # 预期误码数量
        num_errors = int(total_bits * ber)

        if num_errors > 0:
            # 随机选择要翻转的比特位置
            error_positions = np.random.choice(total_bits, num_errors, replace=False)

            # 翻转比特
            for pos in error_positions:
                byte_idx = pos // 8
                bit_idx = pos % 8
                data_array[byte_idx] ^= (1 << bit_idx)

        return data_array.tobytes()

    def _add_gaussian_noise(self, image, snr_db):
        """
        向图像添加高斯噪声（用于传统通信对比）
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


    def _decode(self, data):
            """
            解码接收到的数据
            Args:
                data: 包含量化特征、元数据和JPEG数据的字典
            Returns:
                semantic_image: 语义通信重建图像
                jpeg_image_clean: 发送端传来的未经信道干扰的JPEG图像 (用于计算压缩率或作为参考，但不是Ground Truth)
                feature_quantized: 量化特征
                traditional_image_size: 传统方法图像大小
            """
            # 提取数据
            feature_quantized, min_val, max_val, feature_shape = data['feature']
            jpeg_data_clean = data['image']  # 这里接收到的是JPEG字节流

            # 解码 "发送端压缩后的JPEG" (无信道噪声)
            jpeg_image_clean = cv2.imdecode(
                np.frombuffer(jpeg_data_clean, np.uint8),
                cv2.IMREAD_COLOR
            )

            # 反量化特征
            feature = self._dequantize(feature_quantized, min_val, max_val, feature_shape)
            # 更新模型内部的噪声方差参数
            self.model.change_channel(self.config.channel_type, self.config.snr)

            # 解码重建图像 (语义通信)
            with torch.no_grad():
                # 添加batch维度
                if feature.dim() == 3:
                    feature = feature.unsqueeze(0)
                    # 让特征经过模型内置的 channel 层进行 AWGN/Rayleigh 加噪
                if hasattr(self.model, 'channel') and self.model.channel is not None:
                    feature = self.model.channel(feature)

                # 解码
                recon_tensor = self.model.decoder(feature)

                # 转换为numpy数组
                recon_image = recon_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
                recon_image = np.transpose(recon_image, (1, 2, 0))  # (H, W, C)
                recon_image = (recon_image * 255).clip(0, 255).astype(np.uint8)

                # 转换为BGR格式
                recon_image = cv2.cvtColor(recon_image, cv2.COLOR_RGB2BGR)

            # 计算传统图像大小 (KB)
            traditional_image_size = len(jpeg_data_clean) // 1024

            return recon_image, jpeg_image_clean, feature_quantized, traditional_image_size


def calculate_psnr(img1, img2):
    """
    计算PSNR
    Args:
        img1: 图像1
        img2: 图像2
    Returns:
        psnr: PSNR值 (dB)
    """
    # 确保尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100.0

    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return psnr


def calculate_ssim(img1, img2):
    """
    计算SSIM
    Args:
        img1: 图像1
        img2: 图像2
    Returns:
        ssim: SSIM值
    """
    from pytorch_msssim import ssim

    # 确保尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 转换为tensor
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    ssim_value = ssim(img1_tensor, img2_tensor, data_range=1.0)

    return ssim_value.item()


if __name__ == '__main__':
    # 测试接收端
    config = ConfigReceiver()
    receiver = Receiver(config)

    print(f"Model loaded. Device: {config.device}")
    print(f"Receiver ready on {config.host}:{config.port}")

    # 测试解码
    # 创建模拟数据
    feature_shape = (1, 2 * config.c, config.image_size // 4, config.image_size // 4)
    feature = torch.randn(*feature_shape)

    # 量化
    feature_np = feature.numpy()
    min_val = feature_np.min()
    max_val = feature_np.max()
    normalized = (feature_np - min_val) / (max_val - min_val)
    quantized = np.round(normalized * 255).astype(np.uint8)

    # 测试图像
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    test_image_data = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()

    # 构造数据包
    data = {
        'feature': [quantized, min_val, max_val, feature_shape],
        'image': test_image_data,
    }

    # 解码
    semantic_image, original_image, _, _ = receiver._decode(data)

    print(f"\nDecoding test:")
    print(f"Semantic image shape: {semantic_image.shape}")
    print(f"Original image shape: {original_image.shape}")
