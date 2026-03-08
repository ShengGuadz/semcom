# -*- coding: utf-8 -*-
"""
Kodak 数据集测试脚本 - 包含 JPEG+LDPC+QAM 对比曲线
支持 DeepJSCC、DeepJSCC+CSI 和传统 JPEG+LDPC+QAM 方案
"""

import os
from glob import glob
import io
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pytorch_msssim import ms_ssim

# =========================
# 配置区
# =========================
CONFIG = {
    "kodak_root": "./dataset/Kodak",
    "batch_size": 1,
    "device": "cuda:0",
    "channel": "AWGN",
    "ratio": "1/6",

    # SNR 扫描范围
    "snr_min": -5,
    "snr_max": 25,
    "snr_step": 3,

    # 结果输出目录
    "out_dir": "./eval_results",

    # JPEG+LDPC+QAM 配置
    "jpeg_ldpc_configs": [
        {
            "label": "JPEG + 1/2 LDPC + 16QAM",
            "jpeg_quality": 15,  # JPEG质量
            "ldpc_rate": 0.5,  # LDPC码率
            "qam_order": 16,  # QAM阶数
            "color": "magenta",
            "marker": "o",
            "linestyle": "-"
        },
    ],

    # 神经网络模型列表
    "models": [
        {
            "path": "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_fb32_22h33m02s_on_Nov_28_2025/epoch_405.pkl",
            "type": "csi",
            "label": "JSCC+CM SNR$_{train}$=1dB",
            "color": "red",
            "marker": "v",
            "linestyle": "-"
        },
        {
            "path": "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl",
            "type": "baseline",
            "label": "JSCC SNR$_{train}$=1dB",
            "color": "red",
            "marker": "s",
            "linestyle": "--"
        },
        {
            "path": "./out/checkpoints/CIFAR10_8_4.0_0.17_AWGN_fb32_03h00m05s_on_Nov_28_2025/epoch_990.pkl",
            "type": "csi",
            "label": "JSCC+CM SNR$_{train}$=4dB",
            "color": "blue",
            "marker": "v",
            "linestyle": "-"
        },
        {
            "path": "./out/checkpoints/CIFAR10_8_4.0_0.17_AWGN_22h13m52s_on_Jun_07_2024/epoch_999.pkl",
            "type": "baseline",
            "label": "JSCC SNR$_{train}$=4dB",
            "color": "blue",
            "marker": "s",
            "linestyle": "--"
        },
    ]
}


# =========================
# JPEG + LDPC + QAM 传统方案
# =========================
class JPEGLDPCQAMCodec:
    """JPEG + LDPC + QAM 传统编解码方案"""

    def __init__(self, jpeg_quality=15, ldpc_rate=0.5, qam_order=16):
        self.jpeg_quality = jpeg_quality
        self.ldpc_rate = ldpc_rate
        self.qam_order = qam_order
        self.bits_per_symbol = int(np.log2(qam_order))

        # 构建QAM星座图
        m = int(np.sqrt(qam_order))
        x = np.arange(m) - (m - 1) / 2
        y = np.arange(m) - (m - 1) / 2
        self.constellation = np.array([complex(a, b) for a in x for b in y])

        # 归一化星座图功率
        avg_power = np.mean(np.abs(self.constellation) ** 2)
        self.constellation = self.constellation / np.sqrt(avg_power)

    def jpeg_encode(self, img_tensor):
        """将tensor图像压缩为JPEG字节流"""
        # tensor to PIL
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # 压缩为JPEG
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=self.jpeg_quality)
        jpeg_bytes = buffer.getvalue()

        return jpeg_bytes, img_pil.size

    def jpeg_decode(self, jpeg_bytes, original_size=None):
        """从JPEG字节流解码图像"""
        try:
            buffer = io.BytesIO(jpeg_bytes)
            img_pil = Image.open(buffer).convert('RGB')
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return img_tensor
        except:
            return None

    def bytes_to_bits(self, data_bytes):
        """字节流转比特流"""
        bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        return bits.astype(np.float32)

    def bits_to_bytes(self, bits):
        """比特流转字节流"""
        bits = np.clip(np.round(bits), 0, 1).astype(np.uint8)
        # 确保比特数是8的倍数
        pad_len = (8 - len(bits) % 8) % 8
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        bytes_data = np.packbits(bits)
        return bytes_data.tobytes()

    def ldpc_encode(self, bits):
        """简化的LDPC编码 (重复码模拟)"""
        # 使用简单的重复码来模拟LDPC的冗余
        # 码率 = 1/2 意味着每个信息比特对应2个编码比特
        repeat_factor = int(1 / self.ldpc_rate)
        encoded = np.repeat(bits, repeat_factor)
        return encoded

    def ldpc_decode(self, llr, original_length):
        """简化的LDPC解码 (软判决)"""
        repeat_factor = int(1 / self.ldpc_rate)

        # 将LLR重塑并求和进行软判决
        num_info_bits = len(llr) // repeat_factor
        llr_reshaped = llr[:num_info_bits * repeat_factor].reshape(num_info_bits, repeat_factor)

        # 软判决：对重复的LLR求和
        combined_llr = np.sum(llr_reshaped, axis=1)

        # 硬判决
        decoded_bits = (combined_llr < 0).astype(np.float32)

        return decoded_bits[:original_length]

    def qam_modulate(self, bits):
        """QAM调制"""
        # 填充到符号边界
        pad_len = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len)])

        # 重塑为符号组
        bit_groups = bits.reshape(-1, self.bits_per_symbol).astype(np.uint8)

        # 比特组转索引 (Gray码可选)
        indices = np.zeros(len(bit_groups), dtype=np.int32)
        for i in range(self.bits_per_symbol):
            indices += bit_groups[:, i].astype(np.int32) * (2 ** i)

        # 映射到星座点
        symbols = self.constellation[indices]
        return symbols

    def qam_demodulate_soft(self, received, noise_var):
        """QAM软解调，输出LLR"""
        num_symbols = len(received)
        llr = np.zeros(num_symbols * self.bits_per_symbol)

        for sym_idx in range(num_symbols):
            r = received[sym_idx]

            for bit_idx in range(self.bits_per_symbol):
                # 找出该比特为0和为1的星座点
                bit_0_indices = [i for i in range(len(self.constellation))
                                 if not (i >> bit_idx) & 1]
                bit_1_indices = [i for i in range(len(self.constellation))
                                 if (i >> bit_idx) & 1]

                # 计算到各星座点的距离
                dist_0 = np.min([np.abs(r - self.constellation[i]) ** 2 for i in bit_0_indices])
                dist_1 = np.min([np.abs(r - self.constellation[i]) ** 2 for i in bit_1_indices])

                # LLR = ln(P(bit=0)/P(bit=1)) ≈ (dist_1 - dist_0) / noise_var
                llr[sym_idx * self.bits_per_symbol + bit_idx] = (dist_1 - dist_0) / (noise_var + 1e-10)

        return llr

    def awgn_channel(self, symbols, snr_db):
        """AWGN信道"""
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(symbols) ** 2)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)

        noise = noise_std * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        received = symbols + noise

        return received, noise_power

    def transmit(self, img_tensor, snr_db):
        """完整的传输流程"""
        # 1. JPEG编码
        jpeg_bytes, img_size = self.jpeg_encode(img_tensor)
        original_length = len(jpeg_bytes) * 8

        # 2. 字节转比特
        bits = self.bytes_to_bits(jpeg_bytes)

        # 3. LDPC编码
        encoded_bits = self.ldpc_encode(bits)

        # 4. QAM调制
        symbols = self.qam_modulate(encoded_bits)

        # 5. AWGN信道
        received, noise_var = self.awgn_channel(symbols, snr_db)

        # 6. QAM软解调
        llr = self.qam_demodulate_soft(received, noise_var)

        # 7. LDPC解码
        decoded_bits = self.ldpc_decode(llr, original_length)

        # 8. 比特转字节
        decoded_bytes = self.bits_to_bytes(decoded_bits)

        # 9. JPEG解码
        # 截取到原始长度
        decoded_bytes = decoded_bytes[:len(jpeg_bytes)]
        decoded_img = self.jpeg_decode(decoded_bytes, img_size)

        if decoded_img is None:
            # 解码失败，返回噪声图像
            return torch.rand_like(img_tensor) * 0.5

        return decoded_img

    def get_jpeg_only_psnr(self, img_tensor):
        """获取仅JPEG压缩的PSNR（无信道噪声时的上限）"""
        jpeg_bytes, _ = self.jpeg_encode(img_tensor)
        decoded_img = self.jpeg_decode(jpeg_bytes)

        if decoded_img is None:
            return 0.0

        mse = torch.mean((img_tensor - decoded_img) ** 2)
        if mse == 0:
            return 50.0
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()


def evaluate_jpeg_ldpc_qam(dataloader, snr_list, config, device):
    """评估 JPEG+LDPC+QAM 方案"""
    codec = JPEGLDPCQAMCodec(
        jpeg_quality=config['jpeg_quality'],
        ldpc_rate=config['ldpc_rate'],
        qam_order=config['qam_order']
    )

    results = {'snr': [], 'psnr': [], 'msssim': []}

    for snr in snr_list:
        psnr_list = []
        msssim_list = []

        for images, names in dataloader:
            images = images.to(device)

            # 传输
            outputs = codec.transmit(images, snr)
            outputs = outputs.to(device)

            # 计算PSNR
            mse = torch.mean((images - outputs) ** 2)
            if mse > 0:
                psnr_val = (10 * torch.log10(1.0 / mse)).item()
            else:
                psnr_val = 50.0
            psnr_list.append(psnr_val)

            # 计算MS-SSIM
            outputs_clamp = torch.clamp(outputs, 0.0, 1.0)
            images_clamp = torch.clamp(images, 0.0, 1.0)

            if images.shape[2] >= 160 and images.shape[3] >= 160:
                try:
                    msssim_val = ms_ssim(outputs_clamp, images_clamp, data_range=1.0, size_average=True).item()
                except:
                    msssim_val = 0.0
            else:
                msssim_val = 0.0
            msssim_list.append(msssim_val)

        avg_psnr = np.mean(psnr_list)
        avg_msssim = np.mean(msssim_list)

        results['snr'].append(snr)
        results['psnr'].append(avg_psnr)
        results['msssim'].append(avg_msssim)

        print(f"  SNR={snr:>3} dB | PSNR={avg_psnr:.3f} dB | MS-SSIM={avg_msssim:.4f}")

    return results


# =========================
# 模型组件 (与 test_c=0.16.py 保持一致)
# =========================
class Channel(nn.Module):
    """Channel model supporting AWGN and Rayleigh fading with CSI generation"""

    def __init__(self, channel_type='AWGN', snr=10):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr
        self._csi = None

    def forward(self, x, return_csi=False):
        if self.channel_type == 'AWGN':
            output, csi = self._awgn_channel(x)
        elif self.channel_type == 'Rayleigh':
            output, csi = self._rayleigh_channel(x)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

        self._csi = csi

        if return_csi:
            return output, csi
        return output

    def _awgn_channel(self, x):
        snr_linear = 10 ** (self.snr / 10.0)

        if x.dim() >= 2:
            dims = list(range(1, x.dim()))
            signal_power = torch.mean(x ** 2, dim=dims, keepdim=True)
        else:
            signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)

        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std
        output = x + noise

        batch_size = x.size(0)
        csi = torch.cat([
            signal_power.view(batch_size, 1),
            noise_power.view(batch_size, 1),
            torch.ones(batch_size, 1, device=x.device) * self.snr
        ], dim=1)

        return output, csi

    def _rayleigh_channel(self, x):
        snr_linear = 10 ** (self.snr / 10.0)
        batch_size = x.size(0)

        h_real = torch.randn(batch_size, 1, 1, 1, device=x.device) * np.sqrt(0.5)
        h_imag = torch.randn(batch_size, 1, 1, 1, device=x.device) * np.sqrt(0.5)
        h_mag_sq = h_real ** 2 + h_imag ** 2
        h_mag = torch.sqrt(h_mag_sq)
        faded_signal = x * h_mag

        signal_power = torch.mean(faded_signal ** 2, dim=(1, 2, 3), keepdim=True)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std
        output = faded_signal + noise

        csi = torch.cat([
            h_real.view(batch_size, 1),
            h_imag.view(batch_size, 1),
            h_mag_sq.view(batch_size, 1),
            signal_power.view(batch_size, 1),
            noise_power.view(batch_size, 1),
            torch.ones(batch_size, 1, device=x.device) * self.snr
        ], dim=1)

        return output, csi

    def get_csi_dim(self):
        if self.channel_type == 'AWGN':
            return 3
        elif self.channel_type == 'Rayleigh':
            return 6
        return 0


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.prelu(self.conv(x))


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activate=None, padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate if activate is not None else nn.PReLU()

        if isinstance(self.activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        return self.activate(self.transconv(x))


class _Encoder(nn.Module):
    """修复后的 Encoder，确保输出始终是 4D 张量"""

    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        self.P = P
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2 * c, kernel_size=5, padding=2)

    def _normalize(self, z_hat):
        """功率归一化，确保输出保持 4D"""
        batch_size = z_hat.size(0)
        k = z_hat[0].numel()

        z_flat = z_hat.view(batch_size, -1)
        power = torch.sum(z_flat ** 2, dim=1, keepdim=True)

        scale = torch.sqrt(self.P * k / (power + 1e-8))
        scale = scale.view(batch_size, 1, 1, 1)

        return z_hat * scale

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self._normalize(x)
        return x


class _Decoder(nn.Module):
    """Image Decoder for JSCC"""

    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2 * c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2,
            padding=2, output_padding=1, activate=nn.Sigmoid())

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class _CSIAwareDecoder(nn.Module):
    """CSI-Aware Decoder with FiLM modulation"""

    def __init__(self, c=1, csi_dim=6):
        super(_CSIAwareDecoder, self).__init__()

        self.csi_dim = csi_dim

        self.csi_embed = nn.Sequential(
            nn.Linear(csi_dim, 64),
            nn.PReLU(),
            nn.Linear(64, 64)
        )

        self.film1 = nn.Linear(64, 32 * 2)
        self.film2 = nn.Linear(64, 32 * 2)
        self.film3 = nn.Linear(64, 32 * 2)
        self.film4 = nn.Linear(64, 16 * 2)

        self.tconv1 = _TransConvWithPReLU(
            in_channels=2 * c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2,
            padding=2, output_padding=1, activate=nn.Sigmoid())

    def _apply_film(self, x, film_params):
        """Apply FiLM modulation: y = gamma * x + beta"""
        batch_size = film_params.size(0)
        channels = x.size(1)
        film_dim = film_params.size(1)

        channels_from_film = film_dim // 2
        apply_ch = min(channels, channels_from_film)

        if apply_ch <= 0:
            return x

        gamma = film_params[:, :apply_ch]
        beta = film_params[:, apply_ch:2 * apply_ch]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        gamma = torch.sigmoid(gamma) * 2.0

        x_main = x[:, :apply_ch] * gamma + beta

        if apply_ch == channels:
            return x_main
        else:
            return torch.cat([x_main, x[:, apply_ch:]], dim=1)

    def forward(self, z, csi):
        csi_feat = self.csi_embed(csi)

        film1_params = self.film1(csi_feat)
        film2_params = self.film2(csi_feat)
        film3_params = self.film3(csi_feat)
        film4_params = self.film4(csi_feat)

        x = self.tconv1(z)
        x = self._apply_film(x, film1_params)

        x = self.tconv2(x)
        x = self._apply_film(x, film2_params)

        x = self.tconv3(x)
        x = self._apply_film(x, film3_params)

        x = self.tconv4(x)
        x = self._apply_film(x, film4_params)

        x = self.tconv5(x)

        return x


class CSICompressor(nn.Module):
    def __init__(self, csi_dim, hidden_dim=64, feedback_bits=32, use_quantization=True):
        super(CSICompressor, self).__init__()
        self.csi_dim = csi_dim
        self.feedback_bits = feedback_bits
        self.use_quantization = use_quantization

        self.fc1 = nn.Linear(csi_dim, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(hidden_dim, feedback_bits)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, csi):
        x = self.prelu1(self.bn1(self.fc1(csi)))
        x = self.prelu2(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x


class CSIDecompressor(nn.Module):
    def __init__(self, feedback_bits=32, hidden_dim=64, csi_dim=6):
        super(CSIDecompressor, self).__init__()
        self.feedback_bits = feedback_bits
        self.csi_dim = csi_dim

        self.fc1 = nn.Linear(feedback_bits, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(hidden_dim, csi_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, compressed_csi):
        x = self.prelu1(self.bn1(self.fc1(compressed_csi)))
        x = self.prelu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class CSIFeedbackModule(nn.Module):
    def __init__(self, csi_dim, feedback_bits=32, hidden_dim=64, use_quantization=True):
        super(CSIFeedbackModule, self).__init__()
        self.csi_dim = csi_dim
        self.feedback_bits = feedback_bits

        self.compressor = CSICompressor(csi_dim, hidden_dim, feedback_bits, use_quantization)
        self.decompressor = CSIDecompressor(feedback_bits, hidden_dim, csi_dim)

    def forward(self, csi):
        compressed = self.compressor(csi)
        reconstructed = self.decompressor(compressed)
        return compressed, reconstructed


class DeepJSCC(nn.Module):
    """Baseline DeepJSCC model"""

    def __init__(self, c, channel_type='AWGN', snr=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        self.channel = Channel(channel_type, snr) if snr is not None else None
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        if self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)


class DeepJSCCWithCSIFeedback(nn.Module):
    """DeepJSCC with CSI Feedback"""

    def __init__(self, c, channel_type='AWGN', snr=10, feedback_bits=32,
                 use_csi_aware_decoder=True, csi_loss_weight=0.1):
        super(DeepJSCCWithCSIFeedback, self).__init__()

        self.c = c
        self.channel_type = channel_type
        self.snr = snr
        self.feedback_bits = feedback_bits
        self.use_csi_aware_decoder = use_csi_aware_decoder

        self.encoder = _Encoder(c=c)
        self.channel = Channel(channel_type, snr)
        csi_dim = self.channel.get_csi_dim()

        self.csi_feedback = CSIFeedbackModule(
            csi_dim=csi_dim,
            feedback_bits=feedback_bits,
            hidden_dim=64,
            use_quantization=True
        )

        if use_csi_aware_decoder:
            self.decoder = _CSIAwareDecoder(c=c, csi_dim=csi_dim)
        else:
            self.decoder = _Decoder(c=c)

    def forward(self, x, return_intermediate=False):
        z = self.encoder(x)
        z_channel, csi_original = self.channel(z, return_csi=True)
        csi_compressed, csi_reconstructed = self.csi_feedback(csi_original)

        if self.use_csi_aware_decoder:
            x_hat = self.decoder(z_channel, csi_reconstructed)
        else:
            x_hat = self.decoder(z_channel)

        if return_intermediate:
            return x_hat, csi_original, csi_compressed, csi_reconstructed
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is not None:
            self.channel = Channel(channel_type, snr)
            self.snr = snr
            self.channel_type = channel_type


# =========================
# 数据集
# =========================
class KodakDataset(Dataset):
    def __init__(self, root):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp')
        files = []
        for e in exts:
            files.extend(glob(os.path.join(root, e)))
        if not files:
            raise RuntimeError(f"No images found in {root}")
        self.files = sorted(files)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), os.path.basename(self.files[idx])


# =========================
# 工具函数
# =========================
def infer_c_from_state_dict(sd):
    for k in sd:
        if 'encoder.conv5.conv.weight' in k:
            return sd[k].shape[0] // 2
    raise RuntimeError("无法推断 c")


def infer_feedback_bits_from_state_dict(sd):
    """从 state_dict 推断 feedback_bits"""
    # 尝试多种可能的键名模式
    possible_keys = [
        'csi_feedback.comp.6.weight',  # 旧版本
        'csi_feedback.compressor.fc3.weight',  # test_c=0.16.py 版本
        'csi_feedback.compressor.6.weight',  # nn.Sequential 版本
    ]

    for k in sd:
        for possible_key in possible_keys:
            if possible_key in k:
                return sd[k].shape[0]

    # 如果都没找到，尝试查找与 feedback_bits 相关的其他键
    for k in sd:
        if 'csi_feedback' in k and ('compressor' in k or 'decompressor' in k):
            if 'weight' in k and k.endswith('.weight'):
                # 如果是线性层的权重，检查维度
                weight = sd[k]
                if len(weight.shape) == 2:
                    # 输出维度
                    return weight.shape[0] if 'compressor' in k else 32

    # 默认返回32
    return 32


def load_state_dict_flexible(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def build_model(model_type, path, channel_type, snr, device):
    sd = load_state_dict_flexible(torch.load(path, map_location=device))
    c = infer_c_from_state_dict(sd)
    print(f"  推断的 c: {c}")
    if model_type == 'baseline':
        model = DeepJSCC(c=c, channel_type=channel_type, snr=snr)
    else:
        fb = infer_feedback_bits_from_state_dict(sd)
        print(f"  推断的 feedback_bits: {fb}")
        model = DeepJSCCWithCSIFeedback(
            c=c, channel_type=channel_type, snr=snr, feedback_bits=fb)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def set_model_snr(model, channel_type, snr):
    if hasattr(model, 'change_channel'):
        model.change_channel(channel_type=channel_type, snr=snr)


def evaluate_model(model, dataloader, snr_list, channel_type, device):
    results = {'snr': [], 'psnr': [], 'msssim': []}
    for snr in snr_list:
        set_model_snr(model, channel_type, snr)
        psnr_list, msssim_list = [], []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = model(images)
                mse = torch.mean((images * 255 - outputs * 255) ** 2)
                psnr_list.append((20 * torch.log10(255.0 / torch.sqrt(mse))).item() if mse > 0 else 50.0)
                if images.shape[2] >= 160 and images.shape[3] >= 160:
                    msssim_list.append(
                        ms_ssim(torch.clamp(outputs, 0, 1), torch.clamp(images, 0, 1), data_range=1.0).item())
                else:
                    msssim_list.append(0.0)
        results['snr'].append(snr)
        results['psnr'].append(np.mean(psnr_list))
        results['msssim'].append(np.mean(msssim_list))
        print(f"  SNR={snr:>3} dB | PSNR={results['psnr'][-1]:.3f} dB | MS-SSIM={results['msssim'][-1]:.4f}")
    return results


def plot_curves(all_results, model_configs, out_dir, channel_type, ratio):
    os.makedirs(out_dir, exist_ok=True)
    config_dict = {m['label']: m for m in model_configs}

    # PSNR 曲线
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in all_results.items():
        cfg = config_dict.get(name, {})
        ax.plot(res['snr'], res['psnr'], marker=cfg.get('marker', 'o'),
                linestyle=cfg.get('linestyle', '-'), linewidth=2, markersize=8,
                label=name, color=cfg.get('color'))
    ax.set_title(f'{channel_type} (R={ratio})', fontsize=16, fontweight='bold')
    ax.set_xlabel(r'SNR$_{test}$ (dB)', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kodak_psnr_curve.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "kodak_psnr_curve.pdf"))
    plt.close()

    # MS-SSIM 曲线
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in all_results.items():
        cfg = config_dict.get(name, {})
        ax.plot(res['snr'], res['msssim'], marker=cfg.get('marker', 'o'),
                linestyle=cfg.get('linestyle', '-'), linewidth=2, markersize=8,
                label=name, color=cfg.get('color'))
    ax.set_title(f'{channel_type} (R={ratio})', fontsize=16, fontweight='bold')
    ax.set_xlabel(r'SNR$_{test}$ (dB)', fontsize=14)
    ax.set_ylabel('MS-SSIM', fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kodak_msssim_curve.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "kodak_msssim_curve.pdf"))
    plt.close()
    print(f"\n曲线已保存到: {out_dir}")


def main():
    cfg = CONFIG
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = KodakDataset(cfg["kodak_root"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} images from {cfg['kodak_root']}")

    snr_list = list(range(cfg["snr_min"], cfg["snr_max"] + 1, cfg["snr_step"]))
    print(f"SNR range: {snr_list}")

    all_results = {}
    all_configs = []

    # 1. 先评估神经网络模型（JSCC）
    for m in cfg["models"]:
        print("\n" + "=" * 70)
        print(f"Evaluating: {m['label']}")
        print(f"  Path: {m['path']}")
        print("=" * 70)
        if not os.path.isfile(m["path"]):
            print(f"  [ERROR] 文件不存在: {m['path']}")
            continue
        try:
            model = build_model(m["type"], m["path"], cfg["channel"], snr_list[0], device)
            results = evaluate_model(model, dataloader, snr_list, cfg["channel"], device)
            all_results[m["label"]] = results
            all_configs.append(m)
        except Exception as e:
            print(f"  [ERROR] 加载或评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 2. 最后评估 JPEG+LDPC+QAM 方案
    for jcfg in cfg["jpeg_ldpc_configs"]:
        print("\n" + "=" * 70)
        print(f"Evaluating: {jcfg['label']}")
        print("=" * 70)
        results = evaluate_jpeg_ldpc_qam(dataloader, snr_list, jcfg, device)
        all_results[jcfg['label']] = results
        all_configs.append(jcfg)

    # 3. 绘制曲线
    if all_results:
        print("\n" + "=" * 70)
        print("所有模型评估完成，正在生成对比曲线图...")
        print("=" * 70)
        plot_curves(all_results, all_configs, cfg["out_dir"], cfg["channel"], cfg["ratio"])

        import json
        with open(os.path.join(cfg["out_dir"], "results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()