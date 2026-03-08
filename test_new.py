# -*- coding: utf-8 -*-
"""
DeepJSCC 与 JPEG+LDPC+QAM 数字传输方案对比测试脚本
包含了 "悬崖效应" (Cliff Effect) 的演示
"""

import os
import io
import datetime
from glob import glob
import json
import traceback

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pytorch_msssim import ms_ssim

# 引入 pyldpc
try:
    from pyldpc import make_ldpc, encode, decode
except ImportError:
    print("请先安装 pyldpc: pip install pyldpc")
    exit(1)

# =========================
# 配置区
# =========================
CONFIG = {
    # Kodak 数据集路径
    "kodak_root": "./dataset/Kodak",

    "batch_size": 1,
    "device": "cuda:0",
    "channel": "AWGN",
    "ratio": "1/6",  # 压缩率，用于图表标题

    # SNR 扫描范围 (包含低信噪比以体现悬崖效应)
    "snr_min": 0,
    "snr_max": 20,
    "snr_step": 2,

    # 结果输出目录
    "out_dir": "./eval_results_comparison",

    # 待评估的模型列表
    "models": [
        # --- DeepJSCC Models (保留你原有的) ---
        {
            "path": "./out/checkpoints/CIFAR10_8_1.0_0.17_AWGN_fb32_22h33m02s_on_Nov_28_2025/epoch_405.pkl",
            "type": "csi",
            "label": "JSCC+CM SNR$_{train}$=1dB",
            "color": "red",
            "marker": "v",
            "linestyle": "-"
        },
        {
            "path": "./out/checkpoints/CIFAR10_8_4.0_0.17_AWGN_fb32_03h00m05s_on_Nov_28_2025/epoch_990.pkl",
            "type": "csi",
            "label": "JSCC+CM SNR$_{train}$=4dB",
            "color": "blue",
            "marker": "v",
            "linestyle": "-"
        },
        # --- JPEG Benchmark (新增) ---
        {
            "type": "jpeg",
            "label": "JPEG+LDPC (1/2 Rate)",
            "color": "black",
            "marker": "o",
            "linestyle": "--",
            "target_bpp": 0.5,  # 设定目标BPP，模拟一定的压缩率
            "qam_order": 16,  # QAM16
            "ldpc_n": 1440,  # LDPC码长
            "ldpc_dv": 2,
            "ldpc_dc": 3
        }
    ]
}


# =========================
# JPEG + LDPC + QAM 系统仿真类 (移植自 11dB_jpeg...py)
# =========================
class JPEG_LDPC_System:
    def __init__(self, target_bpp=2, qam_order=16, n=960, d_v=2, d_c=3, snr=10):
        self.target_bpp = target_bpp
        self.qam_order = qam_order
        self.snr = snr

        # 初始化 LDPC 矩阵 (只做一次以节省时间)
        print(f"Initializing LDPC Matrices (n={n}, dv={d_v}, dc={d_c})...")
        self.H, self.G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
        self.n = n
        self.k = self.G.shape[1]
        print("LDPC Initialized.")

    def set_snr(self, snr):
        self.snr = snr

    def pillow_encode(self, img, fmt='JPEG', quality=10):
        buffer = io.BytesIO()
        img.save(buffer, format=fmt, quality=quality)
        size = buffer.tell()
        bpp = size * 8.0 / (img.size[0] * img.size[1])
        return bpp, buffer

    def find_closest_bpp(self, img):
        lower = 0
        upper = 100
        prev_mid = upper
        best_buffer = None

        # 二分查找最佳 Quality
        for i in range(10):
            mid = (upper - lower) / 2 + lower
            if int(mid) == int(prev_mid):
                break
            quality = int(mid)
            if quality == 0: quality = 1
            bpp, buffer = self.pillow_encode(img, quality=quality)

            best_buffer = buffer  # 暂存
            prev_mid = mid

            if bpp > self.target_bpp:
                upper = mid - 1
            else:
                lower = mid

        if best_buffer is None:
            _, best_buffer = self.pillow_encode(img, quality=50)

        return best_buffer.getvalue()

    def qam_modulate(self, bits):
        qam_order = self.qam_order
        bits_per_symbol = int(np.log2(qam_order))

        # Padding if necessary
        pad_len = (bits_per_symbol - (len(bits) % bits_per_symbol)) % bits_per_symbol
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

        bit_groups = bits.reshape(-1, bits_per_symbol)
        symbols = np.packbits(bit_groups, axis=-1, bitorder='little').flatten()

        m = int(np.sqrt(qam_order))
        x = np.arange(m) - (m - 1) / 2
        y = np.arange(m) - (m - 1) / 2
        constellation = np.array([complex(a, b) for a in x for b in y])

        modulated_signal = constellation[symbols]
        return modulated_signal, pad_len

    def awgn_channel(self, signal):
        snr_db = self.snr
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
                    np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape))
        return signal + noise

    def qam_demodulate(self, received_signal, pad_len):
        qam_order = self.qam_order
        m = int(np.sqrt(qam_order))
        x = np.arange(m) - (m - 1) / 2
        y = np.arange(m) - (m - 1) / 2
        constellation = np.array([complex(a, b) for a in x for b in y])

        distances = np.abs(received_signal.reshape(-1, 1) - constellation.reshape(1, -1))
        nearest_points = np.argmin(distances, axis=1)

        bits_per_symbol = int(np.log2(qam_order))
        demodulated_bits = np.unpackbits(nearest_points.astype(np.uint8), bitorder='little')[-bits_per_symbol:]

        # Flatten and remove padding
        bits = demodulated_bits.flatten()
        if pad_len > 0:
            bits = bits[:-pad_len]
        return bits

    def process_single_image(self, img_tensor):
        """
        输入: Tensor (C, H, W) range [0, 1]
        输出: Tensor (C, H, W) range [0, 1]
        """
        # 1. Tensor -> PIL
        pil_img = ToPILImage()(img_tensor)
        width, height = pil_img.size

        # 2. JPEG Compression (Source Coding) -> Bytes
        jpeg_bytes = self.find_closest_bpp(pil_img)

        # 3. Bytes -> Bits Array
        # 优化：直接在内存操作，不写txt
        byte_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        # unpackbits 得到的是 big-endian (high bit first), 原脚本使用的是 bin().replace... 逻辑
        # 原脚本逻辑: bin(j) -> '0b101' -> rjust 8 -> '00000101'. 即 high bit first.
        # np.unpackbits 默认就是 high bit first.
        bits = np.unpackbits(byte_arr)

        # 4. LDPC Encoding (Channel Coding)
        # LDPC 需要分块编码，因为 n 是固定的
        k = self.k
        n = self.n

        # Pad bits to multiple of k
        total_bits = len(bits)
        pad_bits_len = (k - (total_bits % k)) % k
        if pad_bits_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_bits_len, dtype=int)])

        num_blocks = len(bits) // k
        encoded_blocks = []

        # 注意: pyldpc encode 比较慢，这里逐块进行
        # snr used in encode is usually for LLR init, setting high for pure encoding
        for i in range(num_blocks):
            block = bits[i * k: (i + 1) * k]
            coded = encode(self.G, block, snr=100)
            encoded_blocks.append(coded)

        encoded_bits = np.concatenate(encoded_blocks)

        # 5. QAM Mod -> AWGN -> QAM Demod
        mod_sig, qam_pad = self.qam_modulate(encoded_bits)
        rx_sig = self.awgn_channel(mod_sig)
        demod_bits_raw = self.qam_demodulate(rx_sig, qam_pad)

        # 确保长度匹配 (截断可能多出的解调位)
        if len(demod_bits_raw) > len(encoded_bits):
            demod_bits_raw = demod_bits_raw[:len(encoded_bits)]

        # 6. LDPC Decoding
        decoded_bits_all = []

        # Reshape to (num_blocks, n)
        # QAM 解调后的比特可能因为噪声 flip
        # pyldpc decode 需要 LLR.
        # approximate LLR: 0 -> 1, 1 -> -1 (or scaled by SNR)
        # Simplified: 1 - 2*bit

        rx_blocks = demod_bits_raw.reshape(num_blocks, n)

        try:
            for i in range(num_blocks):
                # 这里的 maxiter 决定了解码能力和速度
                # maxiter=20 is faster but weaker than 50
                block_input = 1 - 2 * rx_blocks[i]
                decoded_block = decode(self.H, block_input, snr=self.snr, maxiter=20)
                decoded_bits_all.append(decoded_block)

            recovered_bits = np.concatenate(decoded_bits_all)

            # Remove Padding
            if pad_bits_len > 0:
                recovered_bits = recovered_bits[:-pad_bits_len]

            # 7. Bits -> Bytes
            recovered_bytes = np.packbits(recovered_bits)

            # 8. Bytes -> Image (Check for Cliff Effect)
            img_io = io.BytesIO(recovered_bytes.tobytes())

            try:
                rec_img = Image.open(img_io).convert('RGB')
                # 即使 open 成功，尺寸也可能不对，校验一下
                if rec_img.size != (width, height):
                    raise IOError("Size mismatch")
                return ToTensor()(rec_img)

            except (IOError, OSError, ValueError):
                raise IOError("Image corrupted")

        except Exception:
            # Cliff Effect: Return Random Noise
            # print("  [Cliff Effect] Decoding failed, returning noise.")
            return torch.rand(3, height, width)


# =========================
# 模型组件 (DeepJSCC 部分保持不变)
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

    def forward(self, x): return self.prelu(self.conv(x))


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=None, padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate if activate is not None else nn.PReLU()
        if isinstance(self.activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        return self.activate(self.transconv(x))


class _Encoder(nn.Module):
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
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        self.tconv1 = _TransConvWithPReLU(in_channels=2 * c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.tconv5 = _TransConvWithPReLU(in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2,
                                          output_padding=1, activate=nn.Sigmoid())

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class _CSIAwareDecoder(nn.Module):
    def __init__(self, c=1, csi_dim=6):
        super(_CSIAwareDecoder, self).__init__()
        self.csi_dim = csi_dim
        self.csi_embed = nn.Sequential(nn.Linear(csi_dim, 64), nn.PReLU(), nn.Linear(64, 64))
        self.film1 = nn.Linear(64, 32 * 2)
        self.film2 = nn.Linear(64, 32 * 2)
        self.film3 = nn.Linear(64, 32 * 2)
        self.film4 = nn.Linear(64, 16 * 2)
        self.tconv1 = _TransConvWithPReLU(in_channels=2 * c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.tconv5 = _TransConvWithPReLU(in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2,
                                          output_padding=1, activate=nn.Sigmoid())

    def _apply_film(self, x, film_params):
        batch_size = film_params.size(0)
        channels = x.size(1)
        film_dim = film_params.size(1)
        channels_from_film = film_dim // 2
        apply_ch = min(channels, channels_from_film)
        if apply_ch <= 0: return x
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


class CSIFeedbackModule(nn.Module):
    def __init__(self, csi_dim, feedback_bits=32, hidden_dim=64, use_quantization=True):
        super(CSIFeedbackModule, self).__init__()
        self.csi_dim = csi_dim
        self.feedback_bits = feedback_bits
        self.compressor = nn.Sequential(
            nn.Linear(csi_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, feedback_bits), nn.Sigmoid()
        )
        self.decompressor = nn.Sequential(
            nn.Linear(feedback_bits, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, csi_dim)
        )

    def forward(self, csi):
        compressed = self.compressor(csi)
        reconstructed = self.decompressor(compressed)
        return compressed, reconstructed


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        self.channel = Channel(channel_type, snr) if snr is not None else None
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        if self.channel is not None: z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)


class DeepJSCCWithCSIFeedback(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=10, feedback_bits=32, use_csi_aware_decoder=True):
        super(DeepJSCCWithCSIFeedback, self).__init__()
        self.c = c
        self.channel_type = channel_type
        self.snr = snr
        self.encoder = _Encoder(c=c)
        self.channel = Channel(channel_type, snr)
        csi_dim = self.channel.get_csi_dim()
        self.csi_feedback = CSIFeedbackModule(csi_dim=csi_dim, feedback_bits=feedback_bits)
        self.decoder = _CSIAwareDecoder(c=c, csi_dim=csi_dim) if use_csi_aware_decoder else _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        z_channel, csi_original = self.channel(z, return_csi=True)
        csi_compressed, csi_reconstructed = self.csi_feedback(csi_original)
        x_hat = self.decoder(z_channel, csi_reconstructed)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is not None:
            self.channel = Channel(channel_type, snr)
            self.snr = snr


# =========================
# 数据集
# =========================
class KodakDataset(Dataset):
    def __init__(self, root):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp')
        files = []
        for e in exts: files.extend(glob(os.path.join(root, e)))
        if len(files) == 0: raise RuntimeError(f"No images found in {root}")
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
def image_normalization(norm_type):
    def _inner(tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0

    return _inner


def infer_c_from_state_dict(state_dict):
    for k in state_dict.keys():
        if 'encoder.conv5.conv.weight' in k: return state_dict[k].shape[0] // 2
    return 16  # Default fallback


def load_state_dict_flexible(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def build_model(model_cfg, channel_type, snr, device):
    model_type = model_cfg['type']

    # === 处理 JPEG 模型 ===
    if model_type == 'jpeg':
        # 返回一个封装好的 JPEG Simulator 对象
        return JPEG_LDPC_System(
            target_bpp=model_cfg.get('target_bpp', 1.0),
            qam_order=model_cfg.get('qam_order', 16),
            n=model_cfg.get('ldpc_n', 960),
            d_v=model_cfg.get('ldpc_dv', 2),
            d_c=model_cfg.get('ldpc_dc', 3),
            snr=snr
        )

    # === 处理 JSCC 模型 ===
    state_dict_path = model_cfg['path']
    state_dict = torch.load(state_dict_path, map_location=device)
    state_dict = load_state_dict_flexible(state_dict)

    c = infer_c_from_state_dict(state_dict)

    if model_type == 'baseline':
        model = DeepJSCC(c=c, channel_type=channel_type, snr=snr)
    elif model_type == 'csi':
        model = DeepJSCCWithCSIFeedback(c=c, channel_type=channel_type, snr=snr, feedback_bits=32)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_psnr(img1, img2, max_val=255.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()


def evaluate_model(model, dataloader, snr_list, channel_type, device, model_type='jscc'):
    """评估模型，支持 JSCC (GPU) 和 JPEG (CPU simulation)"""
    img_norm = image_normalization
    results = {'snr': [], 'psnr': [], 'msssim': []}

    is_jpeg = (model_type == 'jpeg')

    for snr in snr_list:
        psnr_list = []
        msssim_list = []

        # 设置 SNR
        if is_jpeg:
            model.set_snr(snr)
        else:
            if hasattr(model, 'change_channel'):
                model.change_channel(channel_type=channel_type, snr=snr)
            elif hasattr(model, 'channel'):
                model.channel.channel_type = channel_type
                model.channel.snr = snr

        # --- 推理循环 ---
        if not is_jpeg:
            # JSCC: Batch processing on GPU
            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(device)
                    outputs = model(images)

                    outputs_den = img_norm('denormalization')(outputs)
                    images_den = img_norm('denormalization')(images)
                    psnr_val = compute_psnr(outputs_den, images_den, max_val=255.0)
                    psnr_list.append(psnr_val)

                    outputs_clamp = torch.clamp(outputs, 0.0, 1.0)
                    images_clamp = torch.clamp(images, 0.0, 1.0)
                    if images.shape[2] >= 160 and images.shape[3] >= 160:
                        msssim_val = ms_ssim(outputs_clamp, images_clamp, data_range=1.0, size_average=True).item()
                    else:
                        msssim_val = 0.0
                    msssim_list.append(msssim_val)
        else:
            # JPEG: Per-image processing on CPU (Simulated)
            # 因为 pyldpc 和 PIL 是 CPU 操作
            for images, _ in dataloader:
                for i in range(images.size(0)):
                    img = images[i]  # (C, H, W)

                    # Process image through JPEG+LDPC+QAM pipeline
                    # 这里会非常慢，所以通常测试集图片少一点比较好
                    output_img = model.process_single_image(img)

                    # Calculate Metrics
                    # Move to same device/type if needed, but here simple CPU calc is fine
                    psnr_val = compute_psnr(output_img * 255.0, img * 255.0, max_val=255.0)
                    psnr_list.append(psnr_val)

                    # For SSIM, use batch dimension
                    out_b = output_img.unsqueeze(0)
                    img_b = img.unsqueeze(0)
                    if img.shape[1] >= 160 and img.shape[2] >= 160:
                        msssim_val = ms_ssim(out_b, img_b, data_range=1.0, size_average=True).item()
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


def plot_curves(all_results, model_configs, out_dir, channel_type, ratio):
    os.makedirs(out_dir, exist_ok=True)
    config_dict = {m['label']: m for m in model_configs}

    # Plot PSNR
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in all_results.items():
        cfg = config_dict.get(name, {})
        ax.plot(res['snr'], res['psnr'], marker=cfg.get('marker', 'o'), linestyle=cfg.get('linestyle', '-'),
                linewidth=2, markersize=8, label=name, color=cfg.get('color'))
    ax.set_title(f'{channel_type} (Compression Ratio ~ {ratio})', fontsize=16)
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, "comparison_psnr.png"))
    plt.close()

    print(f"Curves saved to {out_dir}")


def main():
    cfg = CONFIG
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    dataset = KodakDataset(cfg["kodak_root"])
    # Batch size 必须为 1 如果要跑 JPEG (因为图片尺寸不一，且 pyldpc 逐个处理)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    snr_list = list(range(cfg["snr_min"], cfg["snr_max"] + 1, cfg["snr_step"]))
    all_results = {}
    successful_models = []

    for m in cfg["models"]:
        print(f"\nEvaluating: {m['label']} ({m['type']})")
        try:
            model = build_model(m, cfg["channel"], snr_list[0], device)
            results = evaluate_model(model, dataloader, snr_list, cfg["channel"], device, model_type=m['type'])
            all_results[m['label']] = results
            successful_models.append(m)
        except Exception as e:
            print(f"Error evaluating {m['label']}: {e}")
            traceback.print_exc()

    if all_results:
        plot_curves(all_results, successful_models, cfg["out_dir"], cfg["channel"], cfg["ratio"])


if __name__ == "__main__":
    main()