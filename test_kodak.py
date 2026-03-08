# -*- coding: utf-8 -*-
"""
Kodak 数据集原图测试脚本 (无需 resize)
支持 DeepJSCC 和 DeepJSCC+CSI 模型在不同 SNR 下的评估
"""

import os
from glob import glob

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim

# =========================
# 配置区
# =========================
CONFIG = {
    # Kodak 数据集路径
    "kodak_root": "./dataset/Kodak",

    "batch_size": 1,  # Kodak 原图测试建议用 1
    "device": "cuda:0",
    "channel": "AWGN",

    # SNR 扫描范围
    "snr_min": -5,
    "snr_max": 10,
    "snr_step": 3,

    # 结果输出目录
    "out_dir": "./eval_results",

    # 待评估的模型列表
    "models": [
        {
            "path": "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_fb32_10h59m40s_on_Nov_29_2025/epoch_447.pkl",
            "type": "csi",
            "label": "DeepJSCC_CSI_SNR4"
        },
        # 可以添加更多模型
        {
            "path": "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl",
            "type": "baseline",  # 或 "csi"
            "label": "DeepJSCC_SNR4"
        },
        {
            "path": "./out/checkpoints/CIFAR10_4_13.0_0.08_AWGN_fb32_01h19m52s_on_Nov_29_2025/epoch_449.pkl",
            "type": "csi",
            "label": "DeepJSCC_CSI_SNR13"
        },
        {
            "path": "./out/checkpoints/CIFAR10_4_13.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl",
            "type": "baseline",
            "label": "DeepJSCC_SNR13"
        },
        {
            "path": "./out/checkpoints/CIFAR10_4_19.0_0.08_AWGN_fb32_22h34m06s_on_Nov_28_2025/epoch_419.pkl",
            "type": "csi",
            "label": "DeepJSCC_CSI_SNR19"
        },
        {
            "path": "./out/checkpoints/CIFAR10_4_19.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl",
            "type": "baseline",
            "label": "DeepJSCC_SNR19"
        },

    ]
}


# =========================
# 修复后的模型组件
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
        # 始终按 4D 处理
        batch_size = z_hat.size(0)
        k = z_hat[0].numel()  # 单个样本的元素数量

        # 计算每个样本的功率
        z_flat = z_hat.view(batch_size, -1)
        power = torch.sum(z_flat ** 2, dim=1, keepdim=True)  # (B, 1)

        # 归一化
        scale = torch.sqrt(self.P * k / (power + 1e-8))  # (B, 1)
        scale = scale.view(batch_size, 1, 1, 1)  # 扩展到 4D

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

        # 使用 unsqueeze 确保正确的维度扩展
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
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
    """Kodak 数据集 (原图，无 resize)"""

    def __init__(self, root):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp')
        files = []
        for e in exts:
            files.extend(glob(os.path.join(root, e)))
        if len(files) == 0:
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
def image_normalization(norm_type):
    def _inner(tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0

    return _inner


def infer_c_from_state_dict(state_dict):
    """从 state_dict 推断 c"""
    for k in state_dict.keys():
        if 'encoder.conv5.conv.weight' in k:
            return state_dict[k].shape[0] // 2
    raise RuntimeError("无法推断 c")


def infer_feedback_bits_from_state_dict(state_dict):
    """从 state_dict 推断 feedback_bits"""
    for k in state_dict.keys():
        if 'csi_feedback.compressor.fc3.weight' in k:
            return state_dict[k].shape[0]
    return 32


def load_state_dict_flexible(state_dict):
    """处理 DataParallel 保存的权重"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def build_model(model_type, state_dict_path, channel_type, snr, device):
    """构建并加载模型"""
    state_dict = torch.load(state_dict_path, map_location=device)
    state_dict = load_state_dict_flexible(state_dict)

    c = infer_c_from_state_dict(state_dict)
    print(f"  推断的 c: {c}")

    if model_type == 'baseline':
        model = DeepJSCC(c=c, channel_type=channel_type, snr=snr)
    elif model_type == 'csi':
        feedback_bits = infer_feedback_bits_from_state_dict(state_dict)
        print(f"  推断的 feedback_bits: {feedback_bits}")
        model = DeepJSCCWithCSIFeedback(
            c=c, channel_type=channel_type, snr=snr,
            feedback_bits=feedback_bits, use_csi_aware_decoder=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_psnr(img1, img2, max_val=255.0):
    """计算 PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()


def set_model_snr(model, channel_type, snr):
    """设置模型 SNR"""
    if hasattr(model, 'change_channel'):
        model.change_channel(channel_type=channel_type, snr=snr)
    elif hasattr(model, 'channel'):
        model.channel.channel_type = channel_type
        model.channel.snr = snr


def evaluate_model(model, dataloader, snr_list, channel_type, device):
    """评估模型"""
    img_norm = image_normalization
    results = {'snr': [], 'psnr': [], 'msssim': []}

    for snr in snr_list:
        set_model_snr(model, channel_type, snr)

        psnr_list = []
        msssim_list = []

        with torch.no_grad():
            for images, names in dataloader:
                images = images.to(device)

                # 前向传播
                outputs = model(images)

                # 计算 PSNR
                outputs_den = img_norm('denormalization')(outputs)
                images_den = img_norm('denormalization')(images)
                psnr_val = compute_psnr(outputs_den, images_den, max_val=255.0)
                psnr_list.append(psnr_val)

                # 计算 MS-SSIM
                outputs_clamp = torch.clamp(outputs, 0.0, 1.0)
                images_clamp = torch.clamp(images, 0.0, 1.0)

                # MS-SSIM 需要至少 160x160 的图像
                if images.shape[2] >= 160 and images.shape[3] >= 160:
                    msssim_val = ms_ssim(outputs_clamp, images_clamp, data_range=1.0, size_average=True).item()
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


def plot_curves(all_results, out_dir):
    """绘制曲线"""
    os.makedirs(out_dir, exist_ok=True)

    # PSNR 曲线
    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(res['snr'], res['psnr'], marker='o', linewidth=2, markersize=6, label=name)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("Kodak Dataset - PSNR vs SNR", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kodak_psnr_curve.png"), dpi=300)
    plt.close()

    # MS-SSIM 曲线
    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(res['snr'], res['msssim'], marker='s', linewidth=2, markersize=6, label=name)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("MS-SSIM", fontsize=12)
    plt.title("Kodak Dataset - MS-SSIM vs SNR", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kodak_msssim_curve.png"), dpi=300)
    plt.close()

    print(f"\n曲线已保存到: {out_dir}")


def main():
    cfg = CONFIG
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    dataset = KodakDataset(cfg["kodak_root"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} images from {cfg['kodak_root']}")

    # SNR 列表
    snr_list = list(range(cfg["snr_min"], cfg["snr_max"] + 1, cfg["snr_step"]))
    print(f"SNR range: {snr_list}")

    all_results = {}

    # 评估每个模型
    for m in cfg["models"]:
        model_path = m["path"]
        model_type = m["type"]
        label = m.get("label", os.path.basename(model_path))

        print("\n" + "=" * 70)
        print(f"Evaluating: {label}")
        print(f"  Path: {model_path}")
        print(f"  Type: {model_type}")
        print("=" * 70)

        if not os.path.isfile(model_path):
            print(f"  [ERROR] 文件不存在: {model_path}")
            continue

        try:
            model = build_model(model_type, model_path, cfg["channel"], snr_list[0], device)
            results = evaluate_model(model, dataloader, snr_list, cfg["channel"], device)
            all_results[label] = results
        except Exception as e:
            print(f"  [ERROR] 加载或评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 绘制曲线
    if all_results:
        plot_curves(all_results, cfg["out_dir"])

        # 保存数值结果
        import json
        with open(os.path.join(cfg["out_dir"], "results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"数值结果已保存到: {os.path.join(cfg['out_dir'], 'results.json')}")
    else:
        print("\n没有成功评估任何模型")


if __name__ == "__main__":
    main()