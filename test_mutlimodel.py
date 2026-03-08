# -*- coding: utf-8 -*-
"""
多模型 DeepJSCC / DeepJSCC+CSI 在 Kodak 上的 SNR-PSNR / SNR-MS-SSIM 曲线评估脚本
使用方式：
    1. 修改下方 CONFIG 中的路径和模型列表
    2. 直接运行：python eval_kodak_multi_models.py
"""

import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim

from model import DeepJSCC
from model_csi import DeepJSCCWithCSIFeedback
from utils import image_normalization as image_normalization_baseline
from utils_csi import image_normalization as image_normalization_csi
from channel_csi import Channel as CSIchannel

# =========================
# 配置区：直接在这里改
# =========================
CONFIG = {
    # Kodak 数据集路径：文件夹内是若干 PNG/JPG/TIF 等图像
    "kodak_root": "./dataset/Kodak",

    # 评估时的输入尺寸：训练时如果是 128x128，就填 128；如果用原始尺寸，就填 0 或负数
    "image_size": 128,

    "batch_size": 1,
    "device": "cuda:0",       # 没有 GPU 就改成 "cpu"
    "channel": "AWGN",        # "AWGN" 或 "Rayleigh"

    # SNR 扫描范围
    "snr_min": -10,
    "snr_max": 10,
    "snr_step": 3,

    # 结果输出目录
    "out_dir": "./eval_results",

    # 待评估的模型列表：可以混合 baseline 和 csi
    #  type: # "baseline" -> model.py 里的 DeepJSCC
    #       "csi"      -> model_csi.py 里的 DeepJSCCWithCSIFeedback
    "models": [
        # 示例，按实际路径修改
        {
            "path": "./out/checkpoints/CIFAR10_4_1.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl",
            "type": "baseline",
            "label": "DeepJSCC_noCSI"
        },
        {
            "path": "./out/checkpoints/CIFAR10_8_19.0_0.17_AWGN_fb32_10h52m14s_on_Nov_30_2025/epoch_49.pkl",
            "type": "csi",
            "label": "DeepJSCC_CSI"
        },
    ]
}


# =========================
# 数据集：Kodak
# =========================
class KodakDataset(Dataset):
    """
    简单的 Kodak 数据集加载器。
    root 目录下是若干 PNG/JPG/JPEG/TIF/BMP 图片。
    """
    def __init__(self, root, image_size=None):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp')
        files = []
        for e in exts:
            files.extend(glob(os.path.join(root, e)))
        if len(files) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.files = sorted(files)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)  # [0,1]
        return img, 0


# =========================
# 工具函数
# =========================
def infer_c_from_state_dict(state_dict):
    """
    从 state_dict 里自动推断通道数 c
    通过 encoder.conv5.conv.weight 的 out_channels = 2c 反推 c
    适用于 model.py 和 model_csi.py 中的结构
    """
    key_candidates = [
        'encoder.conv5.conv.weight',
        'module.encoder.conv5.conv.weight'
    ]
    for k in state_dict.keys():
        for pat in key_candidates:
            if pat in k:
                out_channels = state_dict[k].shape[0]
                return out_channels // 2
    raise RuntimeError("无法从 state_dict 推断 c，请检查模型结构。")

def infer_feedback_bits_from_state_dict(state_dict):
    """
    从 state_dict 推断 feedback_bits
    通过检查 CSI 解码器的输出维度
    """
    # 查找 CSI 解码器最后一层的输出维度
    for key in state_dict.keys():
        if 'csi_decoder.2.weight' in key and len(state_dict[key].shape) == 2:
            output_dim = state_dict[key].shape[0]
            # CSI解码器输出的是FiLM参数(gamma和beta)，每个通道需要2个参数
            # 所以feedback_bits应该是output_dim的一半
            return output_dim // 2
        elif 'csi_decoder.fc.weight' in key and len(state_dict[key].shape) == 2:
            output_dim = state_dict[key].shape[0]
            return output_dim // 2
    return 32  # 默认值
def build_model(model_type, state_dict_path, channel_type, snr, device):
    """
    根据 model_type 构建模型并加载权重。
    model_type: 'baseline' -> model.DeepJSCC
                'csi'      -> model_csi.DeepJSCCWithCSIFeedback
    """
    state_dict = torch.load(state_dict_path, map_location=device)
    c = infer_c_from_state_dict(state_dict)

    if model_type == 'baseline':
        model = DeepJSCC(c=c, channel_type=channel_type, snr=snr)
        model.load_state_dict(state_dict)
    elif model_type == 'csi':
        feedback_bits = infer_feedback_bits_from_state_dict(state_dict)
        print(f"推断的 feedback_bits: {feedback_bits}")
        model = DeepJSCCWithCSIFeedback(
            c=c,
            channel_type=channel_type,
            snr=snr,
            feedback_bits=feedback_bits,
            use_csi_aware_decoder=True,
            csi_loss_weight=0.1
        )
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Unknown model_type")

    model.to(device)
    model.eval()
    return model


def compute_psnr(img1, img2, max_val=255.0):
    """
    计算 PSNR，img1 / img2 为 [0,max_val] 区间张量
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def set_model_snr(model, model_type, channel_type, snr):
    """
    统一设置模型内部信道的 SNR。
    baseline: 使用 DeepJSCC 的 change_channel()
    csi:      直接改 model.channel.snr（channel_csi.Channel）
    """
    if hasattr(model, "change_channel"):
        # model.py 里的 DeepJSCC 有 change_channel()
        model.change_channel(channel_type=channel_type, snr=snr)
    else:
        # 对于 DeepJSCCWithCSIFeedback，没有 change_channel，只能改 channel 对象
        if hasattr(model, "channel"):
            # 更新模型自身记录
            if hasattr(model, "channel_type"):
                model.channel_type = channel_type
            if hasattr(model, "snr"):
                model.snr = snr
            # 更新通道模块
            if isinstance(model.channel, CSIchannel):
                model.channel.channel_type = channel_type
                model.channel.snr = snr
            else:
                # 如果将来结构有变，可以在这里扩展
                model.channel = CSIchannel(channel_type, snr)
        else:
            raise RuntimeError("该模型没有 channel 属性，无法修改 SNR")


def evaluate_one_model(model, model_type, dataloader, snr_list, channel_type, device):
    """
    对单个模型在一组 SNR 上进行评估。
    返回：snr_list, psnr_list, ms_ssim_list
    """
    if model_type == 'baseline':
        img_norm = image_normalization_baseline
    else:
        img_norm = image_normalization_csi

    snrs = []
    avg_psnrs = []
    avg_msssims = []

    for snr in snr_list:
        set_model_snr(model, model_type, channel_type, snr)

        psnr_sum = 0.0
        msssim_sum = 0.0
        n_images = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)  # [0,1]

                # 前向传播
                # baseline: DeepJSCC(x) -> x_hat
                # csi:      DeepJSCCWithCSIFeedback(x) -> x_hat
                outputs = model(images)

                # 恢复到 [0,255] 作 PSNR
                outputs_den = img_norm('denormalization')(outputs)
                images_den = img_norm('denormalization')(images)
                psnr_val = compute_psnr(outputs_den, images_den, max_val=255.0)

                # MS-SSIM 在 [0,1] 上计算
                outputs_01 = torch.clamp(outputs, 0.0, 1.0)
                images_01 = torch.clamp(images, 0.0, 1.0)
                msssim_val = ms_ssim(
                    outputs_01, images_01,
                    data_range=1.0,
                    size_average=True
                ).item()

                b = images.size(0)
                psnr_sum += psnr_val * b
                msssim_sum += msssim_val * b
                n_images += b

        snrs.append(snr)
        avg_psnrs.append(psnr_sum / n_images)
        avg_msssims.append(msssim_sum / n_images)
        print(f"SNR={snr:>3} dB | PSNR={avg_psnrs[-1]:.3f} dB | MS-SSIM={avg_msssims[-1]:.4f}")

    return snrs, avg_psnrs, avg_msssims


def plot_curves(results, out_dir, title_prefix="DeepJSCC_Kodak"):
    """
    results: dict
        key: 模型 label
        val: dict{'snr':..., 'psnr':..., 'msssim':...}
    """
    os.makedirs(out_dir, exist_ok=True)

    # PSNR 曲线
    plt.figure()
    for name, res in results.items():
        plt.plot(res['snr'], res['psnr'], marker='o', label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average PSNR (dB)")
    plt.title(f"{title_prefix} - PSNR")
    plt.grid(True)
    plt.legend()
    psnr_path = os.path.join(out_dir, "kodak_psnr_curve.png")
    plt.savefig(psnr_path, bbox_inches="tight", dpi=300)

    # MS-SSIM 曲线
    plt.figure()
    for name, res in results.items():
        plt.plot(res['snr'], res['msssim'], marker='o', label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average MS-SSIM")
    plt.title(f"{title_prefix} - MS-SSIM")
    plt.grid(True)
    plt.legend()
    msssim_path = os.path.join(out_dir, "kodak_msssim_curve.png")
    plt.savefig(msssim_path, bbox_inches="tight", dpi=300)

    print(f"保存曲线：\n  {psnr_path}\n  {msssim_path}")


def main():
    cfg = CONFIG
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    dataset = KodakDataset(cfg["kodak_root"])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0
    )
    print(f"Loaded Kodak dataset from {cfg['kodak_root']}, "
          f"{len(dataset)} images.")

    # SNR 列表
    snr_list = list(range(cfg["snr_min"], cfg["snr_max"] + 1, cfg["snr_step"]))
    print(f"Evaluate SNRs: {snr_list}")

    results = {}

    # 遍历多个模型（可以同时包含 baseline 和 csi）
    for m in cfg["models"]:
        model_path = m["path"]
        model_type = m["type"]
        label = m.get("label", os.path.basename(model_path))

        print("\n" + "=" * 80)
        print(f"Evaluating model: {label}")
        print(f"  path = {model_path}")
        print(f"  type = {model_type}")
        print("=" * 80)

        if not os.path.isfile(model_path):
            print(f"  [WARN] 模型文件不存在：{model_path}，跳过该模型。")
            continue

        model = build_model(
            model_type=model_type,
            state_dict_path=model_path,
            channel_type=cfg["channel"],
            snr=snr_list[0],
            device=device
        )

        snrs, psnrs, msssims = evaluate_one_model(
            model=model,
            model_type=model_type,
            dataloader=dataloader,
            snr_list=snr_list,
            channel_type=cfg["channel"],
            device=device
        )

        results[label] = {
            "snr": snrs,
            "psnr": psnrs,
            "msssim": msssims
        }

    plot_curves(results, cfg["out_dir"])


if __name__ == "__main__":
    main()
